#include <Arduino.h>
#include <SPI.h>
#include <lmic.h>
#include <hal/hal.h>
#include <Adafruit_ISM330DHCX.h>
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "autoencoder_model.h"

// ─── OTAA Keys ─────────────────────────────────────────────────────────────────
static const u1_t PROGMEM APPEUI[8]  = {0x78,0xDE,0xFE,0x95,0x46,0x14,0xA8,0x68};
static const u1_t PROGMEM DEVEUI[8]  = {0x2B,0x09,0x17,0x87,0x97,0xC7,0x3D,0x9A};
static const u1_t PROGMEM APPKEY[16] = {
  0xB8,0x46,0x92,0x72,0x71,0xC1,0x00,0xBE,
  0x94,0x32,0xC9,0xC2,0xA3,0xF7,0x5C,0x0F
};
void os_getArtEui(u1_t* buf){ memcpy_P(buf, APPEUI, 8); }
void os_getDevEui(u1_t* buf){ memcpy_P(buf, DEVEUI, 8); }
void os_getDevKey(u1_t* buf){ memcpy_P(buf, APPKEY,16); }

// ─── Pin Mapping ───────────────────────────────────────────────────────────────
const lmic_pinmap lmic_pins = {
  .nss  = 33, .rxtx = LMIC_UNUSED_PIN, .rst = 15,
  .dio  = {27, 14},
};

// ─── Globals ───────────────────────────────────────────────────────────────────
volatile bool hasJoined = false;
Adafruit_ISM330DHCX accel;
static osjob_t sendjob;

// Buffers for 100 samples on each axis
const unsigned long SAMPLE_INTERVAL_US = 10000; // 100 Hz
unsigned long lastSampleTime = 0;
float raw_x[100], raw_y[100], raw_z[100];
float denoised_x[100], denoised_y[100], denoised_z[100];
int buf_index = 0;

// TFLite globals
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter*      error_reporter = nullptr;
const tflite::Model*        model          = nullptr;
tflite::MicroInterpreter*   interpreter    = nullptr;
TfLiteTensor*               tensor_in      = nullptr;
TfLiteTensor*               tensor_out     = nullptr;
constexpr int               kArenaSize = 16 * 1024;
static uint8_t              tensor_arena[kArenaSize];

// ─── Function Declarations ─────────────────────────────────────────────────────
void onEvent(ev_t ev);
void setupAccelerometer();
void TfliteSetup();
void doInference(const float* src, float* dst);
void sendFirstXYZ();

// ─── LMIC Event Handler ───────────────────────────────────────────────────────
void onEvent(ev_t ev) {
  Serial.print(os_getTime()); Serial.print(": ");
  switch (ev) {
    case EV_JOINING:   Serial.println(F("EV_JOINING")); break;
    case EV_JOINED:
      Serial.println(F("EV_JOINED"));
      hasJoined = true;
      LMIC_setLinkCheckMode(0);
      break;
    case EV_TXSTART:   Serial.println(F("EV_TXSTART")); break;
    case EV_TXCOMPLETE:Serial.println(F("EV_TXCOMPLETE")); break;
    default:           Serial.println(F("EV_OTHER")); break;
  }
}

// ─── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  while(!Serial) delay(10);
  Serial.println(F("=== Init ==="));

  setupAccelerometer();
  TfliteSetup();

  randomSeed(esp_random());
  os_init();
  LMIC_reset();
  LMIC_startJoining();

  buf_index = 0;
  lastSampleTime = micros();
}

// ─── Main Loop ─────────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = micros();
  if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime = now;

    // 1) Sample each axis
    sensors_event_t a, g, t;
    accel.getEvent(&a, &g, &t);
    raw_x[buf_index] = a.acceleration.x;
    raw_y[buf_index] = a.acceleration.y;
    raw_z[buf_index] = a.acceleration.z;
    buf_index++;

    // 2) Once we have 100 samples, denoise & send first XYZ
    if (buf_index >= 100) {
      buf_index = 0;
      Serial.println(F("→ Running TFLite inference on X, Y, Z…"));

      // Run three inferences
      doInference(raw_x, denoised_x);
      doInference(raw_y, denoised_y);
      doInference(raw_z, denoised_z);

      // ** Show raw vs denoised for sample[0] **
      Serial.print(F("Raw[0]  X,Y,Z: "));
      Serial.print(raw_x[0],3); Serial.print(F(", "));
      Serial.print(raw_y[0],3); Serial.print(F(", "));
      Serial.println(raw_z[0],3);

      Serial.print(F("Denoised[0] X,Y,Z: "));
      Serial.print(denoised_x[0],3); Serial.print(F(", "));
      Serial.print(denoised_y[0],3); Serial.print(F(", "));
      Serial.println(denoised_z[0],3);

      // 3) Send the first denoised X, Y, Z
      if (hasJoined) {
        sendFirstXYZ();
      } else {
        Serial.println(F("Not joined yet, skipping send"));
      }
    }
  }

  // Service LoRa
  os_runloop_once();
}

// ─── Quantize, invoke, dequantize ──────────────────────────────────────────────
void doInference(const float* src, float* dst) {
  // Fetch quant params
  float in_scale  = tensor_in->params.scale;
  int   in_zp     = tensor_in->params.zero_point;
  float out_scale = tensor_out->params.scale;
  int   out_zp    = tensor_out->params.zero_point;

  // Quantize inputs
  for (int i = 0; i < 100; i++) {
    int32_t q = lround(src[i] / in_scale + in_zp);
    q = q < -128 ? -128 : (q > 127 ? 127 : q);
    tensor_in->data.int8[i] = (int8_t)q;
  }

  // Invoke model
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println(F("!!! Inference failed"));
    memset(dst, 0, 100 * sizeof(float));
    return;
  }

  // Dequantize outputs
  for (int i = 0; i < 100; i++) {
    int8_t qout = tensor_out->data.int8[i];
    dst[i] = (qout - out_zp) * out_scale;
  }
}

// ─── Send first denoised X, Y, Z via LoRa ──────────────────────────────────────
void sendFirstXYZ() {
  float x = denoised_x[0],
        y = denoised_y[0],
        z = denoised_z[0];

  Serial.print(F("→ Sending denoised X,Y,Z: "));
  Serial.print(x,3); Serial.print(F(", "));
  Serial.print(y,3); Serial.print(F(", "));
  Serial.println(z,3);

  uint8_t payload[3 * sizeof(float)];
  memcpy(payload + 0, &x, sizeof(float));
  memcpy(payload + 4, &y, sizeof(float));
  memcpy(payload + 8, &z, sizeof(float));

  Serial.print(F("Payload bytes: "));
  for (int i = 0; i < sizeof(payload); i++) {
    if (payload[i] < 0x10) Serial.print('0');
    Serial.print(payload[i], HEX);
    Serial.print(' ');
  }
  Serial.println();

  if (!(LMIC.opmode & OP_TXRXPEND)) {
    LMIC_setTxData2(1, payload, sizeof(payload), 0);
    Serial.println(F("Packet queued"));
  } else {
    Serial.println(F("TX pending, skipping"));
  }
}

// ─── Accelerometer Setup ───────────────────────────────────────────────────────
void setupAccelerometer() {
  Serial.println(F("Init ISM330DHCX…"));
  if (!accel.begin_I2C()) {
    Serial.println(F("Accel not found!"));
    while (1) delay(10);
  }
  accel.setAccelRange(LSM6DS_ACCEL_RANGE_4_G);
  accel.setAccelDataRate(LSM6DS_RATE_104_HZ);
  Serial.println(F("Accel @ ~104 Hz"));
}

// ─── TFLite Setup ──────────────────────────────────────────────────────────────
void TfliteSetup() {
  Serial.println(F("Loading TFLite model…"));
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(autoencoder_model_INT8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Schema mismatch");
    while (1);
  }
  static tflite::MicroInterpreter static_interpreter(
    model, tflite::AllOpsResolver(), tensor_arena, kArenaSize, error_reporter
  );
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  tensor_in  = interpreter->input(0);
  tensor_out = interpreter->output(0);
  Serial.print(F("Input tensor type="));  Serial.println(tensor_in->type);
  Serial.print(F("Output tensor type=")); Serial.println(tensor_out->type);
  Serial.println(F("TFLite ready"));
}
