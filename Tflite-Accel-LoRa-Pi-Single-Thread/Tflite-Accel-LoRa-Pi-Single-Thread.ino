#include <Arduino.h>
#include <Adafruit_ISM330DHCX.h>
#include <SPI.h>
#include <lmic.h>
#include <hal/hal.h>
 
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "autoencoder_model.h"


#include "main_functions.h"
#include "constants.h"
#include "output_handler.h"

 // Definitions
 #define CFG_us915 1  // For USA
 
 // LoRaWAN Keys (replace with your own if different)
 static const u1_t PROGMEM APPEUI[8]={0x78,0xde,0xfe,0x95,0x46,0x14,0xa8,0x68};
 static const u1_t PROGMEM DEVEUI[8]={0x2b,0x09,0x17,0x87,0x97,0xc7,0x3d,0x9a};
 static const u1_t PROGMEM APPKEY[16] = {0xb8,0x46,0x92,0x72,0x71,0xc1,0x00,0xbe,0x94,0x32,0xc9,0xc2,0xa3,0xf7,0x5c,0x0f};
 
 void os_getArtEui(u1_t* buf) { memcpy_P(buf, APPEUI, 8); }
 void os_getDevEui(u1_t* buf) { memcpy_P(buf, DEVEUI, 8); }
 void os_getDevKey(u1_t* buf) { memcpy_P(buf, APPKEY, 16); }
 
 
 // Function Declarations
 void accelSetup();
 void getAccelData();
 void loRaSetup();
 void printHex2(unsigned v);
 void onEvent(ev_t ev);
 void do_send(osjob_t* j);
 void TfliteSetup();
 void TfliteLoop();
 
 // Objects
 Adafruit_ISM330DHCX ism330dhcx; // Accelerometer object
 
 // Globals
 const unsigned TX_INTERVAL = 60; // Transmission interval in seconds
 static uint8_t mydata[12];       // Buffer for accelerometer data
 static osjob_t sendjob;          // LMIC job for sending data
 
 // Pin Mapping for LoRa Module (adjust based on your hardware)
 const lmic_pinmap lmic_pins = {
   .nss = 33,           // Chip select pin
   .rxtx = LMIC_UNUSED_PIN,
   .rst = 15,           // Reset pin
   .dio = {27},         // DIO0 pin (check if DIO1/DIO2 are needed per your module)
 };
 
// Globals for TensorFlow Lite
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 16 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Declare Error Reporter and Resolver globally (but don't declare interpreter yet!)
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;


 // MAIN SETUP
 void setup() {
   Serial.begin(115200);
   while (!Serial) { delay(10); } // Wait for Serial to initialize
   Serial.println(F("Starting"));
 
   loRaSetup();    // Initialize LoRa
   accelSetup();   // Initialize accelerometer
   TfliteSetup();
   do_send(&sendjob); // Start the join process immediately
 }
 
 // MAIN LOOP
 void loop() {
   os_runloop_once(); // Continuously run the LMIC state machine
 }
 
 // === Accelerometer Functions ===
 void accelSetup() {
   if (!ism330dhcx.begin_I2C()) {
     Serial.println("Failed to find ISM330DHCX chip");
     while (1) { delay(10); }
   }
 
   ism330dhcx.setAccelRange(LSM6DS_ACCEL_RANGE_4_G);
   ism330dhcx.setAccelDataRate(LSM6DS_RATE_208_HZ);
   ism330dhcx.setGyroRange(LSM6DS_GYRO_RANGE_500_DPS);
   ism330dhcx.setGyroDataRate(LSM6DS_RATE_208_HZ);
   ism330dhcx.configInt1(false, false, true); // Accelerometer DRDY on INT1
 }
 
 void getAccelData() {
   sensors_event_t accel, gyro, temp;
   ism330dhcx.getEvent(&accel, &gyro, &temp);
   memcpy(&mydata[0], &accel.acceleration.x, sizeof(float));
   memcpy(&mydata[4], &accel.acceleration.y, sizeof(float));
   memcpy(&mydata[8], &accel.acceleration.z, sizeof(float));
 }
 
 // === LoRaWAN Functions ===
 void loRaSetup() {
   os_init();    // Initialize the LMIC library
   LMIC_reset(); // Reset the LMIC state
 }
 
 void printHex2(unsigned v) {
   v &= 0xff;
   if (v < 16)
     Serial.print('0');
   Serial.print(v, HEX);
 }
 
 void onEvent(ev_t ev) {
   Serial.print(os_getTime());
   Serial.print(": ");
   switch (ev) {
     case EV_JOINING:
       Serial.println(F("EV_JOINING"));
       break;
     case EV_JOINED:
       Serial.println(F("EV_JOINED"));
       {
         u4_t netid = 0;
         devaddr_t devaddr = 0;
         u1_t nwkKey[16];
         u1_t artKey[16];
         LMIC_getSessionKeys(&netid, &devaddr, nwkKey, artKey);
         Serial.print("netid: ");
         Serial.println(netid, DEC);
         Serial.print("devaddr: ");
         Serial.println(devaddr, HEX);
         Serial.print("AppSKey: ");
         for (size_t i = 0; i < sizeof(artKey); ++i) {
           if (i != 0) Serial.print("-");
           printHex2(artKey[i]);
         }
         Serial.println("");
         Serial.print("NwkSKey: ");
         for (size_t i = 0; i < sizeof(nwkKey); ++i) {
           if (i != 0) Serial.print("-");
           printHex2(nwkKey[i]);
         }
         Serial.println();
       }
       LMIC_setLinkCheckMode(0); // Disable link check for simplicity
       // Schedule the first transmission after joining
       os_setTimedCallback(&sendjob, os_getTime() + sec2osticks(TX_INTERVAL), do_send);
       break;
     case EV_JOIN_FAILED:
       Serial.println(F("EV_JOIN_FAILED"));
       // Retry joining after a delay
       os_setTimedCallback(&sendjob, os_getTime() + sec2osticks(60), do_send);
       break;
     case EV_RXSTART:
       Serial.println(F("EV_RXSTART")); // Debug RX attempts
       break;
     case EV_RXCOMPLETE:
       Serial.println(F("EV_RXCOMPLETE")); // Confirm RX completion
       break;
     case EV_TXCOMPLETE:
       Serial.println(F("EV_TXCOMPLETE (includes waiting for RX windows)"));
       if (LMIC.txrxFlags & TXRX_ACK)
         Serial.println(F("Received ack"));
       if (LMIC.dataLen) {
         Serial.print(F("Received "));
         Serial.print(LMIC.dataLen);
         Serial.println(F(" bytes of payload"));
       }
       // Schedule next transmission
       os_setTimedCallback(&sendjob, os_getTime() + sec2osticks(TX_INTERVAL), do_send);
       break;
     default:
       Serial.print(F("Unknown event: "));
       Serial.println((unsigned)ev);
       break;
   }
 }
 
 void do_send(osjob_t* j) {
   if (LMIC.opmode & OP_TXRXPEND) {
     Serial.println(F("OP_TXRXPEND, not sending"));
   } else {
     getAccelData(); // Get fresh accelerometer data
 
     // Log the data being sent
     float x, y, z;
     memcpy(&x, &mydata[0], sizeof(float));
     memcpy(&y, &mydata[4], sizeof(float));
     memcpy(&z, &mydata[8], sizeof(float));
     Serial.print("Sending acceleration data: x=");
     Serial.print(x);
     Serial.print(", y=");
     Serial.print(y);
     Serial.print(", z=");
     Serial.println(z);
 
     LMIC_setTxData2(1, mydata, sizeof(mydata), 0); // Queue the packet
     Serial.println(F("Packet queued"));
   }
 }
  // === Tflite Functions ===
  void TfliteSetup(){
    Serial.println("ESP32 Autoencoder Inference");

    // Set the error reporter
    error_reporter = &micro_error_reporter;

    // Load Autoencoder Model
    model = tflite::GetModel(autoencoder_model_INT8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Now create interpreter after model is loaded
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
      return;
    }

    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
  }

  void TfliteLoop(){
    Serial.println("Running Autoencoder Inference...");

    float sample_input[100];  // Autoencoder expects 100 values

    // Generate a Noisy Sine Wave (Simulated Data)
    for (int i = 0; i < 100; i++) {
        sample_input[i] = sin(i * 0.1) + 0.2 * ((float)random(-100, 100) / 100.0);  // Noisy sine wave
    }

    // Copy data into model input tensor
    memcpy(input->data.f, sample_input, sizeof(sample_input));

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.println("Model inference failed!");
        return;
    }

    // Print the Denoised Output
    Serial.print("Denoised Output: ");
    for (int i = 0; i < 100; i++) {
        Serial.print(output->data.f[i]);
        Serial.print(", ");
    }
    Serial.println();

    delay(5000);  // Run inference every 5 seconds
  }