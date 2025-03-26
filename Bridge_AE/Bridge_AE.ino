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

// Globals for TensorFlow Lite
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 16 * 1024;  // Increase memory for autoencoder
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// SETUP FUNCTION (Initialize TensorFlow Lite)
void setup() {
    Serial.begin(115200);
    Serial.println("ESP32 Autoencoder Inference");

    // Initialize TensorFlow Lite Error Reporter
    static tflite::MicroErrorReporter micro_error_reporter;
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

    // Setup TensorFlow Lite Interpreter
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate Tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }

    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
}

// LOOP FUNCTION (Run Autoencoder on Noisy Sine Wave)
void loop() {
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
