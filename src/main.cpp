#include <Arduino.h>
#include <camera.h> // Now properly referenced and understood
#include <grove_two_rgb_led_matrix.h> // Correctly referenced based on provided header
#include <MoodiPlant_inferencing.h> // Edge Impulse library
#include <gc2145.h> // Specific camera sensor library for Nicla Vision Pro

// Forward declaration of the specific ImageSensor instance for Nicla Vision Pro.
// This is crucial! The Camera class constructor requires an ImageSensor object.
// The Arduino core for Nicla Vision typically defines this extern object.
// Common options are GC2145_Sensor or HIMAX_HM0360_Sensor.
// We are assuming GC2145_Sensor here based on common Nicla Vision setups.
// If your board uses a different sensor (e.g., Himax HM0360), you must change
// 'GC2145_Sensor' to the correct extern ImageSensor object provided by your
// Arduino core/camera library (e.g., extern ImageSensor HIMAX_HM0360_Sensor;).
// Use the correct concrete sensor type as provided by your camera library.
// For the GC2145 sensor, the class is usually named GC2145.
extern GC2145 GC2145_Sensor;

// LED Matrix instance
// The matrix object is initialized through its constructor globally.
GroveTwoRGBLedMatrixClass matrix;

// Camera and inference settings
#define CONFIDENCE_THRESHOLD 0.6
#define MONITORING_INTERVAL 3000 // 3 seconds (3000ms) for auto-capture
#define IMG_WIDTH 96
#define IMG_HEIGHT 96
#define BYTES_PER_PIXEL 2 // RGB565 uses 2 bytes per pixel (16 bits)

// Global variables
// Initialize FrameBuffer with desired dimensions and bits per pixel (RGB565 = 16 bits)
// The constructor for FrameBuffer is (width, height, bpp)
// Using IMG_WIDTH and IMG_HEIGHT for the framebuffer size for consistency with EI model input.
FrameBuffer fb(IMG_WIDTH, IMG_HEIGHT, 16); // 16 bpp for RGB565

// Initialize Camera by passing the specific ImageSensor instance (GC2145_Sensor in this case)
// This resolves the "no matching function for call to 'Camera::Camera()'" error.
Camera cam(GC2145_Sensor); 

static float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

// Simple emoji patterns for 8x8 matrix (as 64-byte arrays)
// 0 = off, 1 = on (will be rendered white)
const uint8_t SMILE_PATTERN[64] = {
    0,0,1,1,1,1,0,0,
    0,1,0,0,0,0,1,0,
    1,0,1,0,0,1,0,1,
    1,0,0,0,0,0,0,1,
    1,0,1,0,0,1,0,1,
    1,0,0,1,1,0,0,1,
    0,1,0,0,0,0,1,0,
    0,0,1,1,1,1,0,0
};

// Adjusted FROWN pattern to be more distinct from smile
const uint8_t FROWN_PATTERN[64] = {
    0,0,1,1,1,1,0,0,
    0,1,0,0,0,0,1,0,
    1,0,0,1,1,0,0,1, // Eyes closer together
    1,0,0,0,0,0,0,1,
    1,0,1,0,0,1,0,1, // Eyebrows/Upper part
    1,0,1,1,1,1,0,1, // Mouth as inverted curve
    0,1,0,0,0,0,1,0,
    0,0,1,1,1,1,0,0
};

const uint8_t CROSS_PATTERN[64] = {
    1,0,0,0,0,0,0,1,
    0,1,0,0,0,0,1,0,
    0,0,1,0,0,1,0,0,
    0,0,0,1,1,0,0,0,
    0,0,0,1,1,0,0,0,
    0,0,1,0,0,1,0,0,
    0,1,0,0,0,0,1,0,
    1,0,0,0,0,0,0,1
};

/**
 * @brief Displays a given 8x8 pixel pattern on the LED matrix.
 * This function now uses `displayFrames` to send the custom pattern.
 * It will display the pattern using white pixels for '1' and black for '0'.
 * @param pattern A pointer to a 64-byte array representing the 8x8 pattern.
 */
void displayPattern(const uint8_t* pattern) {
    // The displayFrames function expects a uint8_t* buffer, duration, forever_flag, and frames_number.
    // For a single static pattern, duration_time can be 0 (no explicit duration, will stay until changed)
    // and forever_flag can be true, with frames_number = 1.
    // The library's displayFrames will handle the rendering based on the 0/1 values in the pattern.
    matrix.displayFrames((uint8_t*)pattern, 0, true, 1);
}

/**
 * @brief Shows the smile emoji on the LED matrix using the custom pattern.
 */
void showSmile() { displayPattern(SMILE_PATTERN); }

/**
 * @brief Shows the frown emoji on the LED matrix using the custom pattern.
 */
void showFrown() { displayPattern(FROWN_PATTERN); }

/**
 * @brief Shows the cross (X) emoji on the LED matrix using the custom pattern.
 */
void showCross() { displayPattern(CROSS_PATTERN); }

/**
 * @brief Clears the LED matrix using the `stopDisplay()` function from the library.
 */
void clearMatrix() {
    matrix.stopDisplay(); // This command cleans the display
}

/**
 * @brief Converts a 16-bit RGB565 color value to an 8-bit grayscale value.
 * @param rgb565 The 16-bit RGB565 color.
 * @return The 8-bit grayscale value.
 */
uint8_t rgb565ToGray(uint16_t rgb565) {
    // Extract R, G, B components and scale to 8-bit
    uint8_t r = ((rgb565 >> 11) & 0x1F) << 3; // 5-bit Red, scaled to 8-bit
    uint8_t g = ((rgb565 >> 5) & 0x3F) << 2;  // 6-bit Green, scaled to 8-bit
    uint8_t b = (rgb565 & 0x1F) << 3;        // 5-bit Blue, scaled to 8-bit

    // Simple luminance conversion (ITU-R BT.601)
    return (r * 77 + g * 151 + b * 28) >> 8;
}

/**
 * @brief Prepares image features from the camera framebuffer for Edge Impulse classification.
 * It scales and converts the captured RGB565 image to grayscale features.
 * This version directly accesses the raw pixel buffer from FrameBuffer.
 * @return 0 on success, non-zero on failure.
 */
int prepareFeatures() {
    // Get the actual resolution from the Camera object, not the FrameBuffer object
    uint32_t camWidth = cam.getResolutionWidth();
    uint32_t camHeight = cam.getResolutionHeight();
    uint8_t* frameBufferPtr = fb.getBuffer(); // Get pointer to the raw pixel data

    // Ensure framebuffer and camera dimensions are valid before proceeding
    if (camWidth == 0 || camHeight == 0 || frameBufferPtr == nullptr) {
        Serial.println("Error: Camera or Framebuffer not ready (zero dimensions or null buffer).");
        return -1; // Indicate error
    }
    
    // Convert framebuffer to features array
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            // Scale coordinates from desired IMG_WIDTH/HEIGHT to actual camera resolution
            // Cast srcX and srcY to uint32_t for comparison to avoid -Wsign-compare warnings
            int srcX = (x * camWidth) / IMG_WIDTH;
            int srcY = (y * camHeight) / IMG_HEIGHT;
            
            // Ensure source coordinates are within bounds
            // Using static_cast<uint32_t> to explicitly resolve signed/unsigned comparison warnings.
            if (static_cast<uint32_t>(srcX) >= camWidth) srcX = camWidth - 1;
            if (static_cast<uint32_t>(srcY) >= camHeight) srcY = camHeight - 1;
            
            // Calculate pixel index in the raw buffer (RGB565 is 2 bytes per pixel)
            // Pixel data is stored row by row. Each pixel is 2 bytes.
            uint32_t pixelIndex = (srcY * camWidth + srcX) * BYTES_PER_PIXEL;

            // Extract the 16-bit RGB565 pixel value
            // Assuming little-endian byte order for RGB565: Low byte first, then high byte.
            uint16_t pixel = (frameBufferPtr[pixelIndex + 1] << 8) | frameBufferPtr[pixelIndex];
            
            uint8_t gray = rgb565ToGray(pixel);
            
            // Normalize to 0-1 range for Edge Impulse input
            features[y * IMG_WIDTH + x] = (float)gray / 255.0f;
        }
    }
    return 0;
}

/**
 * @brief Main function to capture an image, classify it using Edge Impulse,
 * and display the result on the LED matrix.
 */
void classifyImage() {
    Serial.println("Capturing image...");
    
    // Capture image into the global framebuffer 'fb'
    if (cam.grabFrame(fb, 3000) == 0) { // 3000ms timeout
        Serial.println("Failed to capture image. Camera may not be initialized or connected.");
        showCross(); // Show cross on failure
        return;
    }
    
    Serial.println("Image captured successfully.");
    
    // Prepare features for classification
    if (prepareFeatures() != 0) {
        Serial.println("Feature preparation failed.");
        showCross(); // Show cross on failure
        return;
    }
    
    // Create signal for Edge Impulse classifier
    signal_t signal;
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = [](size_t offset, size_t length, float *out_ptr) -> int {
        memcpy(out_ptr, features + offset, length * sizeof(float));
        return 0; // Indicate success
    };
    
    // Run the Edge Impulse classifier
    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false); // false for non-debug mode
    
    if (res != EI_IMPULSE_OK) {
        Serial.print("Classification failed: Error code ");
        Serial.println(res);
        showCross(); // Show cross on classification error
        return;
    }
    
    // Find the best prediction (highest confidence)
    float maxConf = 0.0;
    int bestIdx = -1;
    
    Serial.println("Classification Results:");
    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        float conf = result.classification[i].value;
        Serial.print(result.classification[i].label);
        Serial.print(": ");
        Serial.println(conf, 4); // Print confidence with 4 decimal places
        
        if (conf > maxConf) {
            maxConf = conf;
            bestIdx = i;
        }
    }
    
    // Display result based on confidence and label
    if (maxConf < CONFIDENCE_THRESHOLD) {
        Serial.println("Low confidence prediction. Showing cross.");
        showCross();
    } else {
        String label = String(result.classification[bestIdx].label);
        Serial.print("Predicted: ");
        Serial.print(label);
        Serial.print(" (Confidence: ");
        Serial.print(maxConf * 100, 2); // Display confidence as percentage with 2 decimal places
        Serial.println("%)");
        
        // Show appropriate emoji based on the predicted label
        if (label.indexOf("healthy") >= 0 || label.indexOf("good") >= 0) {
            Serial.println("Condition: Healthy/Good. Showing smile.");
            showSmile();
        } else if (label.indexOf("unhealthy") >= 0 || label.indexOf("sick") >= 0) {
            Serial.println("Condition: Unhealthy/Sick. Showing frown.");
            showFrown();
        } else {
            Serial.println("Condition: Unknown/Anomaly. Showing cross.");
            showCross();
        }
    }
    
    Serial.println("--- End of Classification ---");
}

/**
 * @brief Arduino setup function. Initializes serial communication,
 * LED matrix, and camera.
 */
void setup() {
    Serial.begin(115200); // Start serial communication
    while (!Serial) delay(10); // Wait for serial port to connect (useful for debugging)
    
    Serial.println("Starting Plant Monitor System...");
    
    // Initialize LED matrix
    Serial.println("Attempting to use Grove RGB LED Matrix. (No explicit begin() function)");
    // The matrix object is initialized via its constructor at global scope.
    // You can optionally call matrix.scanGroveTwoRGBLedMatrixI2CAddress(); here
    // if you need to ensure the device is found or its address is re-scanned.
    // However, it doesn't return a status, so no 'if' check is possible.
    
    // Test patterns to confirm matrix is working
    Serial.println("Testing LED matrix patterns (assuming matrix is connected)...");
    showSmile(); delay(1000);
    showFrown(); delay(1000);
    showCross(); delay(1000);
    clearMatrix(); delay(500);
    
    // Initialize camera
    Serial.print("Initializing Camera (CAMERA_R320x240, CAMERA_RGB565, 30fps)...");
    if (cam.begin(CAMERA_R320x240, CAMERA_RGB565, 30)) {
        Serial.println("Success!");
    } else {
        Serial.println("Failed!");
        Serial.println("ERROR: Camera failed to initialize. Ensure connections are correct.");
        // If camera fails, show a continuous cross on the matrix to indicate error
        while (1) {
            showCross();
            delay(1000);
            clearMatrix();
            delay(1000);
        }
    }
    
    Serial.println("System ready and running!");
    showSmile(); // Show a smile when system is ready
    delay(2000);
    clearMatrix(); // Clear matrix after initial smile
}

/**
 * @brief Arduino loop function. Periodically captures and classifies images,
 * and handles serial commands.
 */
void loop() {
    static unsigned long lastCapture = 0; // Stores time of last capture

    // Automatic image capture based on MONITORING_INTERVAL
    if (millis() - lastCapture >= MONITORING_INTERVAL) {
        classifyImage(); // Perform image capture and classification
        lastCapture = millis(); // Update last capture time
    }
    
    // Handle incoming serial commands
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n'); // Read command until newline
        cmd.trim();        // Remove leading/trailing whitespace
        cmd.toLowerCase(); // Convert to lowercase for easier comparison
        
        if (cmd == "c" || cmd == "capture") {
            classifyImage();
        } else if (cmd == "s" || cmd == "smile") {
            showSmile();
        } else if (cmd == "f" || cmd == "frown") {
            showFrown();
        } else if (cmd == "x" || cmd == "cross") {
            showCross();
        } else if (cmd == "clear") {
            clearMatrix();
        } else if (cmd == "help") {
            Serial.println("Available Commands:");
            Serial.println("  c or capture: Capture and classify an image.");
            Serial.println("  s or smile: Display the smile emoji.");
            Serial.println("  f or frown: Display the frown emoji.");
            Serial.println("  x or cross: Display the cross emoji.");
            Serial.println("  clear: Clear the LED matrix.");
            Serial.println("  help: Show this help message.");
        } else {
            Serial.print("Unknown command: '");
            Serial.print(cmd);
            Serial.println("'. Type 'help' for available commands.");
        }
    }
    
    delay(100); // Small delay to prevent busy-waiting on serial or rapid loops
}