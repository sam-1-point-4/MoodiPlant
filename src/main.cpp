/**
 * @file main.cpp
 * @brief Plant Health Monitor using Arduino Nicla Vision and Edge Impulse with Grove RGB LED Matrix.
 *
 * This version primarily adapts the successful camera and Edge Impulse inferencing logic
 * from the provided Arduino IDE .ino file for PlatformIO, and integrates the Grove Seeed
 * LED RGB Matrix Driver v1.1 (MY9221) to display plant health status using pre-defined emojis.
 *
 * It addresses previous build errors, memory constraints, and ensures correct camera operation.
 *
 * The code utilizes an Edge Impulse trained model (Transfer learning MobileNetV2 block with
 * an Anomaly Detection - FOMO AD block) to classify plant leaf images as
 * 'healthy', 'unhealthy', or 'anomaly' (not a plant).
 *
 * Display Output on Grove RGB LED Matrix:
 * - Healthy: Displays a pre-configured SMILE emoji.
 * - Unhealthy: Displays a pre-configured FROWN emoji.
 * - Anomaly: Displays a pre-configured 'X' emoji.
 *
 * Optimized for PlatformIO build on Arduino Nicla Vision Pro.
 */

/* Includes ---------------------------------------------------------------- */
#include <Arduino.h>
#include "MoodiPlant_inferencing.h"
#include "edge-impulse-sdk/dsp/image/image.hpp" // For image processing functions
#include "camera.h"
#include "gc2145.h"
#include <ea_malloc.h> // For memory allocation (used in the .ino example)

// Corrected include path for the Grove RGB LED Matrix library
// Based on your screenshot, the main header is likely directly in the library root or src.
// If problems persist, you might need to find the specific header in the library that defines
// GroveTwoRGBLedMatrixClass and Emoji_t. Assuming 'grove_two_rgb_led_matrix.h' is the correct one.
#include <grove_two_rgb_led_matrix.h> 

/* Constant defines -------------------------------------------------------- */
#define EI_CAMERA_RAW_FRAME_BUFFER_COLS         320
#define EI_CAMERA_RAW_FRAME_BUFFER_ROWS         240
#define EI_CAMERA_RAW_FRAME_BYTE_SIZE           2 // RGB565 is 2 bytes per pixel

/* Edge Impulse ------------------------------------------------------------- */
// Alignment macro from the .ino file
#define ALIGN_PTR(p,a)   ((p & (a-1)) ?(((uintptr_t)p + a) & ~(uintptr_t)(a-1)) : p)

// Forward declarations from the .ino file structure
typedef struct {
    size_t width;
    size_t height;
} ei_device_resize_resolutions_t;

/* Private variables ------------------------------------------------------- */
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static bool is_initialised = false;
// is_ll_initialised is not strictly needed if we manage camera lifecycle centrally in ei_camera_init/deinit
// static bool is_ll_initialised = false; // low-level camera initialization state

GC2145 galaxyCore;
Camera cam(galaxyCore);
FrameBuffer fb; // FrameBuffer instance for camera operations

/*
** @brief points to the output of the capture after RGB565 to RGB888 conversion
** This buffer will hold the image in RGB888 format for Edge Impulse processing.
*/
static uint8_t *ei_camera_capture_out = NULL;

/*
** @brief used to store the raw frame from the camera in RGB565 format
*/
static uint8_t *ei_camera_frame_mem; // Raw allocated memory
static uint8_t *ei_camera_frame_buffer; // 32-byte aligned pointer for actual buffer

// --- Grove RGB LED Matrix Object ---
GroveTwoRGBLedMatrixClass matrix; // Instantiate the Grove RGB LED Matrix object

/* Function prototypes ------------------------------------------------------- */
void ei_printf(const char *format, ...); // For serial output
bool ei_camera_init(void);
void ei_camera_deinit(void);
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf) ; // out_buf is usually NULL
bool RBG565ToRGB888(uint8_t *src_buf, uint8_t *dst_buf, uint32_t src_len);
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr);
int calculate_resize_dimensions(uint32_t out_width, uint32_t out_height, uint32_t *resize_col_sz, uint32_t *resize_row_sz, bool *do_resize);
void display_emoji(Emoji_t emoji); // Function to display emojis on the matrix

/**
 * @brief Arduino setup function.
 */
void setup()
{
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial); // Wait for serial monitor to connect
    ei_printf("MoodiPlant - Plant Health Monitor (Grove LED Matrix Version)\n");
    ei_printf("----------------------------------------------------------\n");

    // Initialize the Grove RGB LED Matrix
    if (!matrix.begin()) {
        ei_printf("ERROR: Failed to initialize Grove RGB LED Matrix! Check connections.\n");
        // Fallback: If matrix fails, flash onboard LED in red (requires LED pins config)
        pinMode(LEDR, OUTPUT); pinMode(LEDG, OUTPUT); pinMode(LEDB, OUTPUT);
        analogWrite(LEDR, 0); analogWrite(LEDG, 255); analogWrite(LEDB, 255); // Red
        while (1); // Halt on critical error
    }
    matrix.clear(); // Ensure matrix is clear initially
    matrix.displayFlashEmoji(MATRIX_EMOJI_SMILE); // Quick test
    delay(500);
    matrix.displayFlashEmoji(MATRIX_EMOJI_X); // Quick test
    delay(500);
    matrix.clear();

    // Initialize M4 RAM as in the .ino file
    // Arduino Nicla Vision has 512KB of RAM allocated for M7 core
    // and additional 244k on the M4 address space.
    // Allocating 288 kB for M4 RAM. This is crucial for Edge Impulse memory.
    malloc_addblock((void*)0x30000000, 288 * 1024);
    ei_printf("M4 RAM block added.\r\n");

    if (ei_camera_init() == false) {
        ei_printf("Failed to initialize Camera! Please check hardware connection and try again.\r\n");
        display_emoji(MATRIX_EMOJI_X); // Display 'X' on matrix for camera error
        while (1); // Halt on critical error
    } else {
        ei_printf("Camera initialized successfully.\r\n");
    }

    if ((EI_CLASSIFIER_INPUT_WIDTH > EI_CAMERA_RAW_FRAME_BUFFER_COLS) || (EI_CLASSIFIER_INPUT_HEIGHT > EI_CAMERA_RAW_FRAME_BUFFER_ROWS)) {
        ei_printf("ERROR: Edge Impulse model input resolution (%dx%d) is larger than camera raw output (%dx%d).\n",
                  EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT, EI_CAMERA_RAW_FRAME_BUFFER_COLS, EI_CAMERA_RAW_FRAME_BUFFER_ROWS);
        display_emoji(MATRIX_EMOJI_X); // Display 'X' for config error
        while(1);
    }
    ei_printf("Edge Impulse model loaded. Input resolution: %dx%d\n", EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);
    ei_printf("\nSystem ready. Point camera at a plant leaf.\n");
    display_emoji(MATRIX_EMOJI_SMILE); // Indicate readiness with a smile
    delay(1000);
    matrix.clear();
}

/**
 * @brief Get data and run inferencing
 */
void loop()
{
    ei_printf("\nStarting inferencing in 2 seconds...\n");

    // Using delay as a simple wait, or ei_sleep if your Edge Impulse library provides it
    delay(2000);

    ei_printf("Attempting to take photo...\n");

    ei::signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data = &ei_camera_get_data; // This will now use ei_camera_capture_out (RGB888)

    // Capture the image and perform necessary conversions/resizing
    // The out_buf is NULL here, meaning ei_camera_capture_out will be used internally
    if (ei_camera_capture((size_t)EI_CLASSIFIER_INPUT_WIDTH, (size_t)EI_CLASSIFIER_INPUT_HEIGHT, NULL) == false) {
        ei_printf("ERR: Failed to capture image during ei_camera_capture().\r\n");
        display_emoji(MATRIX_EMOJI_X);
        delay(2000);
        return;
    }
    ei_printf("Photo captured successfully. Running classifier...\n");

    // Run the classifier
    ei_impulse_result_t result = { 0 };

    EI_IMPULSE_ERROR err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", err);
        display_emoji(MATRIX_EMOJI_X);
        delay(2000);
        return;
    }

    // print the predictions
    ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
                result.timing.dsp, result.timing.classification, result.timing.anomaly);

    bool object_found = false;

#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        auto bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }

        object_found = true;
        ei_printf("    %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                  bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);

        // Display emoji based on detected object label
        if (strcmp(bb.label, "healthy") == 0) {
            display_emoji(MATRIX_EMOJI_SMILE);
            break; // Assuming only one dominant object is needed for display
        }
        else if (strcmp(bb.label, "unhealthy") == 0) {
            display_emoji(MATRIX_EMOJI_FROWN);
            break;
        }
        else if (strcmp(bb.label, "anomaly") == 0) { // Or if high anomaly score
            display_emoji(MATRIX_EMOJI_X);
            break;
        }
    }
#else // Classification model
    ei_printf("Predictions:\r\n");
    float healthy_score = 0;
    float unhealthy_score = 0;
    float anomaly_score = 0;

    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("    %s: %.5f\r\n", ei_classifier_inferencing_categories[i], result.classification[i].value);
        if (strcmp(ei_classifier_inferencing_categories[i], "healthy") == 0) {
            healthy_score = result.classification[i].value;
        } else if (strcmp(ei_classifier_inferencing_categories[i], "unhealthy") == 0) {
            unhealthy_score = result.classification[i].value;
        } else if (strcmp(ei_classifier_inferencing_categories[i], "anomaly") == 0) {
            anomaly_score = result.classification[i].value;
        }
    }

    // Determine the most confident class
    if (healthy_score > unhealthy_score && healthy_score > anomaly_score) {
        display_emoji(MATRIX_EMOJI_SMILE);
        object_found = true;
    } else if (unhealthy_score > healthy_score && unhealthy_score > anomaly_score) {
        display_emoji(MATRIX_EMOJI_FROWN);
        object_found = true;
    } else if (anomaly_score > healthy_score && anomaly_score > unhealthy_score) {
        display_emoji(MATRIX_EMOJI_X);
        object_found = true;
    } else {
        // Fallback for inconclusive results
        ei_printf("    - No clear classification. Displaying Anomaly (X).\n");
        display_emoji(MATRIX_EMOJI_X);
    }
#endif

#if EI_CLASSIFIER_HAS_ANOMALY
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
    // You might want to add logic here to override if anomaly score is very high,
    // e.g., if (result.anomaly > 0.8) { display_emoji(MATRIX_EMOJI_X); }
#endif

#if EI_CLASSIFIER_HAS_VISUAL_ANOMALY
    ei_printf("Visual anomalies:\r\n");
    for (uint32_t i = 0; i < result.visual_ad_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.visual_ad_grid_cells[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("    %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                  bb.label, bb.value, bb.x, bb.y, bb.width, bb.height);
    }
#endif

    if (!object_found) {
        ei_printf("  - No objects detected or primary classification found. Clearing matrix.\n");
        matrix.clear(); // Clear matrix if nothing specific was found
    }
    
    delay(3000); // Hold the emoji for 3 seconds
    matrix.clear(); // Clear the matrix before next cycle
    delay(500); // Small pause before next capture
}

/**
 * @brief Setup image sensor & start streaming
 * This function heavily relies on the logic from your successful .ino file.
 * @retval false if initialisation failed
 */
bool ei_camera_init(void) {
    if (is_initialised) {
        ei_printf("Camera already marked as initialized. Returning true.\r\n");
        return true;
    }

    ei_printf("Attempting cam.begin() with 1 frame buffer...\r\n");
    // Using 1 frame buffer as in the successful .ino file for memory optimization
    if (!cam.begin(CAMERA_R320x240, CAMERA_RGB565, 1)) {
        ei_printf("ERR: cam.begin() failed! This means the camera hardware or low-level driver could not be initialized.\r\n");
        return false;
    }
    ei_printf("cam.begin() successful. Allocating camera frame memory...\r\n");

    // initialize frame buffer for raw RGB565 data from camera
    ei_camera_frame_mem = (uint8_t *) ei_malloc(EI_CAMERA_RAW_FRAME_BUFFER_COLS * EI_CAMERA_RAW_FRAME_BUFFER_ROWS * EI_CAMERA_RAW_FRAME_BYTE_SIZE + 32 /*alignment*/);
    if(ei_camera_frame_mem == NULL) {
        ei_printf("ERR: Failed to allocate ei_camera_frame_mem! Check M4 RAM allocation or available memory.\r\n");
        return false;
    }
    ei_printf("ei_camera_frame_mem allocated. Setting frame buffer...\r\n");

    ei_camera_frame_buffer = (uint8_t *)ALIGN_PTR((uintptr_t)ei_camera_frame_mem, 32);
    fb.setBuffer(ei_camera_frame_buffer); // Set the buffer for the FrameBuffer object

    is_initialised = true;
    ei_printf("Camera system successfully initialized (is_initialised = true).\r\n");

    return true;
}

/**
 * @brief Stop streaming of sensor data
 */
void ei_camera_deinit(void) {
    if (ei_camera_frame_mem) { // Only free if allocated
        ei_free(ei_camera_frame_mem);
        ei_camera_frame_mem = NULL;
    }
    ei_camera_frame_buffer = NULL;
    is_initialised = false;
}

/**
 * @brief Capture, rescale and crop image
 * This function is directly from your .ino file, responsible for camera capture,
 * RGB565 to RGB888 conversion, and potential resizing/cropping.
 *
 * @param[in] img_width     width of output image
 * @param[in] img_height    height of output image
 * @param[in] out_buf       pointer to store output image, NULL may be used
 * if ei_camera_capture_out is to be used for processing.
 *
 * @retval false if not initialised, image captured, rescaled or cropped failed
 */
bool ei_camera_capture(uint32_t img_width, uint32_t img_height, uint8_t *out_buf) {
    bool do_resize = false;
    bool do_crop = false;

    // Allocate memory for the RGB888 output buffer that Edge Impulse expects
    // This buffer will contain the converted and potentially resized image
    ei_camera_capture_out = (uint8_t*)ea_malloc(img_width * img_height * 3 + 32); // 3 bytes per pixel for RGB888
    if(ei_camera_capture_out == NULL) {
        ei_printf("ERR: Failed to allocate ei_camera_capture_out memory.\r\n");
        return false;
    }
    ei_camera_capture_out = (uint8_t *)ALIGN_PTR((uintptr_t)ei_camera_capture_out, 32);
    ei_printf("Allocated memory for RGB888 capture output.\r\n");

    if (!is_initialised) {
        ei_printf("ERR: Camera is not initialized when calling ei_camera_capture()!\r\n");
        ea_free(ei_camera_capture_out); // Free allocated memory on failure
        ei_camera_capture_out = NULL;
        return false;
    }
    ei_printf("Camera is initialized. Attempting to grab frame...\r\n");

    // Grab the frame into the pre-allocated fb (ei_camera_frame_buffer)
    int snapshot_response = cam.grabFrame(fb, 1000); // Using 1000ms timeout
    if (snapshot_response != 0) {
        ei_printf("ERR: Failed to get snapshot (%d) from cam.grabFrame()! This is where the actual capture failed.\r\r\n", snapshot_response);
        ea_free(ei_camera_capture_out); // Free allocated memory on failure
        ei_camera_capture_out = NULL;
        return false;
    }
    ei_printf("Snapshot grabbed successfully. Converting to RGB888...\r\n");

    // Convert the captured RGB565 frame to RGB888
    bool converted = RBG565ToRGB888(ei_camera_frame_buffer, ei_camera_capture_out, cam.frameSize());

    if(!converted){
        ei_printf("ERR: Conversion from RGB565 to RGB888 failed!\n");
        ea_free(ei_camera_capture_out); // Free allocated memory on failure
        ei_camera_capture_out = NULL;
        return false;
    }
    ei_printf("Conversion to RGB888 successful. Calculating resize dimensions...\r\n");

    uint32_t resize_col_sz;
    uint32_t resize_row_sz;
    
    // Choose resize dimensions based on the Edge Impulse model's input size
    int res = calculate_resize_dimensions(img_width, img_height, &resize_col_sz, &resize_row_sz, &do_resize);
    if (res) {
        ei_printf("ERR: Failed to calculate resize dimensions (%d)\r\n", res);
        ea_free(ei_camera_capture_out); // Free allocated memory on failure
        ei_camera_capture_out = NULL;
        return false;
    }

    if ((img_width != resize_col_sz) || (img_height != resize_row_sz)) {
        do_crop = true; // This means a crop/resize operation will occur
    }

    if (do_resize || do_crop) { // If any resizing or cropping is needed
        ei_printf("Resizing/cropping image to %lux%lu...\r\n", resize_col_sz, resize_row_sz);
        // The image::processing::crop_and_interpolate_rgb888 function handles both.
        // It operates in-place if source and destination buffers are the same.
        ei::image::processing::crop_and_interpolate_rgb888(
            ei_camera_capture_out, // Source buffer (RGB888)
            EI_CAMERA_RAW_FRAME_BUFFER_COLS, // Source width
            EI_CAMERA_RAW_FRAME_BUFFER_ROWS, // Source height
            ei_camera_capture_out, // Destination buffer (in-place processing)
            resize_col_sz, // Target width (EI_CLASSIFIER_INPUT_WIDTH)
            resize_row_sz // Target height (EI_CLASSIFIER_INPUT_HEIGHT)
        );
    }
    ei_printf("Image processing complete.\r\n");
    // Note: ei_camera_capture_out is NOT freed here, as it's needed by ei_camera_get_data for inference.
    // It will be freed later or reused.

    return true;
}

/**
 * @brief Convert rgb565 data to rgb888
 * @param[in] src_buf The rgb565 data
 * @param     dst_buf The rgb888 data (will be written here)
 * @param     src_len length of rgb565 data in bytes
 */
bool RBG565ToRGB888(uint8_t *src_buf, uint8_t *dst_buf, uint32_t src_len)
{
    uint8_t hb, lb; // High byte, Low byte
    uint32_t pix_count = src_len / 2; // Each RGB565 pixel is 2 bytes

    for(uint32_t i = 0; i < pix_count; i ++) {
        hb = *src_buf++; // Read high byte
        lb = *src_buf++; // Read low byte

        // Convert RGB565 (RRRRRGGG GGGBBBBB) to RGB888 (RRRRRRRR GGGGGGGG BBBBBBBB)
        // RRRRR -> RRRRR000 (shift left by 3)
        // GGGGGG -> GGGGGG00 (shift left by 2)
        // BBBBB -> BBBBB000 (shift left by 3)
        *dst_buf++ = (hb & 0xF8); // R (top 5 bits of high byte)
        *dst_buf++ = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3); // G (bottom 3 bits of high byte + top 3 bits of low byte)
        *dst_buf++ = (lb & 0x1F) << 3; // B (bottom 5 bits of low byte)
    }
    return true;
}

/**
 * @brief Provides a chunk of the camera image to the Edge Impulse classifier.
 * This function reads from the ei_camera_capture_out (RGB888) buffer.
 */
static int ei_camera_get_data(size_t offset, size_t length, float *out_ptr)
{
    // The offset and length here are in terms of 'pixels' or the flat dimension of the input signal,
    // where each 'pixel' for Edge Impulse in this context is a single float representing concatenated RGB.
    
    // Calculate the byte offset into the RGB888 buffer
    size_t byte_offset = offset * 3; // Each pixel is 3 bytes (R, G, B)
    size_t bytes_to_copy = length * 3;

    // Check if the source buffer is valid and has enough data
    if (ei_camera_capture_out == NULL || (byte_offset + bytes_to_copy) > (EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * 3)) {
        ei_printf("ERR: ei_camera_get_data: Invalid buffer access! offset=%zu, length=%zu\n", offset, length);
        return -1; // Indicate error
    }

    for (size_t i = 0; i < length; i++) {
        uint8_t r = ei_camera_capture_out[byte_offset + (i * 3)];
        uint8_t g = ei_camera_capture_out[byte_offset + (i * 3) + 1];
        uint8_t b = ei_camera_capture_out[byte_offset + (i * 3) + 2];

        // Edge Impulse expects concatenated RGB888 as a float for image input
        out_ptr[i] = (float)((r << 16) | (g << 8) | b);
    }
    
    // After the classifier runs, free the allocated RGB888 buffer
    // This is crucial for memory management on subsequent loops.
    ea_free(ei_camera_capture_out);
    ei_camera_capture_out = NULL; // Reset pointer after freeing
    
    return 0; // Success
}

/**
 * @brief Determine whether to resize and to which dimension
 * This function is directly from your .ino file.
 *
 * @param[in] out_width     width of output image (EI_CLASSIFIER_INPUT_WIDTH)
 * @param[in] out_height    height of output image (EI_CLASSIFIER_INPUT_HEIGHT)
 * @param[out] resize_col_sz   pointer to frame buffer's column/width value
 * @param[out] resize_row_sz   pointer to frame buffer's rows/height value
 * @param[out] do_resize     returns whether to resize (or not)
 *
 */
int calculate_resize_dimensions(uint32_t out_width, uint32_t out_height, uint32_t *resize_col_sz, uint32_t *resize_row_sz, bool *do_resize)
{
    // These are common camera resolutions supported by the Nicla Vision,
    // used to determine the intermediate buffer size for efficient cropping/resizing.
    size_t list_size = 6;
    const ei_device_resize_resolutions_t list[list_size] = {
        {64, 64},
        {96, 96},
        {160, 120},
        {160, 160},
        {320, 240},
        {640, 480} // Adding 640x480 as it's a common camera output resolution
    };

    // (default) conditions
    *resize_col_sz = EI_CAMERA_RAW_FRAME_BUFFER_COLS; // Default to camera's raw resolution
    *resize_row_sz = EI_CAMERA_RAW_FRAME_BUFFER_ROWS;
    *do_resize = false; // Assume no resize needed initially

    // Find the smallest resolution from the list that is greater than or equal to the output dimensions
    for (size_t ix = 0; ix < list_size; ix++) {
        if ((out_width <= list[ix].width) && (out_height <= list[ix].height)) {
            *resize_col_sz = list[ix].width;
            *resize_row_sz = list[ix].height;
            *do_resize = true;
            break;
        }
    }
    
    // If the target size is precisely the camera's raw output, no resize is needed.
    if (!*do_resize && (out_width == EI_CAMERA_RAW_FRAME_BUFFER_COLS) && (out_height == EI_CAMERA_RAW_FRAME_BUFFER_ROWS)) {
        *do_resize = false; 
    } 
    // If we still haven't set do_resize, but the target dimensions are different from raw camera output,
    // it implies we need to resize directly to the target dimensions.
    else if (!*do_resize) {
        *resize_col_sz = out_width;
        *resize_row_sz = out_height;
        *do_resize = true;
        ei_printf("Warning: Target resolution (%lu,%lu) not an exact match in predefined list or raw. Attempting direct resize.\r\n", out_width, out_height);
    }

    return 0;
}

/**
 * @brief Displays a predefined emoji on the Grove RGB LED Matrix.
 * @param emoji The emoji to display (e.g., MATRIX_EMOJI_SMILE, MATRIX_EMOJI_FROWN, MATRIX_EMOJI_X).
 */
void display_emoji(Emoji_t emoji) {
    matrix.displayFlashEmoji(emoji);
}

/**
 * @brief A wrapper around Serial.printf for the Edge Impulse SDK.
 */
void ei_printf(const char *format, ...) {
    char print_buf[1024] = { 0 }; // Buffer for formatted string

    va_list args;
    va_start(args, format);
    int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
    va_end(args);

    if (r > 0) {
        Serial.print(print_buf);
    }
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#warning "EI_CLASSIFIER_SENSOR is not defined as EI_CLASSIFIER_SENSOR_CAMERA. Ensure your Edge Impulse model is trained for camera input."
#endif