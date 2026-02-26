# Beecam Script Logic Summary - High-Level Pseudo-Code

## Overview
This script implements a dual-arena bee motion detection and recording system using a Raspberry Pi camera and GPIO controls.

## Main Components

### 1. Initialization Phase
```
IMPORT required libraries (cv2, picamera2, GPIO, datetime, etc.)

DEFINE utility function get_mp4_NameWithDate():
    - Generate unique timestamped filename
    - Avoid filename conflicts by adding counter if needed

INITIALIZE camera:
    - Set resolution to 1640x1232
    - Configure RGB888 format
    - Start camera preview

DEFINE rotate_image() function for 180-degree rotation

SETUP GPIO pins:
    - LED_LIST = [17, 27, 22, 23] (Red, Blue, Green, Status LEDs)
    - BUTTON_PIN = 26
    - Initialize all LEDs to OFF state

SET recording parameters:
    - FRAME_RATE = 5
    - SECONDS_TO_RECORD_AFTER_DETECTION = 3.0
    - Motion detection thresholds and counters
```

### 2. Arena Configuration
```
DEFINE two detection areas:
    LEFT_ARENA:
        - Recording region: x1_L_rec to x2_L_rec
        - Detection region: x1_L_det to x2_L_det (smaller, more sensitive area)
    
    RIGHT_ARENA:
        - Recording region: x1_R_rec to x2_R_rec  
        - Detection region: x1_R_det to x2_R_det

DEFINE shared Y coordinates for both arenas:
    - y1_rec to y2_rec (recording height)
    - y1_det to y2_det (detection height)
```

### 3. Startup Sequence
```
WAIT for button press to start:
    WHILE button not pressed:
        Flash red LED (standby indicator)
    
    WHEN button pressed:
        Turn off red LED
        Flash green LED (startup confirmation)

CREATE timestamped directory structure:
    - Main folder: "Beecam_RPi9_<timestamp>"
    - Subfolder: "Videos"
    - Initialize log file: "Motion_log.txt"

DEFINE logging functions:
    - update_log_file_start_L/R()
    - update_log_file_end_L/R()

Flash green LED sequence (folder creation confirmation)
```

### 4. Main Detection Loop
```
INITIALIZE state variables:
    - recording = False
    - L_detection = False, R_detection = False  
    - L_detection_counter = 0, R_detection_counter = 0
    - timer_started = False
    - frame_counter = 0

WHILE True:
    Turn on status LED (23)
    
    CAPTURE new frame from camera
    ROTATE frame 180 degrees
    CONVERT to grayscale and apply Gaussian blur
    
    IF first frame:
        Store as reference frame
    ELSE:
        PERFORM motion detection for both arenas:
            Calculate frame difference in detection regions
            Apply threshold (value: 3) to create binary mask
            
        FOR each arena (LEFT and RIGHT):
            IF motion detected (threshold sum > 125):
                INCREMENT detection counter (max 30)
                Turn on green LED, turn off red LED
                Reset opposite arena counter to 0
            ELSE:
                DECREMENT detection counter (min 0)
                Turn on red LED, turn off green LED
            
            IF detection counter >= 7:
                IF not already recording:
                    START recording:
                        Set detection flag = True
                        Create video writer with unique filename
                        Log recording start
                    Reset timer and opposite counter
            
            ELSE IF currently recording:
                IF timer not started:
                    START latency timer
                    Turn on blue LED (latency indicator)
                ELSE IF timer expired (>= 3 seconds):
                    STOP recording:
                        Set detection flag = False
                        Release video file
                        Log recording end
                        Turn off blue LED
        
        IF recording active:
            WRITE current frame to video file
    
    UPDATE reference frame
    
    CHECK exit conditions:
        IF 'q' key pressed OR button pressed:
            BREAK loop
```

### 5. Cleanup Phase
```
CLEANUP GPIO pins
RELEASE video writer
STOP camera
DESTROY OpenCV windows
```

## Key Features

### Motion Detection Algorithm
- **Dual-arena monitoring**: Simultaneous detection in left and right chambers
- **Cumulative detection**: Uses counters instead of instant triggers for stability
- **Hysteresis**: Different thresholds for start (≥7) and stop (countdown) detection
- **Mutual exclusivity**: Only one arena can record at a time

### Recording Logic
- **Latency buffer**: Continues recording 3 seconds after motion stops
- **Unique filenames**: Timestamp-based with conflict resolution
- **Selective recording**: Only records the active detection region (437x437 pixels)

### Visual Feedback System
- **Red LED**: Standby/no motion detected
- **Green LED**: Motion detected/recording active  
- **Blue LED**: Latency period (motion stopped, still recording)
- **Status LED (23)**: System active indicator

### File Management
- **Timestamped directories**: Organized by session date/time
- **Separate video folder**: All recordings stored in "Videos" subfolder
- **Activity logging**: Text log of all recording start/stop events

## Sensitivity Parameters
- **Motion threshold**: 125 (pixel difference sum)
- **Binary threshold**: 3 (grayscale difference)
- **Detection counter threshold**: 7 (frames of consistent motion)
- **Maximum counter value**: 30 (prevents overflow)
- **Post-detection recording**: 3 seconds