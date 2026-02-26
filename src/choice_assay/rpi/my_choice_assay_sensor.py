####################################################################################################
# Sensor class that provides motion-detected video recording using picamera2 and OpenCV with LED indicators.
#
# This implementation includes the LED management functionality from the original Beecam script:
# - Pin 17 (Red): Motion counter status - ON when motion counter below threshold, OFF during detection
# - Pin 22 (Green): Motion detection status - ON during motion detection, OFF otherwise
# - Pin 27 (Blue): Grace period indicator - ON during grace period after motion stops
# - Pin 23: Frame capture indicator - brief pulse for each frame capture
# - Pin 26 (Button): Input for manual control/emergency stop
#
# Key features:
# - Real-time frame capture using picamera2
# - Motion detection using frame differencing with configurable sensitivity
# - Hysteresis-based detection system to prevent false triggers
# - Grace period continues recording briefly after motion stops
# - True frame rate control for accurate temporal analysis
# - Dual-arena mutual exclusivity with separate LED indicators
# - Visual LED feedback matching original Beecam behavior
# - Compatible with existing sensor framework
#
# Motion Detection Algorithm:
# 1. Capture frames at specified frame rate with LED indicator
# 2. Convert to grayscale and apply Gaussian blur
# 3. Calculate frame-to-frame differences for each arena
# 4. Apply binary threshold to create motion mask
# 5. Use cumulative counter system for robust detection with LED feedback
# 6. Start recording when counter exceeds threshold - turn on green LED
# 7. Continue recording during grace period after motion stops - turn on blue LED
#
# LED Behavior:
# - Red LED: Shows when motion counter is below detection threshold
# - Green LED: Shows when motion is actively detected and recording
# - Blue LED: Shows during grace period (continuing recording after motion stops)
# - Frame LED: Brief pulse for each captured frame
####################################################################################################
import time
from dataclasses import dataclass

import cv2
import numpy as np
from expidite_rpi.core import api, file_naming
from expidite_rpi.core import configuration as root_cfg
from expidite_rpi.core.dp_config_objects import Stream
from expidite_rpi.core.sensor import Sensor, SensorCfg

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

try:
    from RPi import GPIO
except ImportError:
    GPIO = None

logger = root_cfg.setup_logger("choice_assay")

# GPIO Pin definitions (from original Beecam script)
RED_LED_PIN = 17  # Motion counter status
GREEN_LED_PIN = 22  # Motion detection active
BLUE_LED_PIN = 27  # Grace period indicator
FRAME_LED_PIN = 23  # Frame capture indicator
BUTTON_PIN = 26  # Emergency stop button

CA_LEFT_VIDEO_DATA_TYPE_ID = "CAVIDEOLEFT"
CA_RIGHT_VIDEO_DATA_TYPE_ID = "CAVIDEORIGHT"
CA_LEFT_VIDEO_STREAM_INDEX: int = 0
CA_RIGHT_VIDEO_STREAM_INDEX: int = 1


@dataclass
class ChoiceAssaySensorCfg(SensorCfg):
    # Camera configuration
    width: int = 1640
    height: int = 1232

    # Motion detection configuration parameters
    motion_threshold: int = 125  # Pixel difference threshold for motion detection
    detection_frames_needed: int = 7  # Consecutive frames needed to confirm motion start
    grace_period_seconds: float = 3.0  # Continue recording after motion stops (seconds)
    frame_rate: int = 5  # Target recording frame rate
    blur_kernel: tuple[int, int] = (21, 21)  # Gaussian blur kernel size for noise reduction
    binary_threshold: int = 3  # Threshold for binary motion mask
    max_detection_counter: int = 30  # Maximum value for detection counter

    # LED configuration
    enable_leds: bool = True  # Enable LED indicators (disable for testing)

    # Dual arena configuration (based on original Beecam script coordinates)
    left_detection_roi: tuple[int, int, int, int] = (210, 373, 560, 578)  # (x1, y1, x2, y2)
    right_detection_roi: tuple[int, int, int, int] = (1170, 373, 1520, 578)  # (x1, y1, x2, y2)
    left_recording_roi: tuple[int, int, int, int] = (164, 167, 601, 604)  # (x1, y1, x2, y2)
    right_recording_roi: tuple[int, int, int, int] = (1124, 167, 1561, 604)  # (x1, y1, x2, y2)


DEFAULT_CA_SENSOR_CFG = ChoiceAssaySensorCfg(
    sensor_type=api.SENSOR_TYPE.CAMERA,
    sensor_index=0,
    sensor_model="PiCameraModule3",
    description="Dual-arena motion-detected video sensor with LED indicators using picamera2 and OpenCV",
    outputs=[
        Stream(
            description="Motion-triggered video recording from left arena.",
            type_id=CA_LEFT_VIDEO_DATA_TYPE_ID,
            index=CA_LEFT_VIDEO_STREAM_INDEX,
            format=api.FORMAT.MP4,
            cloud_container="expidite-upload",
            sample_probability="1.0",
        ),
        Stream(
            description="Motion-triggered video recording from right arena.",
            type_id=CA_RIGHT_VIDEO_DATA_TYPE_ID,
            index=CA_RIGHT_VIDEO_STREAM_INDEX,
            format=api.FORMAT.MP4,
            cloud_container="expidite-upload",
            sample_probability="1.0",
        ),
    ],
    # All other values use dataclass defaults
)


class ChoiceAssaySensorWithLEDs(Sensor):
    def __init__(self, config: ChoiceAssaySensorCfg) -> None:
        """Initialize the ChoiceAssaySensor class with motion detection and LED indicators."""
        super().__init__(config)
        self.config = config
        self.recording_format = self.get_stream(
            CA_LEFT_VIDEO_STREAM_INDEX
        ).format  # Both streams have same format

        # Dual arena motion detection state variables
        self.left_motion_detected = False
        self.right_motion_detected = False
        self.left_detection_counter = 0
        self.right_detection_counter = 0
        self.detection_stopped_time = None
        self.timer_started = False
        self.previous_frame = None
        self.frame_counter = 0

        # Video recording state
        self.video_writer = None
        self.current_recording_start_time = None
        self.current_recording_arena = None  # Track which arena is being recorded
        self.active_arena = None  # 'left', 'right', or None

        # Camera initialization (will be done in run method for proper error handling)
        self.picam2 = None

        # GPIO initialization
        self.gpio_initialized = False

    def _initialize_gpio(self) -> None:
        """Initialize GPIO pins for LED control."""
        if not self.config.enable_leds:
            logger.info("LED control disabled in configuration")
            return

        if not GPIO:
            logger.warning("RPi.GPIO library not available. LED control disabled.")
            return

        try:
            # Set up GPIO mode and pins
            GPIO.setmode(GPIO.BCM)

            # Define LED pins for output
            led_pins = [RED_LED_PIN, GREEN_LED_PIN, BLUE_LED_PIN, FRAME_LED_PIN]
            for pin in led_pins:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)  # Start with LEDs off

            # Set up button pin for input
            GPIO.setup(BUTTON_PIN, GPIO.IN)

            self.gpio_initialized = True
            logger.info("GPIO initialized successfully for LED control")

            # Flash green LED to indicate initialization complete
            self._flash_initialization_sequence()

        except Exception:
            logger.exception("Failed to initialize GPIO")
            self.gpio_initialized = False

    def _flash_initialization_sequence(self) -> None:
        """Flash LEDs in sequence to indicate successful initialization."""
        if not self.gpio_initialized:
            return

        try:
            # Flash green LED 3 times to indicate ready
            for _ in range(3):
                GPIO.output(GREEN_LED_PIN, GPIO.HIGH)
                time.sleep(0.5)
                GPIO.output(GREEN_LED_PIN, GPIO.LOW)
                time.sleep(0.5)
        except Exception:
            logger.exception("Error in LED initialization sequence")

    def _set_led_state(self, pin: int, state: bool) -> None:
        """Safely set LED state with error handling."""
        if not self.gpio_initialized or not GPIO:
            return

        try:
            GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
        except Exception:
            logger.exception("Error setting LED pin %s to %s", pin, state)

    def _pulse_frame_led(self) -> None:
        """Brief pulse on frame LED to indicate frame capture."""
        if not self.gpio_initialized:
            return

        try:
            GPIO.output(FRAME_LED_PIN, GPIO.HIGH)
            time.sleep(0.01)  # Very brief pulse
            GPIO.output(FRAME_LED_PIN, GPIO.LOW)
        except Exception:
            logger.exception("Error pulsing frame LED")

    def _check_emergency_stop(self) -> bool:
        """Check if emergency stop button is pressed."""
        if not self.gpio_initialized:
            return False

        try:
            return GPIO.input(BUTTON_PIN) == GPIO.HIGH
        except Exception:
            logger.exception("Error reading button state")
            return False

    def _cleanup_gpio(self) -> None:
        """Clean up GPIO resources."""
        if self.gpio_initialized and GPIO:
            try:
                GPIO.cleanup()
                logger.info("GPIO cleanup completed")
            except Exception:
                logger.exception("Error during GPIO cleanup")

    def _initialize_camera(self) -> None:
        """Initialize the Picamera2 instance."""
        if not Picamera2:
            msg = "picamera2 library not available. Please install it."
            raise ImportError(msg)

        self.picam2 = Picamera2()

        # Configure camera
        self.picam2.preview_configuration.main.size = (self.config.width, self.config.height)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.preview_configuration.align()
        self.picam2.start()

        logger.info(f"Camera initialized: {self.config.width}x{self.config.height} (with 180° rotation)")

    def _detect_motion_dual_arena(self, current_frame: np.ndarray) -> tuple[bool, bool]:
        """Detect motion in both left and right arenas.

        Returns (left_motion_detected, right_motion_detected).
        """
        config = self.config

        # Convert to grayscale and apply Gaussian blur
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, config.blur_kernel, 0)

        # Skip motion detection on first frame
        if self.previous_frame is None:
            self.previous_frame = blurred_frame
            return False, False

        # Calculate motion in left arena
        x1_l, y1_l, x2_l, y2_l = config.left_detection_roi
        left_diff = cv2.absdiff(
            self.previous_frame[y1_l:y2_l, x1_l:x2_l], blurred_frame[y1_l:y2_l, x1_l:x2_l]
        )
        _, left_threshold = cv2.threshold(left_diff, config.binary_threshold, 255, cv2.THRESH_BINARY)
        left_motion_pixels = left_threshold.sum()
        left_motion = left_motion_pixels > config.motion_threshold

        # Calculate motion in right arena
        x1_r, y1_r, x2_r, y2_r = config.right_detection_roi
        right_diff = cv2.absdiff(
            self.previous_frame[y1_r:y2_r, x1_r:x2_r], blurred_frame[y1_r:y2_r, x1_r:x2_r]
        )
        _, right_threshold = cv2.threshold(right_diff, config.binary_threshold, 255, cv2.THRESH_BINARY)
        right_motion_pixels = right_threshold.sum()
        right_motion = right_motion_pixels > config.motion_threshold

        # Update previous frame
        self.previous_frame = blurred_frame

        return left_motion, right_motion

    def _update_dual_arena_motion_state(
        self, left_motion: bool, right_motion: bool
    ) -> tuple[bool, str | None]:
        """Update motion detection state for dual arenas with mutual exclusivity and LED indicators.

        Returns (should_record, active_arena) where active_arena is 'left', 'right', or None.
        """
        config = self.config

        # Update detection counters
        if left_motion:
            self.left_detection_counter = min(self.left_detection_counter + 1, config.max_detection_counter)
        else:
            self.left_detection_counter = max(self.left_detection_counter - 1, 0)

        if right_motion:
            self.right_detection_counter = min(self.right_detection_counter + 1, config.max_detection_counter)
        else:
            self.right_detection_counter = max(self.right_detection_counter - 1, 0)

        # Update LED states based on motion counters
        # Red LED indicates when motion counter is below threshold (like original Beecam)
        motion_below_threshold = (
            self.left_detection_counter < config.detection_frames_needed
            and self.right_detection_counter < config.detection_frames_needed
        )
        self._set_led_state(RED_LED_PIN, motion_below_threshold)

        # Determine if motion is confirmed in each arena
        left_confirmed = self.left_detection_counter >= config.detection_frames_needed
        right_confirmed = self.right_detection_counter >= config.detection_frames_needed

        # Handle mutual exclusivity - if motion detected in one arena, reset the other
        if left_confirmed:
            self.right_detection_counter = 0  # Reset right arena when left is active
        elif right_confirmed:
            self.left_detection_counter = 0  # Reset left arena when right is active

        # Determine which arena should be active
        target_arena = None
        if left_confirmed:
            target_arena = "left"
        elif right_confirmed:
            target_arena = "right"

        # Handle recording state transitions
        if target_arena:
            # Motion detected in one arena
            if not (self.left_motion_detected or self.right_motion_detected):
                # No previous motion - start recording
                self.left_motion_detected = target_arena == "left"
                self.right_motion_detected = target_arena == "right"
                self.active_arena = target_arena
                self.timer_started = False

                # LED state: Green ON (motion detected), Blue OFF, Red OFF
                self._set_led_state(GREEN_LED_PIN, True)
                self._set_led_state(BLUE_LED_PIN, False)
                self._set_led_state(RED_LED_PIN, False)

                logger.info(f"Motion detected in {target_arena} arena - starting recording")
                return True, target_arena
            if self.active_arena == target_arena:
                # Motion continuing in same arena
                self.timer_started = False  # Reset timer since motion is still active

                # Keep green LED on, turn off blue LED
                self._set_led_state(GREEN_LED_PIN, True)
                self._set_led_state(BLUE_LED_PIN, False)

                return True, target_arena
            # Motion switched to different arena - stop current and start new
            logger.info(f"Motion switched from {self.active_arena} to {target_arena} arena")
            self.left_motion_detected = target_arena == "left"
            self.right_motion_detected = target_arena == "right"
            self.active_arena = target_arena
            self.timer_started = False

            # LED state: Green ON (new motion), Blue OFF
            self._set_led_state(GREEN_LED_PIN, True)
            self._set_led_state(BLUE_LED_PIN, False)

            return True, target_arena

        if self.left_motion_detected or self.right_motion_detected:
            # Motion was active but counters dropped below threshold
            if not self.timer_started:
                # Start grace period timer
                self.timer_started = True
                self.detection_stopped_time = time.time()

                # LED state: Green OFF (no active motion), Blue ON (grace period)
                self._set_led_state(GREEN_LED_PIN, False)
                self._set_led_state(BLUE_LED_PIN, True)

                logger.info(
                    "Motion stopped in %s arena - grace period started (%ss)",
                    self.active_arena,
                    config.grace_period_seconds,
                )
                return True, self.active_arena
            # Check if grace period has expired
            elapsed = time.time() - self.detection_stopped_time
            if elapsed >= config.grace_period_seconds:
                self.left_motion_detected = False
                self.right_motion_detected = False
                self.timer_started = False
                old_arena = self.active_arena
                self.active_arena = None

                # LED state: All detection LEDs OFF (recording stopped)
                self._set_led_state(GREEN_LED_PIN, False)
                self._set_led_state(BLUE_LED_PIN, False)

                logger.info(f"Grace period expired in {old_arena} arena - stopping recording")
                return False, None
            # Still in grace period - blue LED should already be on, no need to set repeatedly
            return True, self.active_arena

        # No motion detected and not in recording state
        # Red LED will be controlled by motion counter logic above
        self._set_led_state(GREEN_LED_PIN, False)
        self._set_led_state(BLUE_LED_PIN, False)

        return False, None

    def _get_stream_index_for_arena(self, arena: str) -> int:
        """Get the appropriate stream index for the given arena."""
        if arena == "left":
            return CA_LEFT_VIDEO_STREAM_INDEX
        # arena == 'right'
        return CA_RIGHT_VIDEO_STREAM_INDEX

    def _start_video_recording(self, filename: str, arena: str) -> None:
        """Start video recording to specified file for the specified arena."""
        if self.video_writer is not None:
            logger.warning("Video recording already active")
            return

        config = self.config
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")

        # Determine frame size based on recording ROI for the active arena
        if arena == "left":
            x1, y1, x2, y2 = config.left_recording_roi
        else:  # arena == 'right'
            x1, y1, x2, y2 = config.right_recording_roi

        frame_size = (x2 - x1, y2 - y1)

        self.video_writer = cv2.VideoWriter(str(filename), fourcc, config.frame_rate, frame_size)

        self.current_recording_start_time = api.utc_now()
        self.current_recording_arena = arena  # Track which arena is being recorded
        logger.info(f"Started video recording for {arena} arena: {filename}, size: {frame_size}")

    def _write_frame_to_video(self, frame: np.ndarray, arena: str) -> None:
        """Write frame to active video recording for the specified arena."""
        if self.video_writer is None:
            return

        config = self.config

        # Extract the recording region for the active arena
        if arena == "left":
            x1, y1, x2, y2 = config.left_recording_roi
        else:  # arena == 'right'
            x1, y1, x2, y2 = config.right_recording_roi

        roi_frame = frame[y1:y2, x1:x2]

        # Convert from RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(roi_frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(bgr_frame)

    def _stop_video_recording(self) -> tuple | None:
        """Stop video recording and return (start_time, end_time) tuple."""
        if self.video_writer is None:
            return None

        self.video_writer.release()
        self.video_writer = None

        end_time = api.utc_now()
        logger.info(f"Stopped video recording for {self.current_recording_arena} arena")

        # Clear arena tracking but save reference for return
        start_time = self.current_recording_start_time
        self.current_recording_arena = None
        self.current_recording_start_time = None

        return start_time, end_time

    def run(self) -> None:
        """Main loop for motion-detected video recording with LED indicators."""
        if not root_cfg.running_on_rpi and root_cfg.TEST_MODE != root_cfg.MODE.TEST:
            logger.warning("Video configuration is only supported on Raspberry Pi.")
            return

        exception_count = 0
        current_filename = None
        last_frame_time = time.time()
        frame_interval = 1.0 / self.config.frame_rate

        try:
            # Initialize GPIO for LED control
            self._initialize_gpio()

            # Initialize camera
            self._initialize_camera()
            logger.info("Motion detection video sensor with LEDs started")

        except Exception:
            logger.exception("Failed to initialize camera or GPIO")
            self.sensor_failed()
            return

        # Main motion detection loop
        while self.continue_recording():
            try:
                # Check emergency stop button
                # @@@ remove
                # if self._check_emergency_stop():
                #    logger.info("Emergency stop button pressed - exiting")
                #    break

                # Maintain consistent frame rate
                current_time = time.time()
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue
                last_frame_time = current_time

                # Pulse frame LED to indicate frame capture
                self._pulse_frame_led()

                # Capture frame from camera
                raw_frame = self.picam2.capture_array()
                self.frame_counter += 1

                # Apply 180-degree rotation (as per original Beecam script)
                frame = np.rot90(raw_frame, k=2)

                # Detect motion in both arenas
                left_motion, right_motion = self._detect_motion_dual_arena(frame)

                # Update dual-arena motion state and determine if recording should be active
                should_record, active_arena = self._update_dual_arena_motion_state(left_motion, right_motion)

                # Handle recording state changes
                if should_record and self.video_writer is None:
                    # Start new recording for the active arena
                    current_filename = file_naming.get_temporary_filename(self.recording_format)
                    self._start_video_recording(current_filename, active_arena)

                elif not should_record and self.video_writer is not None:
                    # Save arena name before stopping recording (since _stop_video_recording clears it)
                    arena_name = self.current_recording_arena

                    # Stop current recording
                    start_time, end_time = self._stop_video_recording()

                    # Save the completed video file
                    if current_filename and start_time and arena_name:
                        stream_index = self._get_stream_index_for_arena(arena_name)
                        self.save_recording(
                            stream_index, current_filename, start_time=start_time, end_time=end_time
                        )
                        logger.info(f"Saved {arena_name} arena video recording: {current_filename}")

                    current_filename = None

                # Write frame to video if recording is active
                if should_record and self.video_writer is not None and self.active_arena is not None:
                    self._write_frame_to_video(frame, self.active_arena)

                # Reset exception count on successful frame processing
                exception_count = 0

            except Exception:
                logger.exception("%sError in motion detection loop", root_cfg.RAISE_WARN())
                exception_count += 1

                # Clean up any active recording on error
                if self.video_writer is not None:
                    try:
                        # Save arena name before it gets cleared by _stop_video_recording
                        arena_name = self.current_recording_arena
                        result = self._stop_video_recording()
                        # Save the recording even on error to avoid losing data
                        if result and current_filename and arena_name:
                            start_time, end_time = result
                            stream_index = self._get_stream_index_for_arena(arena_name)
                            self.save_recording(
                                stream_index, current_filename, start_time=start_time, end_time=end_time
                            )
                            logger.info(
                                f"Emergency saved {arena_name} arena video on error: {current_filename}"
                            )
                    except Exception:
                        logger.exception("Error during emergency video cleanup")

                # On the assumption that the error is transient, we will continue to run but sleep for 5s
                self.stop_requested.wait(5)
                if exception_count > 30:
                    msg = f"ChoiceAssaySensorWithLEDs has failed {exception_count} times. Exiting."
                    logger.exception(msg)
                    self.sensor_failed()
                    break

        # Cleanup on exit
        try:
            if self.video_writer is not None:
                # Save arena name before it gets cleared by _stop_video_recording
                arena_name = self.current_recording_arena
                start_time, end_time = self._stop_video_recording()
                if current_filename and start_time and arena_name:
                    stream_index = self._get_stream_index_for_arena(arena_name)
                    self.save_recording(
                        stream_index, current_filename, start_time=start_time, end_time=end_time
                    )

            if self.picam2 is not None:
                self.picam2.stop()
                logger.info("Camera stopped")

            # Clean up GPIO
            self._cleanup_gpio()

        except Exception:
            logger.exception("Error during cleanup")

        logger.warning("Exiting ChoiceAssaySensorWithLEDs motion detection loop")
