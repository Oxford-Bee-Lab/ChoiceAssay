import cv2
from picamera2 import Picamera2

picam2 = Picamera2()
import datetime
import os
import time

import numpy as np
import RPi.GPIO as GPIO

# ==============================


def get_mp4_NameWithDate(nameIn=".mp4"):
    """Needs a file ending on .mp4, inserts _<date> before .mp4.
    If file exists, it appends a additional _number after the <date>
    ensuring filename uniqueness at this time."""
    if not nameIn.endswith(".mp4"):
        raise ValueError("filename must end in .mp4")
    filename = nameIn.replace(".mp4", "_{0}.mp4").format(
        datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    )
    if os.path.isfile(filename):  # if already exists
        fn2 = filename[0:-4] + "_{0}.mp4"  # modify pattern to include a number
        count = 1
        while os.path.isfile(fn2.format(count)):  # increase number until file not exists
            count += 1
        return fn2.format(count)  # return file with number in it

    else:  # filename ok, return it
        return filename


# SETTING UP VIDEO STUFF
# Initialise capture
# print("Initialising capture")
picam2.preview_configuration.main.size = (1640, 1232)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
# picam2.configure("preview")
picam2.start()
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# print(cap.isOpened())


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# -------------------------
###Set up GPIOs
GPIO.setmode(GPIO.BCM)
# Assign pins
LED_LIST = [17, 27, 22, 23]
BUTTON_PIN = 26
# Set them up
for pin in LED_LIST:
    GPIO.setup(pin, GPIO.OUT)
GPIO.setup(BUTTON_PIN, GPIO.IN)
# Initialise LEDs (power off to start with)
for pin in LED_LIST:
    GPIO.output(pin, GPIO.LOW)
# print("GPIOs initialised")

# >> Anything else may need to install <<
# print("Initialisation complete")

# ============================
# Set parameters
FRAME_RATE = 5
# frame_size = (int(cap.get(3)),int(cap.get(4))) #get frame size tuple (needs to be the same as the video capture device) (can be reshaped later if necessary)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Define the codec

# Parameters to set up so that can continue recording for a bit after bee stops moving/leaves frame
recording = False
L_detection = False
R_detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 3.0

frame_counter = 0  # Used to initialise the program - can't do anything with the very first frame but compare frames sequentially after that
L_detection_counter = 0
R_detection_counter = 0

print("NOTE TO USER: Press 'q' to exit the program safely.")

# RECORDING FRAME PARAMETERS
x1_L_rec = 164  # Top left corner: x coordinate
x2_L_rec = 601  # Top left corner: y coordinate (down)
x1_R_rec = 1124
x2_R_rec = 1561

y1_rec = 167  # Top corner: x coordinate
y2_rec = 604  # Bottom corner: y coordinate

# DETECTION FRAME PARAMETERS
x1_L_det = 210  # 308
x2_L_det = 560  # 463
x1_R_det = 1170  # 1208
x2_R_det = 1520  # 1362

y1_det = 373
y2_det = 578

# -------------------------------------------------------------
# WAIT FOR BUTTON TO BE PRESSED TO START THE PROGRAM
while True:
    time.sleep(0.01)
    GPIO.output(17, GPIO.HIGH)  # Power on red pin
    if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
        GPIO.output(17, GPIO.LOW)  # Power off red pin
        GPIO.output(22, GPIO.HIGH)  # Switch green pin on
        time.sleep(1)
        break
GPIO.output(22, GPIO.LOW)  # Switch green pin off

# ============================================================
# ==== ENTER NEW FOLDER NAME====
# RECORD TO NEW FOLDER
newpath = "/home/RPi9/Documents/Beecam_RPi9_" + datetime.datetime.now().strftime(
    "%d-%m-%Y_%H-%M-%S"
)  ###EDIT THIS BEFORE RUNNING PROGRAM
if not os.path.exists(newpath):
    os.makedirs(newpath)
    os.chdir(newpath)
#    print("Directory set")
videofolderpath = newpath + "/Videos"
if not os.path.exists(videofolderpath):
    os.makedirs(videofolderpath)
#    print("Video folder created")
# INITIALISE LOG FILE
LOG_FILE_NAME = newpath + "/Motion_log.txt"  # Initialise log file
# Create/reset log file
if os.path.exists(LOG_FILE_NAME):
    os.remove(LOG_FILE_NAME)


#    print("Log file reset")
# else:
#    print("Creating log file")
# Create log file function
def update_log_file_start_L():
    with open(LOG_FILE_NAME, "a") as f:
        # Opens the file and adds a new line to it
        f.write(
            "Video recording in left arena started at "
            + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        f.write("\n")


def update_log_file_end_L():
    with open(LOG_FILE_NAME, "a") as f:
        # Opens the file and adds a new line to it
        f.write(
            "Video recording in left arena ended at " + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        f.write("\n")


def update_log_file_start_R():
    with open(LOG_FILE_NAME, "a") as f:
        # Opens the file and adds a new line to it
        f.write(
            "Video recording in right arena started at "
            + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        f.write("\n")


def update_log_file_end_R():
    with open(LOG_FILE_NAME, "a") as f:
        # Opens the file and adds a new line to it
        f.write(
            "Video recording in right arena ended at " + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        f.write("\n")


# print("Log functions set")

# Make green LED flash to indicate folder created
GPIO.output(22, GPIO.HIGH)  # Switch green pin on
time.sleep(0.5)
GPIO.output(22, GPIO.LOW)  # Switch green pin off
time.sleep(0.5)
GPIO.output(22, GPIO.HIGH)  # Switch green pin on
time.sleep(0.5)
GPIO.output(22, GPIO.LOW)  # Switch green pin off
time.sleep(0.5)
GPIO.output(22, GPIO.HIGH)  # Switch green pin on
time.sleep(0.5)
GPIO.output(22, GPIO.LOW)  # Switch green pin off
time.sleep(2)

while True:
    GPIO.output(23, GPIO.HIGH)
    time.sleep(0.01)
    new_frame = picam2.capture_array()
    #    cv2.imshow("live", new_frame)
    #    ret, new_frame = cap.read()
    frame_counter += 1
    new_frame = rotate_image(new_frame, 180)
    #    cv2.rectangle(new_frame, (x1_L_rec, y1_rec), (x2_L_rec, y2_rec), (255, 255, 255), 2) #Image frame: top left corner (x,y), bottom right (x,y)
    #    cv2.rectangle(new_frame, (x1_R_rec, y1_rec), (x2_R_rec, y2_rec), (255, 255, 255), 2)
    #    cv2.rectangle(new_frame, (x1_L_det, y1_det), (x2_L_det, y2_det), (0, 255, 255), 2) #Detection frame
    #    cv2.rectangle(new_frame, (x1_R_det, y1_det), (x2_R_det, y2_det), (0, 255, 255), 2)
    new_frame_bw = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    new_frame_bwgb = cv2.GaussianBlur(new_frame_bw, (21, 21), 0)  # IN THE YOUTUBE VID THIS WAS SET TO (5, 5)
    #    cv2.imshow("live video", new_frame) #Change to 'threshold' on later line to see black and white vector changes; 'new_frame' to see unedited image
    if frame_counter == 1:
        old_frame_bwgb = new_frame_bwgb
    else:  # Motion ROI: 150 tall, 80 wide; filming ROI: 170 tall, 120 wide
        difference_L = cv2.absdiff(
            old_frame_bwgb[y1_det:y2_det, x1_L_det:x2_L_det], new_frame_bwgb[y1_det:y2_det, x1_L_det:x2_L_det]
        )  # Calculate difference bw old and new frames in left ROI
        difference_R = cv2.absdiff(
            old_frame_bwgb[y1_det:y2_det, x1_R_det:x2_R_det], new_frame_bwgb[y1_det:y2_det, x1_R_det:x2_R_det]
        )  # Do the same for the right ROI
        threshold_L = cv2.threshold(difference_L, 3, 255, cv2.THRESH_BINARY)[
            1
        ]  # Modify the 25 threshold: change to lower value makes more sensitive
        threshold_R = cv2.threshold(difference_R, 3, 255, cv2.THRESH_BINARY)[1]
        subimg_L = new_frame[y1_rec:y2_rec, x1_L_rec:x2_L_rec]  # 170 tall, 170 wide,
        subimg_R = new_frame[y1_rec:y2_rec, x1_R_rec:x2_R_rec]
        old_frame_bwgb = new_frame_bwgb

        if (
            threshold_L.sum() > 125
        ):  # If the images are over a certain threshold difference... ###EDIT THIS TO CHANGE SENSITIVITY (default = 300) - tweak in situ w apparatus
            L_detection_counter += 1  # Add one to the motion counter
            GPIO.output(17, GPIO.LOW)  # Power off red pin
            GPIO.output(22, GPIO.HIGH)  # Switch green pin on
            if L_detection_counter >= 30:
                L_detection_counter = 30  # Set so can't go above a certain number - gives buffer for motion detection but means counter doesn't skyrocket
        elif threshold_L.sum() < 126:
            if L_detection_counter > 0:  # ...and the detection counter is not already 0...
                L_detection_counter -= 1  # ...remove one from the counter unless 0 (counter thus cumulative and modifiable - allows dynamic triggering and customisable permissiveness)
                GPIO.output(17, GPIO.HIGH)  # Power on red pin
                GPIO.output(22, GPIO.LOW)  # Switch green pin off

        if (
            threshold_R.sum() > 125
        ):  # If the images are over a certain threshold difference... ###EDIT THIS TO CHANGE SENSITIVITY (default = 300) - tweak in situ w apparatus
            R_detection_counter += 1  # Add one to the motion counter
            GPIO.output(17, GPIO.LOW)  # Power off red pin
            GPIO.output(22, GPIO.HIGH)  # Switch green pin on
            if R_detection_counter >= 30:
                R_detection_counter = 30  # Set so can't go above a certain number - gives buffer for motion detection but means counter doesn't skyrocket
        elif threshold_R.sum() < 126:
            if R_detection_counter > 0:  # ...and the detection counter is not already 0...
                R_detection_counter -= 1  # ...remove one from the counter unless 0 (counter thus cumulative and modifiable - allows dynamic triggering and customisable permissiveness)
                GPIO.output(17, GPIO.HIGH)  # Power on red pin
                GPIO.output(22, GPIO.LOW)  # Switch green pin off

        # DETERMINING WHETHER MOTION FROM MOTION COUNTER
        if (
            L_detection_counter >= 7
        ):  # Regardless of whether those two specific images were much different, if the detection counter is high enough:
            R_detection_counter = 0
            if L_detection:  # If detection mode already active, just keep latency timer at 0
                timer_started = False
                R_detection_counter = 0
            else:
                L_detection = True  # When first crossing the threshold, set detection mode to active
                #                print("Motion detected in the left chamber!")
                out = cv2.VideoWriter(
                    str(newpath) + "/Videos/" + get_mp4_NameWithDate("left.mp4"), fourcc, 5, (437, 437)
                )  # Make output stream (to write out content to and to be closed when we're done recording the file)
                #                print("Recording in left chamber starting")
                update_log_file_start_L()
        elif L_detection:  # If motion counter below the threshold but detection mode active
            GPIO.output(17, GPIO.HIGH)  # Power on red pin
            GPIO.output(22, GPIO.LOW)  # Switch green pin off
            if timer_started:  # and if latency timer already started
                if (
                    time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION
                ):  # Monitor when latency timer exceeds threshold
                    #                    print("No motion detected for 3s")
                    L_detection = False  # Exit motion mode
                    timer_started = False  # Reset latency timer
                    out.release()  # Release the output stream: stop writing to file, and save it
                    GPIO.output(27, GPIO.LOW)
                    #                    print("Stopped recording in left chamber")
                    update_log_file_end_L()
            else:
                timer_started = True  # If detection mode was active but motion counter below threshold and latency timer had not been started
                GPIO.output(27, GPIO.HIGH)  # Power on blue latency pin
                detection_stopped_time = (
                    time.time()
                )  # Take timestamp that motion counter dipped below threshold

        if (
            R_detection_counter >= 7
        ):  # Regardless of whether those two specific images were much different, if the detection counter is high enough:
            L_detection_counter = 0
            if R_detection:  # If detection mode already active, just keep latency timer at 0
                timer_started = False
                L_detection_counter = 0
            else:
                R_detection = True  # When first crossing the threshold, set detection mode to active
                L_detection_counter = 0
                #                print("Motion detected in the right chamber!")
                out = cv2.VideoWriter(
                    str(newpath) + "/Videos/" + get_mp4_NameWithDate("right.mp4"), fourcc, 5, (437, 437)
                )  # Make output stream (to write out content to and to be closed when we're done recording the file)
                #                print("Recording in right chamber starting")
                update_log_file_start_R()
        elif R_detection:  # If motion counter below the threshold but detection mode active
            GPIO.output(17, GPIO.HIGH)  # Power on red pin
            GPIO.output(22, GPIO.LOW)  # Switch green pin off
            if timer_started:  # and if latency timer already started
                if (
                    time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION
                ):  # Monitor when latency timer exceeds threshold
                    #                    print("No motion detected for 3s")
                    R_detection = False  # Exit motion mode
                    timer_started = False  # Reset latency timer
                    out.release()  # Release the output stream: stop writing to file, and save it
                    GPIO.output(27, GPIO.LOW)
                    #                    print("Stopped recording in right chamber")
                    update_log_file_end_R()
            else:
                timer_started = True  # If detection mode was active but motion counter below threshold and latency timer had not been started
                GPIO.output(27, GPIO.HIGH)  # Power on blue latency pin
                detection_stopped_time = (
                    time.time()
                )  # Take timestamp that motion counter dipped below threshold

        if L_detection:
            #            print(L_detection_counter)
            out.write(subimg_L)
        #            cv2.imshow("live video", subimg_L)
        elif R_detection:
            #            print(R_detection_counter)
            out.write(subimg_R)
        #            cv2.imshow("live video", subimg_R)
        #        else:
        #            cv2.imshow("live video", new_frame)
        # End the program with q key or button pressed
        if cv2.waitKey(1) == ord("q"):
            break
        if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
            break


GPIO.cleanup()
# print("GPIOs cleared")
out.release()
picam2.stop()
# cap.release()
# print("Camera released")
cv2.destroyAllWindows()
