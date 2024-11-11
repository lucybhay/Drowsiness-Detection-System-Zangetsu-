from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()

# Load the alert sound file
alert_sound = pygame.mixer.Sound("static/audio/alarm.mp3")

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds for drowsiness detection
eye_threshold = 0.25  # Threshold for detecting closed eyes
consecutive_frames = 3  # Number of consecutive frames to detect drowsiness
duration_threshold = 0.7  # Duration threshold for each eye closure in seconds
no_eyes_alert_threshold = 20  # Number of frames without eyes to trigger alert

# Load face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Indices for the left and right eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Open video capture device
cap = cv2.VideoCapture(0)

# Initialize variables for drowsiness detection
count = 0  # Counter for consecutive frames with closed eyes
no_eyes_count = 0  # Counter for frames without detected eyes
alert_active = False  # Flag to indicate if alert sound is playing
start_time = None  # Variable to store the start time of eye closure

print("Drowsiness detection started. Press 'Esc' or 'q' to exit.")

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=450)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)

    # Reset alert state if no face is detected
    if len(subjects) == 0:
        alert_active = False
        no_eyes_count += 1
    else:
        no_eyes_count = 0

    # Check if eyes are not detected for consecutive frames
    if no_eyes_count >= no_eyes_alert_threshold and not alert_active:
        alert_sound.play()
        alert_active = True
        print("ALERT! Eyes not detected.")

    # Loop over the detected faces
    for subject in subjects:
        # Predict facial landmarks for each face
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract left and right eye landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate eye aspect ratios
        left_ear = eye_aspect_ratio(leftEye)
        right_ear = eye_aspect_ratio(rightEye)

        # Average the eye aspect ratios
        eye_asp_ratio = (left_ear + right_ear) / 2.0

        # Draw bounding boxes around the eyes
        left_eye_rect = cv2.boundingRect(leftEye)
        right_eye_rect = cv2.boundingRect(rightEye)
        cv2.rectangle(frame, (left_eye_rect[0], left_eye_rect[1]), 
                      (left_eye_rect[0] + left_eye_rect[2], left_eye_rect[1] + left_eye_rect[3]), 
                      (0, 255, 0), 2)
        cv2.rectangle(frame, (right_eye_rect[0], right_eye_rect[1]), 
                      (right_eye_rect[0] + right_eye_rect[2], right_eye_rect[1] + right_eye_rect[3]), 
                      (0, 255, 0), 2)

        # Check if the eye aspect ratio falls below the threshold
        if eye_asp_ratio < eye_threshold:
            count += 1
            # If eyes are closed for the first time, start the timer
            if count == 1:
                start_time = time.time()
        else:
            # If eyes are open, reset the counter and timer
            count = 0
            start_time = None

        # If the number of consecutive frames with closed eyes exceeds the threshold
        if count >= consecutive_frames:
            # Check if eye closure duration exceeds the threshold
            if time.time() - start_time >= duration_threshold:
                # Play alert sound if it's not already playing
                if not alert_active:
                    alert_sound.play()
                    alert_active = True
                    print("ALERT! Drowsiness detected.")

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

# Cleanup
print("Drowsiness detection stopped.")
cv2.destroyAllWindows()
cap.release()
