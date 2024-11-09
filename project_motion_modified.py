import cv2
import numpy as np
import datetime
import time
from threading import Thread
import pygame

# Initialize pygame mixer for non-blocking sound playback
pygame.mixer.init()

def play_sound_non_blocking(sound_file):
    def play():
        pygame.mixer.Sound(sound_file).play()
    thread = Thread(target=play)
    thread.start()

# Measure program execution time
start_program_time = time.time()

# capturing video
cap = cv2.VideoCapture(0)  
# cap2 = cv2.VideoCapture(1)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

motion_detected = False
video_writer = None
min_area = 500 

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

motion_start_time = None
motion_end_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Remove shadows (optional) by thresholding the mask
    _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)

    # Find contours of the detected motion
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any significant motion is detected
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            if not motion_detected:
                motion_detected = True
                # Record the start time of motion detection
                motion_start_time = time.time()
                # Create a new video file with a timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                video_writer = cv2.VideoWriter(f"motion_{timestamp}.avi", fourcc, 20, size)
                print(f"Motion detected, starting recording: motion_{timestamp}.avi")

            # Draw a rectangle around the detected motion
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Play sound in non-blocking mode
            play_sound_non_blocking('sound.mp3')

    # If motion is detected, write the frame to the video file
    if motion_detected and video_writer is not None:
        video_writer.write(frame)
        # Reset motion detection after some time if no motion
        motion_end_time = time.time()
        print(f"Motion duration: {motion_end_time - motion_start_time} seconds")

    # Display the video feed with motion detection
    cv2.imshow('Motion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if video_writer is not None:
    video_writer.release()

# Calculate total execution time
end_program_time = time.time()
print(f"Total program execution time: {end_program_time - start_program_time} seconds")

cv2.destroyAllWindows()