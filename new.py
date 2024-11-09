import cv2
import numpy as np
import datetime
import time
from threading import Thread
import pygame
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize pygame mixer for non-blocking sound playback
pygame.mixer.init()

def play_sound_non_blocking(sound_file):
    if os.path.exists(sound_file):
        sound = pygame.mixer.Sound(sound_file)
        sound.play()
        # pygame.time.delay(.5)  # Wait for the specified duration
    else:
        logging.warning(f"Sound file {sound_file} not found.")

def initialize_video_capture():
    cap2 = cv2.VideoCapture(0)
    if not cap2.isOpened():
        logging.error("Error: Could not open video capture.")
        return None
    return cap2

def main(motion_area_threshold=500, sound_file='sound.mp3'):
    start_program_time = time.time()
    
    cap = initialize_video_capture()
    if cap is None:
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    motion_detected = False
    video_writer = None
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    motion_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Could not read frame.")
            break

        fgmask = fgbg.apply(frame)
        _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > motion_area_threshold:
                if not motion_detected:
                    motion_detected = True
                    motion_start_time = time.time()
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    video_writer = cv2.VideoWriter(f"motion_{timestamp}.avi", fourcc, 20, size)
                    logging.info(f"Motion detected, starting recording: motion_{timestamp}.avi")
                    play_sound_non_blocking(sound_file)

                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if motion_detected and video_writer is not None:
            video_writer.write(frame)

        cv2.imshow('Motion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()

    end_program_time = time.time()
    logging.info(f"Total program execution time: {end_program_time - start_program_time} seconds")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()