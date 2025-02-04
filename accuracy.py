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

class MotionMetrics:
    def __init__(self):
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.last_motion_state = False
        self.consecutive_frames_threshold = 10  # Increased for stability
        self.consecutive_frames = 0

    def calculate_accuracy(self):
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if total == 0:
            return 0
        return (self.true_positives + self.true_negatives) / total

    def update_metrics(self, motion_detected, actual_motion):
        if motion_detected:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = 0

        stable_motion = self.consecutive_frames >= self.consecutive_frames_threshold

        if stable_motion and actual_motion:
            self.true_positives += 1
        elif not stable_motion and not actual_motion:
            self.true_negatives += 1
        elif stable_motion and not actual_motion:
            self.false_positives += 1
        elif not stable_motion and actual_motion:
            self.false_negatives += 1

def play_sound_non_blocking(sound_file):
    if os.path.exists(sound_file):
        sound = pygame.mixer.Sound(sound_file)
        sound.play()
    else:
        logging.warning(f"Sound file {sound_file} not found.")

def initialize_video_capture():
    cap2 = cv2.VideoCapture(0)
    if not cap2.isOpened():
        logging.error("Error: Could not open video capture." )
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
    metrics = MotionMetrics()
    last_accuracy_print = time.time()

    # Create window with trackbar for simulating actual motion
    cv2.namedWindow('Motion Detection')
    cv2.createTrackbar('Actual Motion', 'Motion Detection', 0, 1, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Could not read frame.")
            break

        # Get actual motion state from trackbar (for testing)
        actual_motion = bool(cv2.getTrackbarPos('Actual Motion', 'Motion Detection'))

        fgmask = fgbg.apply(frame)
        _, thresh = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_motion = False
        for contour in contours:
            if cv2.contourArea(contour) > motion_area_threshold:
                current_motion = True
                if not motion_detected:
                    motion_detected = True
                    motion_start_time = time.time()
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    video_writer = cv2.VideoWriter(f"motion_{timestamp}.avi", fourcc, 20, size)
                    logging.info(f"Motion detected, starting recording: motion_{timestamp}.avi")
                    play_sound_non_blocking(sound_file)

                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                

        # Update metrics
        metrics.update_metrics(current_motion, actual_motion)

        # Display accuracy every second
        current_time = time.time()
        # if current_time - last_accuracy_print >=0.5:  #updates every half secand  more frequent updates but more resource intensive
        # if current_time - last_accuracy_print >=2.0:  #updates every tow secand   less frequent updates but more resource efficient
        if current_time - last_accuracy_print >= 1.0:
            accuracy = metrics.calculate_accuracy()
            accuracy_text = f"Accuracy: {accuracy:.2%}"
            metrics_text = f"TP: {metrics.true_positives} TN: {metrics.true_negatives} FP: {metrics.false_positives} FN: {metrics.false_negatives}"
            cv2.putText(frame, accuracy_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, metrics_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            last_accuracy_print = current_time
            logging.info(f"{accuracy_text} | {metrics_text}")

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
    logging.info(f"Final Accuracy: {metrics.calculate_accuracy():.2%}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()