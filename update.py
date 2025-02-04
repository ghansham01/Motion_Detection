import cv2
import numpy as np
import datetime
import pygame
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize pygame mixer for non-blocking sound playback
pygame.mixer.init()

# Load the alert sound
# ALERT_SOUND_PATH = 'alert.wav'  # Replace with the path to your alert sound file
# if os.path.exists(ALERT_SOUND_PATH):
#     alert_sound = pygame.mixer.Sound(ALERT_SOUND_PATH)
# else:
#     alert_sound = None
#     logging.warning("Alert sound file not found. Sound alerts will be disabled.")

class MotionMetrics:
    def __init__(self):
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def calculate_accuracy(self):
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total != 0 else 0

    def update_metrics(self, detected, actual):
        if detected:
            if actual:
                self.true_positives += 1
            else:
                self.false_positives += 1
        else:
            if actual:
                self.false_negatives += 1
            else:
                self.true_negatives += 1

class MotionDetector:
    def __init__(self):
        # Background subtractor for adaptive background modeling
        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
        self.metrics = MotionMetrics()
        self.save_video = False

    def detect_motion(self, frame):
        fg_mask = self.back_sub.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Minimum area filter
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if self.save_video:
                    self.record_video(frame)

                # Play alert sound if motion is detected and sound file is available
                # if alert_sound:
                #     alert_sound.play()

        return motion_detected

    def record_video(self, frame):
        if not hasattr(self, "video_writer"):
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                f"motion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi",
                fourcc, 20.0, (frame.shape[1], frame.shape[0])
            )
        self.video_writer.write(frame)

    def release_video(self):
        if hasattr(self, "video_writer"):
            self.video_writer.release()
            delattr(self, "video_writer")

    def process_frame(self, frame, actual_motion=False):
        motion_detected = self.detect_motion(frame)
        self.metrics.update_metrics(motion_detected, actual_motion)
        accuracy = self.metrics.calculate_accuracy()
        logging.info(f"Detection Accuracy: {accuracy:.2f}")

# Running a sample motion detection (for testing purposes)
def run_motion_detection():
    cap = cv2.VideoCapture(0)  # Capture from the first webcam
    detector = MotionDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detector.process_frame(frame)

        cv2.imshow("Motion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.release_video()
    cap.release()
    cv2.destroyAllWindows()

# Uncomment the line below to test the motion detection in a local environment.
# run_motion_detection()
# if __name__=="__main__":