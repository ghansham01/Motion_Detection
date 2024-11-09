import cv2
import numpy as np
import datetime
import time
from threading import Thread, Timer
import pygame
import os
import logging
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize pygame mixer for sound playback
pygame.mixer.init()

class SoundController:
    def __init__(self, sound_file):
        self.sound_file = sound_file
        self.is_playing = False
        self.stop_timer = None
        
        if os.path.exists(sound_file):
            self.sound = pygame.mixer.Sound(sound_file)
        else:
            self.sound = None
            logging.warning(f"Sound file {sound_file} not found")

    def play_sound(self):
        if self.sound and not self.is_playing:
            self.is_playing = True
            self.sound.play(-1)  # Play on loop
            # Set timer to stop after 20 seconds
            self.stop_timer = Timer(20.0, self.stop_sound)
            self.stop_timer.start()
            logging.info("Alert sound started - will play for 20 seconds")

    def stop_sound(self):
        if self.sound and self.is_playing:
            self.sound.stop()
            self.is_playing = False
            if self.stop_timer:
                self.stop_timer.cancel()
                self.stop_timer = None
            logging.info("Alert sound stopped")

def create_camera_url(protocol, ip, port, username=None, password=None, channel=1, stream=1):
    """
    Create camera URL based on protocol and authentication details
    """
    if protocol.lower() == 'rtsp':
        # RTSP URL format
        if username and password:
            auth = f"{urllib.parse.quote(username)}:{urllib.parse.quote(password)}@"
        else:
            auth = ""
        return f"rtsp://{auth}{ip}:{port}/ch{channel}/stream{stream}"
    else:
        # HTTP URL format
        if username and password:
            auth = f"{urllib.parse.quote(username)}:{urllib.parse.quote(password)}@"
        else:
            auth = ""
        return f"http://{auth}{ip}:{port}/video"

def initialize_wifi_camera(camera_config):
    """
    Initialize WiFi camera connection with various protocols
    """
    # Create camera URL based on configuration
    camera_url = create_camera_url(
        protocol=camera_config.get('protocol', 'rtsp'),
        ip=camera_config['ip'],
        port=camera_config.get('port', 554),
        username=camera_config.get('username'),
        password=camera_config.get('password'),
        channel=camera_config.get('channel', 1),
        stream=camera_config.get('stream', 1)
    )
    
    logging.info(f"Connecting to camera: {camera_url.replace(camera_config.get('password', ''), '***')}")
    
    # Configure OpenCV capture
    cap = cv2.VideoCapture(camera_url)
    
    # Set additional camera properties if needed
    if camera_config.get('width'):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
    if camera_config.get('height'):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
    
    if not cap.isOpened():
        logging.error("Error: Could not connect to WiFi camera.")
        return None
        
    return cap

def main(camera_config, motion_area_threshold=500, sound_file='sound.mp3'):
    start_program_time = time.time()
    
    # Initialize WiFi camera
    cap = initialize_wifi_camera(camera_config)
    if cap is None:
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    # Initialize sound controller
    sound_ctrl = SoundController(sound_file)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    motion_detected = False
    video_writer = None
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    motion_start_time = None
    reconnect_attempts = 0
    max_reconnect_attempts = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Could not read frame.")
            reconnect_attempts += 1
            if reconnect_attempts < max_reconnect_attempts:
                logging.info(f"Attempting to reconnect... ({reconnect_attempts}/{max_reconnect_attempts})")
                time.sleep(2)  # Wait before reconnecting
                cap = initialize_wifi_camera(camera_config)
                if cap is None:
                    continue
            else:
                logging.error("Maximum reconnection attempts reached. Exiting.")
                break
            continue
        
        reconnect_attempts = 0  # Reset reconnect attempts on successful frame read

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
                    sound_ctrl.play_sound()

                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if motion_detected and video_writer is not None:
            video_writer.write(frame)

        # Add timestamp and motion status to frame
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Status: Motion Detected" if motion_detected else "Status: No Motion", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if motion_detected else (0, 255, 0), 2)

        cv2.imshow('Motion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()
    sound_ctrl.stop_sound()

    end_program_time = time.time()
    logging.info(f"Total program execution time: {end_program_time - start_program_time} seconds")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Camera configuration examples for different types of cameras:
    
    # Example 1: Generic RTSP camera
    camera_config = {
        'protocol': 'rtsp',
        'ip': '192.168.1.100',  # Replace with your camera's IP
        'port': 554,            # Common RTSP port
        'username': 'admin',    # Replace with your camera's username
        'password': 'password', # Replace with your camera's password
        'channel': 1,
        'stream': 1
    }
    
    # Example 2: Simple HTTP camera
    # camera_config = {
    #     'protocol': 'http',
    #     'ip': '192.168.1.100',
    #     'port': 8080
    # }
    
    # Example 3: DaHua camera
    # camera_config = {
    #     'protocol': 'rtsp',
    #     'ip': '192.168.1.100',
    #     'port': 554,
    #     'username': 'admin',
    #     'password': 'admin',
    #     'channel': 1,
    #     'stream': 1  # 1 for main stream, 2 for sub stream
    # }
    
    # Example 4: Hikvision camera
    # camera_config = {
    #     'protocol': 'rtsp',
    #     'ip': '192.168.1.100',
    #     'port': 554,
    #     'username': 'admin',
    #     'password': '12345',
    #     'channel': 1,
    #     'stream': 1  # 1 for main stream, 2 for sub stream
    # }

    main(camera_config, motion_area_threshold=500, sound_file='sound.mp3')