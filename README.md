# 🚀 Motion Detection System

## 📌 Overview
This is a **real-time motion detection system** using **OpenCV** and **background subtraction**. When motion is detected, the program:
- Starts video recording.
- Highlights moving objects with bounding boxes.
- Plays a sound alert (if a sound file is available).
- Saves the recorded video with a timestamp.

## 🛠️ Technologies Used
- **Python**
- **OpenCV** (for motion detection)
- **NumPy**
- **Pygame** (for playing alert sounds)
- **Threading** (for efficient sound handling)
- **Logging** (for tracking system activity)

## ⚡ Features
✅ Real-time motion detection using contour analysis  
✅ Video recording on motion detection  
✅ Sound alerts when motion is detected  
✅ Adjustable motion sensitivity  
✅ Logs activity for easy debugging  

## 🚀 Installation & Setup
1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/motion-detection.git
   cd motion-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install opencv-python numpy pygame
   ```

3. **Run the script**:
   ```bash
   python new.py
   ```

## 🎯 Usage
- The program starts detecting motion as soon as it's launched.
- Press **'q'** to quit the program.
- Detected motion triggers recording and an alert sound.

## 🔧 Configuration
You can modify the script parameters:
- **Motion Area Threshold** (default: `500`): Change `motion_area_threshold` in `main()` to adjust sensitivity.
- **Sound Alert**: Replace `sound.mp3` with your preferred sound file.

