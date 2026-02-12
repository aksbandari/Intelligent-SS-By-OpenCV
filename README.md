# Intelligent-SS-By-OpenCV
Intelligent Security System using OpenCV for real-time video surveillance. It performs face detection and recognition, motion detection, and automated alerts to enhance security. The system efficiently processes live camera feeds, logs events, and is scalable for future integrations like notifications, cloud storage, and AI-based threat analysis.
Intelligent Security System

An Intelligent Security System built using OpenCV for real-time video surveillance. The system performs face detection and recognition, motion detection, and automated alert generation to enhance physical security.


Features
Real-time video stream processing
Face detection & recognition
Motion detection for abnormal activity
Automated alerts on detection
Event logging with timestamps
Scalable and modular design


Tech Stack
Python
OpenCV
NumPy

System Pipeline (Flow)
Camera Feed
   ↓
Frame Capture
   ↓
Preprocessing (Resize, Grayscale, Noise Reduction)
   ↓
Motion Detection
   ↓
Face Detection
   ↓
Face Recognition
   ↓
Decision Engine
   ↓
Event Logging + Alerts

Data Processing Pipeline
Input Video → Frame Extraction → Feature Detection → Classification → Action Trigger

Use Cases
Home surveillance
Office and campus security
Restricted-area monitoring
Future Enhancements
SMS / Email / App notifications
Cloud-based video storage
AI-based threat & behavior analysis
Multi-camera support

How to Run
Install required Python dependencies
Connect webcam or CCTV feed
Execute the main Python script

Accuracy = 95%
(20 / 21)preductions = 95 
Accuracy drops to 81% in low light
