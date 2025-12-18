Title: DeepFake Detection in Video Conferencing Environments

Introduction: As remote work and digital communication become the norm, the threat of real-time identity manipulation ("Deepfakes") in video calls is rising. This project implements a lightweight, high-accuracy deep learning model designed to detect facial manipulation in live video streams. Unlike standard detectors that process pre-recorded videos, this system is optimized for low latency, making it suitable for integration with platforms like Zoom, Google Meet, or Skype via virtual camera streams.

Key Features:

•	Real-Time Analysis: Processes video frames on-the-fly with minimal lag.
•	Frame-by-Frame Detection: Uses a CNN-based architecture (trained on Celeb-DF v2) to classify individual frames as "Real" or "Fake."
•	Face Tracking: Automatically detects and crops faces using OpenCV to focus the model's attention.
•	User Dashboard: Interactive web interface built with Streamlit for easy testing and visualization.
•	Confidence Scoring: visualizes the probability of manipulation for each frame.

Architecture:

1.	Input: Live video feed from webcam or OBS virtual camera.
2.	Preprocessing: Faces are detected using Haar Cascades/MTCNN, cropped, and resized to (128, 128) or (224, 224).
3.	Inference: The processed frames are passed to a custom .h5 model (based on Xception/ResNet/MobileNet).
4.	Output: A probability score is generated. If the "Fake" probability exceeds a set threshold (e.g., 0.5), the system alerts the user.

