import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # YOLOv8 nano model

# Title
st.title("🔍 AI-Based Detection System")

# Sidebar for selection
option = st.sidebar.selectbox(
    "Choose a Detection Mode:",
    ("Object Detection (Image)", "Object Detection (Video)", "Real-Time Object Detection")
)

# 🖼️ IMAGE OBJECT DETECTION
if option == "Object Detection (Image)":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")  # Convert to RGB
        image = np.array(image)

        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Process Image"):
            st.write("⚙️ Processing...")

            results = model(image)
            detected_image = image.copy()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                    label = f"{model.names[int(box.cls[0])]}: {box.conf[0]:.2f}"

                    # Draw bounding box
                    cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(detected_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the result
            st.image(detected_image, caption="Detected Objects", use_container_width=True)

# 🎥 VIDEO OBJECT DETECTION (Uploaded Video)
elif option == "Object Detection (Video)":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file:
        temp_dir = tempfile.TemporaryDirectory()
        video_path = os.path.join(temp_dir.name, uploaded_file.name)

        # Save uploaded video
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("⚙️ Processing Video... Please wait.")

        cap = cv2.VideoCapture(video_path)
        video_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            # Draw bounding boxes
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = f"{model.names[int(box.cls[0])]}: {box.conf[0]:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame, channels="RGB", use_container_width=True)

        cap.release()
        st.write("🎯 Video Processing Completed!")

# 📷 REAL-TIME OBJECT DETECTION (Webcam)
elif option == "Real-Time Object Detection":
    st.write("🔴 **Live Stream Running...** Press 'Stop Live Stream' to exit.")

    cap = cv2.VideoCapture(0)  # Open webcam
    video_placeholder = st.empty()

    stop_button = st.button("Stop Live Stream")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Camera Not Detected")
            break

        results = model(frame)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{model.names[int(box.cls[0])]}: {box.conf[0]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB", use_container_width=True)

    cap.release()
    st.write("🛑 **Live Stream Stopped**")
