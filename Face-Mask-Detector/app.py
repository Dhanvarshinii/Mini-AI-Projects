import cv2.data
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import cv2
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = get_model()
model.load_state_dict(torch.load("face_mask_model.pth", map_location= device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

st.title("Face Mask Detector")

# Start webcam
run = st.checkbox("Start Webcam")

# OpenCV face detector
face_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0) 
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to get Frame")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # For each face detected
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            
            # Transform and predict
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                _, pred = torch.max(output, 1)
                prob = torch.softmax(output, dim=1)[0][pred].item() * 100
                label = "Mask" if pred.item() == 0 else "NoMask"
                
            # Put label on Frame
            cv2.rectangle(frame, (x,y), (x+w, h+h), (0,255,0), 2)
            cv2.putText(frame, f"{label}: {prob:.2f}%", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        # Display frame in Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()