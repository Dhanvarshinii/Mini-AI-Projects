import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from model import DigitCNN
from streamlit_drawable_canvas import st_canvas

# Load model
model = DigitCNN()
model.load_state_dict(torch.load("digit_model.pth", map_location=torch.device("cpu")))
model.eval()

# Title
st.title("Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) below and the AI will predict it!")

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Transfrom image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])
        
# Streamlit prediction
if st.button("Predict Digit"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype("uint8")
        image = Image.fromarray(img)
        image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            
        st.success(f"Prediction: {predicted.item()}")