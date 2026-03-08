import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import get_model

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = get_model()
model.load_state_dict(torch.load("catdog_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform for input image
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

# Streamlit UI
st.title("Cats vs Dogs Classifier")

uploaded_file = st.file_uploader("Choose a cat or dog image..", type =["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Prediction
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        prob = torch.softmax(output, dim=1)
        confidence = prob[0][pred].item() * 100
        label = "Cat" if pred.item() == 0 else "Dog"
    
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
        
    