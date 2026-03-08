import torch
from PIL import Image
from torchvision import transforms
from model import get_model

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = get_model()
model.load_state_dict(torch.load("catdog_model.pth", map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load image
img_path = "test_image.jpg"
image = Image.open(img_path)
image = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image)
    _, pred = torch.max(output, 1)
    label = "Cat" if pred.item() == 0 else "Dog"
    print(f"Prediction: {label}")