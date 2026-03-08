import torch 
from torchvision import transforms
from PIL import Image
from model import get_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = get_model()
model.load_state_dict(torch.load("face_mask_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

# Predict function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        prob = torch.softmax(output, dim=1)[0][pred].item() * 100
        label = "Mask" if pred.item() == 0 else "NoMask"
        print(f"Prediction: {label}, Confidence: {prob:.2f}%")
        
predict("test_face.png")
