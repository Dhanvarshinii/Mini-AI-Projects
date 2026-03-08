import torch
import torchvision.transforms as transforms
from PIL import Image

from model import DigitCNN

# Load model
model = DigitCNN()
model.load_state_dict(torch.load("digit_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

def predict_image(image_path):
    
    image = Image.open(image_path)
    
    image = transform(image)
    
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        
        output = model(image)
        
        _, predicted = torch.max(output, 1)
        
    return predicted.item()

img = "digit.jpeg"

prediction = predict_image(img)

print("Predicted Digit:", prediction)