import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
from model import get_model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
])

# Load datasets
train_data = datasets.ImageFolder(root="dataset/train", transform=train_transform)
valid_data = datasets.ImageFolder(root="dataset/validation", transform=test_transform)
test_data = datasets.ImageFolder(root="dataset/test", transform=test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Initialize model, loss, optimizer
model = get_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for i, (images,labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if (i+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
    print(f"Epoch [{epoch+1}/{epochs}] completed. Avg Loss: {running_loss/len(train_loader):.4f}")
    
# Save model
torch.save(model.state_dict(), "face_mask_model.pth")
print("Model saved!")

# Testr accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(f"Test Accuracy: {100 * correct / total:.2f}%")
