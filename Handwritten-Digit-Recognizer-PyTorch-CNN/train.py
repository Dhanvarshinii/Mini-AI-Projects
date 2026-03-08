import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import DigitCNN

# Transform images to tensors
transform = transforms.ToTensor()

# Load datasets
train_datasets = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform
)

# Data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_datasets,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

# Initialize model
model = DigitCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5

for epoch in range(epochs):
    
    running_loss = 0.0
    
    for images, labels in train_loader:
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

# Save model    
torch.save(model.state_dict(), "digit_model.pth")

print("Model saved!")

correct = 0
total = 0

model.eval()

with torch.no_grad():
    
    for images, labels in test_loader:
    
        outputs = model(images)
        
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")