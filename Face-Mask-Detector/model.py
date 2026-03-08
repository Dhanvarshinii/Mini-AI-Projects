import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze early layers
    for param in model.parameters():
        param.require_grad = False
    
    # Unfreeze last two blocks
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Modify the fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model