import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model():
    
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Freeze pretrained layers
    for param in model.parameters():
        param.reqires_grade = False
    
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace final layer for 2 classes: Cat/Dog
    num_fltrs = model.fc.in_features
    model.fc = nn.Linear(num_fltrs, 2)
    return model
    