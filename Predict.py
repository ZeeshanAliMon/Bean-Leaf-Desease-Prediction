import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

class_names = ["angular_leaf_spot", "bean_rust", "healthy"]

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("beans_resnet18.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


image_path = input("Enter image path: ")


image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print("Prediction:", class_names[predicted.item()])