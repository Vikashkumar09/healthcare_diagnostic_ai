from training.dataset import get_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
import os

print("🔥 Training started")

DATA_DIR = "data/brain_mri"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader, test_loader = get_dataloaders(DATA_DIR)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last ResNet block + FC
for name, param in model.named_parameters():
    if name.startswith("layer4") or name.startswith("fc"):
        param.requires_grad = True

print("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)   # <-- THIS WAS MISSING / BROKEN

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "brain_tumor_model.pth"))
print("🎉 Training completed and model saved")