from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

app = FastAPI(title="Brain Tumor Detection API")

# -----------------------------
# -----------------------------
# Load Model
# -----------------------------
import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = "models/brain_tumor_model.pth"

# 1️⃣ Create model FIRST
model = models.resnet18(weights=None)

# 2️⃣ Modify final layer
model.fc = nn.Linear(model.fc.in_features, 2)

# 3️⃣ Load weights
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

# 4️⃣ Set eval mode
model.eval()
# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/")
def root():
    return {"message": "Brain Tumor Detection API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0][predicted_class].item()

    label = "Tumor" if predicted_class == 1 else "No Tumor"

    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }