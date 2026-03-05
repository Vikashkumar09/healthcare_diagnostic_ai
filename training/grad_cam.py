import torchvision.models as models
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import sys

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Forward hook
        def forward_hook(module, input, output):
            self.activations = output

        # Backward hook (FULL backward hook – correct way)
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        output[:, class_idx].backward()

        # Global Average Pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


import torchvision.models as models
import torchvision.transforms as transforms
import os

def run_gradcam(image_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load(
        r"D:\healthcare_diagnostic_ai\models\brain_tumor_model.pth",
        map_location=device
    ))

    model.to(device)
    model.eval()

    target_layer = model.layer4
    gradcam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    cam = gradcam.generate(input_tensor)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    cv2.imwrite("gradcam_output.jpg", overlay)
    print("✅ Grad-CAM saved as gradcam_output.jpg")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m training.grad_cam <image_path>")
        sys.exit(1)

    run_gradcam(sys.argv[1])