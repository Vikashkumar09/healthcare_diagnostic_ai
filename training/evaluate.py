import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

def evaluate(model, dataloader, device):
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)      # Recall
    specificity = tn / (tn + fp)

    print("\n📊 Evaluation Metrics (Medical Standard)")
    print(f"Accuracy     : {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Precision    : {precision_score(y_true, y_pred):.2f}")
    print(f"Sensitivity  : {sensitivity:.2f}")
    print(f"Specificity  : {specificity:.2f}")
    print(f"F1-score     : {f1_score(y_true, y_pred):.2f}")
    print(f"ROC-AUC      : {roc_auc_score(y_true, y_prob):.2f}")

    print("\n🧮 Confusion Matrix")
    print(f"TP: {tp} | FP: {fp}")
    print(f"FN: {fn} | TN: {tn}")


# =======================
# MAIN EXECUTION BLOCK
# =======================
if __name__ == "__main__":
    import torch.nn as nn
    from torchvision import models
    from training.dataset import get_dataloaders
    import os

    DATA_DIR = "data/brain_mri"
    MODEL_PATH = "models/brain_tumor_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_dataloaders(DATA_DIR)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    evaluate(model, test_loader, device)