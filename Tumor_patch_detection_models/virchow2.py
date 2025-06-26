# -*- coding: utf-8 -*-
"""

@author: HBZH
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomRotation, ColorJitter
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Seed setup
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# HuggingFace login 
from huggingface_hub import login
login()  #token

# Load Virchow2 pretrained model
model = timm.create_model(
    "hf-hub:paige-ai/Virchow2", 
    pretrained=True, 
    mlp_layer=SwiGLUPacked, 
    act_layer=nn.SiLU
)

# Data transforms
config = resolve_data_config(model.pretrained_cfg, model=model)
base_transform = create_transform(**config)
train_transform = Compose([
    Resize((518, 518)),
    RandomHorizontalFlip(0.5),
    RandomRotation(15),
    ColorJitter(0.2, 0.2, 0.2, 0.1),
    base_transform
])
eval_transform = Compose([
    Resize((518, 518)),
    base_transform
])

# Load datasets
train_dataset = datasets.ImageFolder('dataset_for_tumor_detection/training/', transform=train_transform)
val_dataset = datasets.ImageFolder('dataset_for_tumor_detection/validation/', transform=eval_transform)
test_dataset = datasets.ImageFolder('dataset_for_tumor_detection/test/', transform=eval_transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

print("Class mapping:", train_dataset.class_to_idx)

# Modify classifier
model.reset_classifier(num_classes=2, global_pool='avg')

# Freeze all layers except classifier
for name, param in model.named_parameters():
    param.requires_grad = any(x in name for x in ['classifier', 'fc', 'head'])

for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=50)

# Training parameters
num_epochs = 50
patience = 5
best_val_loss = float('inf')
early_stop_counter = 0
optimal_threshold = 0.5

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    epoch_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []

    with tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}") as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})

    train_loss = epoch_loss / total
    train_binary_preds = (np.array(all_probs) > optimal_threshold).astype(int)
    train_acc = np.mean(train_binary_preds == np.array(all_labels))
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    scheduler.step()

    # Validation
    model.eval()
    val_loss, val_probs, val_labels = 0.0, [], []
    with torch.no_grad():
        with tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_loss += loss.item() * inputs.size(0)

    # Youden index
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    youden = tpr - fpr
    optimal_threshold = np.clip(thresholds[np.argmax(youden)], 0.3, 0.7)

    val_binary_preds = (np.array(val_probs) > optimal_threshold).astype(int)
    val_acc = np.mean(val_binary_preds == np.array(val_labels))
    val_avg_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_avg_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
          f"Val Loss={val_avg_loss:.4f}, Val Acc={val_acc:.4f}, Threshold={optimal_threshold:.4f}")

    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "Virchow2_best_model.pth")
        with open("optimal_threshold_Virchow2.txt", "w") as f:
            f.write(str(optimal_threshold))
        print("Model saved.")
    else:
        early_stop_counter += 1
        print(f"No improvement for {early_stop_counter} epoch(s).")

    if early_stop_counter >= patience:
        print("Early stopping.")
        break

# Plot loss and accuracy
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Virchow2 Training Loss')
plt.legend()
plt.savefig('Virchow2_loss_plot.pdf')

plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Virchow2 Training Accuracy')
plt.legend()
plt.savefig('Virchow2_accuracy_plot.pdf')



# === Evaluation ===
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from huggingface_hub import login
login()

# Load trained model
model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=False, mlp_layer=SwiGLUPacked, act_layer=nn.SiLU)
model.reset_classifier(num_classes=2, global_pool='avg')
model.load_state_dict(torch.load("Virchow2_best_model.pth"), strict=False)
model.to(device)
model.eval()

# Load transform and datasets
config = resolve_data_config(model.pretrained_cfg, model=model)
transform = create_transform(**config)

train_dataset = datasets.ImageFolder('dataset_for_tumor_detection/training/', transform=transform)
val_dataset = datasets.ImageFolder('dataset_for_tumor_detection/validation/', transform=transform)
test_dataset = datasets.ImageFolder('dataset_for_tumor_detection/test/', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Metric function
def calculate_metrics(model, loader, threshold):
    model.eval()
    probs, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            probs.extend(prob)
            labels.extend(y.cpu().numpy())

    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=1),
        "recall": recall_score(labels, preds, zero_division=1),
        "f1": f1_score(labels, preds, zero_division=1),
        "auc": roc_auc_score(labels, probs),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0
    }, probs, labels

# Evaluate
train_metrics, train_probs, train_labels = calculate_metrics(model, train_loader, 0.5)
val_metrics, val_probs, val_labels = calculate_metrics(model, val_loader, 0.5)
test_metrics, test_probs, test_labels = calculate_metrics(model, test_loader, 0.5)

# ROC-based optimal threshold
fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
youden = tpr - fpr
optimal_threshold = np.clip(thresholds[np.argmax(youden)], 0.3, 0.7)
print(f"Optimal Threshold (Clipped): {optimal_threshold:.4f}")

# Evaluate with optimal threshold
train_opt, _, _ = calculate_metrics(model, train_loader, optimal_threshold)
val_opt, _, _ = calculate_metrics(model, val_loader, optimal_threshold)
test_opt, _, _ = calculate_metrics(model, test_loader, optimal_threshold)

print("\nMetrics with Optimal Threshold:")
print(f"Train: {train_opt}")
print(f"Validation: {val_opt}")
print(f"Test: {test_opt}")

# Histogram
plt.hist(val_probs, bins=50, color='blue', alpha=0.7, label='Validation Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Validation Prediction Distribution')
plt.legend()
plt.show()


