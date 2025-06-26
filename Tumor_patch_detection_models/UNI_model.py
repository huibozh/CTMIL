#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: HBZH
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Resize, Compose, RandomHorizontalFlip, RandomRotation, ColorJitter
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# === Seed setup for reproducibility ===
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# === HuggingFace login 
from huggingface_hub import login
login()  #token

# === Load UNI model ===
model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
config = resolve_data_config(model.pretrained_cfg, model=model)
base_transform = create_transform(**config)

# === Data Transforms ===
train_transform = Compose([
    Resize((512, 512)),
    RandomHorizontalFlip(0.5),
    RandomRotation(15),
    ColorJitter(0.2, 0.2, 0.2, 0.1),
    base_transform
])
eval_transform = Compose([
    Resize((512, 512)),
    base_transform
])

# === Dataset and Dataloader ===
train_dataset = datasets.ImageFolder('dataset_for_tumor_detection/training/', transform=train_transform)
val_dataset = datasets.ImageFolder('dataset_for_tumor_detection/validation/', transform=eval_transform)
test_dataset = datasets.ImageFolder('dataset_for_tumor_detection/test/', transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

print("Class mapping:", train_dataset.class_to_idx)

# === Model Preparation ===
model.reset_classifier(num_classes=2, global_pool='avg')
for name, param in model.named_parameters():
    param.requires_grad = any(key in name for key in ['classifier', 'fc', 'head'])
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# === Optimizer & Scheduler ===
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=50)
criterion = nn.CrossEntropyLoss()

# === Training Settings ===
num_epochs = 50
patience = 5
best_val_loss = float('inf')
early_stop_counter = 0
optimal_threshold = 0.5
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []

    with tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}") as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})

    scheduler.step()
    epoch_loss = running_loss / total
    binary_preds = (np.array(all_probs) > optimal_threshold).astype(int)
    epoch_acc = np.mean(binary_preds == np.array(all_labels))
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # === Validation ===
    model.eval()
    val_loss, val_total = 0, 0
    val_probs, val_labels = [], []

    with torch.no_grad(), tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}") as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            val_probs.extend(probs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            val_loss += loss.item() * inputs.size(0)
            val_total += labels.size(0)
            pred = outputs.argmax(dim=1)
            acc = (pred == labels).sum().item() / labels.size(0)
            pbar.set_postfix({'val_loss': loss.item(), 'val_acc': acc})

    # === Threshold optimization using Youden Index ===
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    youden_index = tpr - fpr
    optimal_threshold = np.clip(thresholds[np.argmax(youden_index)], 0.3, 0.7)
    val_avg_loss = val_loss / val_total
    val_binary_preds = (np.array(val_probs) > optimal_threshold).astype(int)
    val_acc = np.mean(val_binary_preds == np.array(val_labels))
    val_losses.append(val_avg_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
          f"Val Loss: {val_avg_loss:.4f} | Val Acc: {val_acc:.4f} | Threshold: {optimal_threshold:.4f}")

    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "UNI_best_model.pth")
        with open("optimal_threshold_UNI.txt", "w") as f:
            f.write(str(optimal_threshold))
        print("Model saved.")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping.")
            break

# === Plotting ===
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('UNI Training Loss')
plt.legend()
plt.savefig('UNI_loss_plot.pdf')

plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('UNI Training Accuracy')
plt.legend()
plt.savefig('UNI_accuracy_plot.pdf')

# === Evaluation Function ===
def calculate_metrics(model, loader, threshold):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = (np.array(all_probs) > threshold).astype(int)
    labels = np.array(all_labels)
    tn, fp, fn, tp = confusion_matrix(labels, all_preds).ravel()

    return {
        "accuracy": accuracy_score(labels, all_preds),
        "precision": precision_score(labels, all_preds, zero_division=1),
        "recall": recall_score(labels, all_preds, zero_division=1),
        "f1": f1_score(labels, all_preds, zero_division=1),
        "auc": roc_auc_score(labels, all_probs),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0
    }

# === Load model for final evaluation ===
model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=False, init_values=1e-5, dynamic_img_size=True)
model.reset_classifier(num_classes=2, global_pool='avg')
model.load_state_dict(torch.load("UNI_best_model.pth"))
model.to(device)

with open("optimal_threshold_UNI.txt", "r") as f:
    threshold = float(f.read())
print(f"Loaded optimal threshold: {threshold:.4f}")

# === Evaluate on train, val, test ===
train_metrics = calculate_metrics(model, train_loader, threshold)
val_metrics = calculate_metrics(model, val_loader, threshold)
test_metrics = calculate_metrics(model, test_loader, threshold)

# === Print results ===
print("\nFinal Metrics with Optimal Threshold:")
print(f"Train: {train_metrics}")
print(f"Val: {val_metrics}")
print(f"Test: {test_metrics}")

# === Prediction Histogram ===
from matplotlib import pyplot as plt
val_preds = []
with torch.no_grad():
    for inputs, _ in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        val_preds.extend(probs.cpu().numpy())

plt.hist(val_preds, bins=50, alpha=0.75, color='blue', label='Validation Predictions')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Validation Predictions Distribution')
plt.legend()
plt.show()
