#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: HBZH
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data transforms
common_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder("./dataset_for_tumor_detection/training", transform=transform_train)
val_dataset = datasets.ImageFolder("./dataset_for_tumor_detection/validation", transform=common_transforms)
test_dataset = datasets.ImageFolder("./dataset_for_tumor_detection/test", transform=common_transforms)

print("Class mapping:", train_dataset.class_to_idx)

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pretrained InceptionResNetV2 model
model = timm.create_model('inception_resnet_v2', pretrained=True)
model.classif = nn.Linear(model.classif.in_features, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, min_lr=1e-6, verbose=True)

# Training params
num_epochs = 50
patience = 5
best_val_loss = float('inf')
stop_counter = 0
optimal_threshold = 0.5

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    epoch_loss, total_samples = 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        total_samples += labels.size(0)
        all_preds.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy()[:, 1])
        all_labels.extend(labels.cpu().numpy())

    train_loss = epoch_loss / total_samples
    train_preds_binary = (np.array(all_preds) > optimal_threshold).astype(int)
    train_acc = np.mean(train_preds_binary == np.array(all_labels))

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_samples = 0, 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
            val_preds.extend(torch.softmax(outputs, dim=1).cpu().numpy()[:, 1])
            val_labels.extend(labels.cpu().numpy())
            val_samples += labels.size(0)

    fpr, tpr, thresholds = roc_curve(val_labels, val_preds)
    youden_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden_index)]

    val_loss /= val_samples
    val_preds_binary = (np.array(val_preds) > optimal_threshold).astype(int)
    val_acc = np.mean(val_preds_binary == np.array(val_labels))

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Threshold: {optimal_threshold:.4f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        stop_counter = 0
        torch.save(model.state_dict(), 'best_inceptionresnetv2_model.pth')
        with open("optimal_threshold_InceptionResnetV2.txt", "w") as f:
            f.write(str(optimal_threshold))
    else:
        stop_counter += 1
        if stop_counter >= patience:
            print("Early stopping.")
            break

# Plot loss and accuracy
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.savefig('inceptionresnetv2_loss_plot.pdf')

plt.figure()
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.savefig('inceptionresnetv2_accuracy_plot.pdf')


# ===== Evaluation =====

model.load_state_dict(torch.load('best_inceptionresnetv2_model.pth'))
model.eval()
with open("optimal_threshold_InceptionResnetV2.txt", "r") as f:
    threshold = float(f.read())

def evaluate(model, loader, threshold):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[:, 1]
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    preds = (np.array(all_probs) > threshold).astype(int)
    labels = np.array(all_labels)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    metrics = {
        'Accuracy': accuracy_score(labels, preds),
        'Precision': precision_score(labels, preds),
        'Recall': recall_score(labels, preds),
        'F1 Score': f1_score(labels, preds),
        'AUC': roc_auc_score(labels, all_probs),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }
    return metrics

for name, loader in zip(['Train', 'Validation', 'Test'], [train_loader, val_loader, test_loader]):
    results = evaluate(model, loader, threshold)
    print(f"{name} Metrics:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
