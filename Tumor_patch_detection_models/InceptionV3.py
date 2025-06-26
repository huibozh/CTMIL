# -*- coding: utf-8 -*-
"""
training, validation

@author: HBZH
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from torchvision.models.inception import InceptionOutputs

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data transforms
common_transform = transforms.Compose([
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
val_dataset = datasets.ImageFolder("./dataset_for_tumor_detection/validation", transform=common_transform)
test_dataset = datasets.ImageFolder("./dataset_for_tumor_detection/test", transform=common_transform)

print("Class mapping:", train_dataset.class_to_idx)

# DataLoaders
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pretrained InceptionV3 and modify classifier
model = models.inception_v3(pretrained=True, aux_logits=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, min_lr=1e-6, verbose=True)

# Training setup
num_epochs = 50
patience = 5
best_val_loss = float('inf')
stop_counter = 0
optimal_threshold = 0.5

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        if isinstance(outputs, InceptionOutputs):
            main_output, aux_output = outputs.logits, outputs.aux_logits
            loss = criterion(main_output, labels) + 0.4 * criterion(aux_output, labels)
            probs = torch.softmax(main_output, dim=1)[:, 1]
        else:
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:, 1]

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs.logits if isinstance(outputs, InceptionOutputs) else outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    binary_preds = (np.array(all_probs) > optimal_threshold).astype(int)
    epoch_acc = np.mean(binary_preds == np.array(all_labels))

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Validation
    model.eval()
    val_loss, val_total = 0, 0
    val_probs, val_labels_all = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits = outputs.logits if isinstance(outputs, InceptionOutputs) else outputs
            val_loss += criterion(logits, labels).item() * inputs.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]
            val_probs.extend(probs.cpu().numpy())
            val_labels_all.extend(labels.cpu().numpy())
            val_total += labels.size(0)

    # Optimal threshold using Youden's index
    fpr, tpr, thresholds = roc_curve(val_labels_all, val_probs)
    youden = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden)]

    val_epoch_loss = val_loss / val_total
    val_preds = (np.array(val_probs) > optimal_threshold).astype(int)
    val_epoch_acc = np.mean(val_preds == np.array(val_labels_all))

    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
          f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f} | Threshold: {optimal_threshold:.4f}")

    scheduler.step(val_epoch_loss)

    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        stop_counter = 0
        torch.save(model.state_dict(), "best_inceptionv3_model.pth")
        with open("optimal_threshold_InceptionV3.txt", "w") as f:
            f.write(str(optimal_threshold))
    else:
        stop_counter += 1
        if stop_counter >= patience:
            print("Early stopping.")
            break

# Plot loss
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.savefig('InceptionV3_loss_plot.pdf')

# Plot accuracy
plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.savefig('InceptionV3_accuracy_plot.pdf')

# ================= Prediction =================

# Load best model
model.load_state_dict(torch.load("best_inceptionv3_model.pth"))
model.eval()
with open("optimal_threshold_InceptionV3.txt", "r") as f:
    threshold = float(f.read())
print(f"Loaded optimal threshold: {threshold:.4f}")

# Evaluation function
def evaluate(model, loader, threshold):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits = outputs.logits if isinstance(outputs, InceptionOutputs) else outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()[:, 1]
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    preds = (np.array(all_probs) > threshold).astype(int)
    labels = np.array(all_labels)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {
        'Accuracy': accuracy_score(labels, preds),
        'Precision': precision_score(labels, preds),
        'Recall': recall_score(labels, preds),
        'F1 Score': f1_score(labels, preds),
        'AUC': roc_auc_score(labels, all_probs),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    }

# Evaluate on train/val/test
for name, loader in zip(["Train", "Validation", "Test"], [train_loader, val_loader, test_loader]):
    metrics = evaluate(model, loader, threshold)
    print(f"{name} Set Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

