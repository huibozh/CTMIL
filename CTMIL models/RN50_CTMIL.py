# -*- coding: utf-8 -*-
"""

@author: HBZH
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
#import pandas as pd
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau

def set_seed(seed=88):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class WSIFeatureDataset(Dataset):
    def __init__(self, feature_dir):
        super().__init__()
        self.files = []
        self.labels = []

        class_mapping = {'A_nLNM': 0, 'B_LNM': 1}
        for class_name in sorted(os.listdir(feature_dir)):
            class_path = os.path.join(feature_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            label = class_mapping.get(class_name)
            if label is None:
                continue
            for fn in sorted(os.listdir(class_path)):
                if fn.endswith(".pt"):
                    self.files.append(os.path.join(class_path, fn))
                    self.labels.append(label)

        class_counts = np.bincount(self.labels)
        self.class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
        self.sample_weights = [self.class_weights[label] for label in self.labels]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        feats = data['features'].float()
        positions = data['positions'].float()

        epsilon = 1e-7
        positions[:, 0] /= (positions[:, 0].max() + epsilon)
        positions[:, 1] /= (positions[:, 1].max() + epsilon)

        return feats, positions, torch.tensor(self.labels[idx], dtype=torch.long), data.get('wsi_id', None)

class TransLayer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(0.05)

    def forward(self, x, return_attn=False):
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return (x, attn_weights) if return_attn else x

class CTMIL(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512, n_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList([TransLayer(hidden_dim) for _ in range(6)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.patch_classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, feats, positions, return_attn=False, return_patch=False):
        single_sample = feats.dim() == 2
        if single_sample:
            feats, positions = feats.unsqueeze(0), positions.unsqueeze(0)

        x = self.fc1(feats) + self.pos_embedding(positions)
        B = x.size(0)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1).transpose(0, 1)

        attn_weights_list = []
        for i, layer in enumerate(self.layers):
            if return_attn and i == len(self.layers) - 1:
                x, attn_weights = layer(x, return_attn=True)
                attn_weights_list.append(attn_weights)
            else:
                x = layer(x)

        x = x.transpose(0, 1)
        cls_out = self.norm(x[:, 0, :])
        global_logits = self.classifier(cls_out)

        if return_patch:
            patch_logits = self.patch_classifier(x[:, 1:, :])
            if single_sample:
                patch_logits = patch_logits.squeeze(0)

        if single_sample:
            global_logits = global_logits.squeeze(0)

        if return_patch and return_attn:
            return global_logits, patch_logits, attn_weights_list
        elif return_patch:
            return global_logits, patch_logits
        elif return_attn:
            return global_logits, attn_weights_list
        return global_logits

def train_CTMIL(feature_dir, output_dir, seed=88, feature_dim=2048, hidden_dim=512,
                   n_classes=1, lr=1e-4, num_epochs=50, patience=5):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WSIFeatureDataset(feature_dir)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(dataset)), dataset.labels))

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_labels = [dataset.labels[i] for i in train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(sample_weights, len(train_subset), replacement=True)
    train_loader = DataLoader(train_subset, batch_size=1, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

    model = CTMIL(feature_dim, hidden_dim, n_classes).to(device)
    pos_weight = class_weights[1] / class_weights[0]
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_auc = 0
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for feats, pos, label, _ in train_loader:
            feats, pos = feats.to(device), pos.to(device)
            label = label.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            global_logits, patch_logits, attn = model(feats, pos, return_patch=True, return_attn=True)
            global_loss = criterion(global_logits, label)

            patch_logits = patch_logits.squeeze(0).squeeze(-1)
            attn_scores = attn[-1].mean(1) if attn[-1].dim() == 4 else attn[-1]
            weights = attn_scores[0, 0, 1:]
            weights = weights / (weights.sum() + 1e-8)
            patch_score = (patch_logits * weights).sum().unsqueeze(0).unsqueeze(0)
            patch_loss = criterion(patch_score, label)

            loss = global_loss + patch_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for feats, pos, label, _ in val_loader:
                feats, pos = feats.to(device), pos.to(device)
                label = label.float().to(device).unsqueeze(1)
                logits = model(feats, pos)
                val_preds.append(torch.sigmoid(logits).item())
                val_labels.append(label.item())

        val_auc = roc_auc_score(val_labels, val_preds)
        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Training complete. Best AUC: {best_auc:.4f}")
    return best_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--feature_dim', type=int, default=2048)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    train_CTMIL(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        seed=88,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        n_classes=args.n_classes,
        lr=args.lr,
        num_epochs=args.epochs,
        patience=args.patience
    )

if __name__ == '__main__':
    main()

