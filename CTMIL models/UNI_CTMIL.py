# -*- coding: utf-8 -*-
"""

@author: HBZH
"""

import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score
#import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ==============================
# 1. Dataset definition
# ==============================
class WSIFeatureDataset(Dataset):
    def __init__(self, feature_dir):
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
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feats, positions, label, data.get('wsi_id', None)

# ==============================
# 2. Transformer layer
# ==============================
class TransLayer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))
        self.dropout = nn.Dropout(0.05)

    def forward(self, x, return_attn=False):
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return (x, attn_weights) if return_attn else x

# ==============================
# 3. CTMIL Model
# ==============================
class CTMIL(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=512, n_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList([TransLayer(hidden_dim) for _ in range(6)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.patch_classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, feats, positions, return_attn=False, return_patch=False, attn_all_layers=False):
        if feats.dim() == 2:
            feats, positions = feats.unsqueeze(0), positions.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        x = self.fc1(feats) + self.pos_embedding(positions)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1).transpose(0, 1)

        attn_weights_list = []
        for i, layer in enumerate(self.layers):
            if return_attn and (attn_all_layers or i == len(self.layers) - 1):
                x, attn_weights = layer(x, return_attn=True)
                attn_weights_list.append(attn_weights)
            else:
                x = layer(x)

        x = x.transpose(0, 1)
        global_logits = self.classifier(self.norm(x[:, 0]))
        if return_patch:
            patch_logits = self.patch_classifier(x[:, 1:])
            if single_sample:
                return global_logits.squeeze(0), patch_logits.squeeze(0)
            return global_logits, patch_logits
        if single_sample:
            return global_logits.squeeze(0)
        return global_logits

# ==============================
# 4. Training routine
# ==============================
def train_CTMIL(args):
    os.makedirs(args.output_root, exist_ok=True)
    seed = 88
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WSIFeatureDataset(args.feature_dir)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(dataset)), dataset.labels))
    train_data, val_data = Subset(dataset, train_idx), Subset(dataset, val_idx)

    train_weights = [dataset.sample_weights[i] for i in train_idx]
    sampler = WeightedRandomSampler(train_weights, len(train_data), replacement=True)
    train_loader = DataLoader(train_data, batch_size=1, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    #eval_loader = DataLoader(train_data, batch_size=1, shuffle=False)

    model = CTMIL(args.feature_dim, args.hidden_dim, args.n_classes).to(device)
    class_weights = dataset.class_weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=(class_weights[1] / class_weights[0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2)

    best_auc, counter, best_threshold = 0.0, 0, 0.5
    results, out_dir = [], args.output_root

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for feats, pos, lbl, _ in train_loader:
            feats, pos, lbl = feats.to(device), pos.to(device), lbl.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            global_logits, patch_logits = model(feats, pos, return_patch=True)

            global_loss = criterion(global_logits, lbl)
            attn_weights = model.layers[-1].attn(feats.transpose(0, 1), feats.transpose(0, 1), feats.transpose(0, 1))[1]
            patch_logits = patch_logits.squeeze(0).squeeze(-1)
            weights = attn_weights.mean(0)[0, 1:]
            weights = weights / (weights.sum() + 1e-8)
            patch_loss = criterion((patch_logits * weights).sum().unsqueeze(0).unsqueeze(0), lbl)
            loss = global_loss + patch_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for feats, pos, lbl, _ in val_loader:
                logits = model(feats.to(device), pos.to(device))
                val_preds.append(torch.sigmoid(logits).item())
                val_labels.append(lbl.item())
        val_auc = roc_auc_score(val_labels, val_preds)
        fpr, tpr, thresholds = roc_curve(val_labels, val_preds)
        best_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[best_idx]
        scheduler.step(val_auc)

        print(f"Epoch {epoch+1}/{args.epochs} | Val AUC: {val_auc:.4f} | Threshold: {optimal_threshold:.3f}")
        results.append({'epoch': epoch + 1, 'val_auc': val_auc, 'threshold': optimal_threshold})

        if val_auc > best_auc:
            best_auc = val_auc
            best_threshold = optimal_threshold
            counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_UNI_CTMIL.pt'))
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    pd.DataFrame(results).to_csv(os.path.join(out_dir, 'training_log.csv'), index=False)
    print(f"Best validation AUC: {best_auc:.4f}")

# ==============================
# 5. Main entry
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=True, help='Directory with extracted features')
    parser.add_argument('--output_root', type=str, default='./CTMIL_output')
    parser.add_argument('--feature_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    train_CTMIL(args)
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

