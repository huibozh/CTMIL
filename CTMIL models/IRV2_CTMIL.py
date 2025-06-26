# -*- coding: utf-8 -*-
"""

@author: HBZH
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

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

            label = class_mapping.get(class_name, None)
            if label is None:
                continue

            for fn in sorted(os.listdir(class_path)):
                if fn.endswith(".pt"):
                    fp = os.path.join(class_path, fn)
                    self.files.append(fp)
                    self.labels.append(label)

        class_counts = np.bincount(self.labels)
        self.class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
        self.sample_weights = [self.class_weights[label] for label in self.labels]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_dict = torch.load(self.files[idx])
        feats = data_dict['features'].float()
        positions = data_dict['positions'].float()

        epsilon = 1e-7
        max_x = positions[:, 0].max() + epsilon
        max_y = positions[:, 1].max() + epsilon
        positions[:, 0] /= max_x
        positions[:, 1] /= max_y

        return feats, positions, torch.tensor(self.labels[idx], dtype=torch.long), data_dict.get('wsi_id', None)

class TransLayer(nn.Module):
    def __init__(self, dim=512):
        super(TransLayer, self).__init__()
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
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        if return_attn:
            return x, attn_weights
        else:
            return x

class CTMIL(nn.Module):
    def __init__(self, feature_dim=1536, hidden_dim=512, n_classes=1):
        super(CTMIL, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList([TransLayer(dim=hidden_dim) for _ in range(6)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.patch_classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, feats, positions, return_attn=False, return_patch=False, attn_all_layers=False):
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
            positions = positions.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        x = self.fc1(feats)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x.transpose(0, 1)

        attn_weights_list = []
        for i, layer in enumerate(self.layers):
            if return_attn:
                if attn_all_layers or i == len(self.layers) - 1:
                    x, attn_weights = layer(x, return_attn=True)
                    attn_weights_list.append(attn_weights)
                else:
                    x = layer(x)
            else:
                x = layer(x)

        x = x.transpose(0, 1)
        cls_out = x[:, 0, :]
        cls_out = self.norm(cls_out)
        global_logits = self.classifier(cls_out)

        if return_patch:
            patch_tokens = x[:, 1:, :]
            patch_logits = self.patch_classifier(patch_tokens)

        if single_sample:
            global_logits = global_logits.squeeze(0)
            if return_patch:
                patch_logits = patch_logits.squeeze(0)

        if return_patch and return_attn:
            return global_logits, patch_logits, attn_weights_list
        elif return_patch:
            return global_logits, patch_logits
        elif return_attn:
            return global_logits, attn_weights_list
        else:
            return global_logits

def train_CTMIL(feature_dir, output_root, seed, feature_dim=1536, hidden_dim=512, 
                   n_classes=1, lr=1e-4, num_epochs=50, patience=5):
    seed_dir = os.path.join(output_root, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = WSIFeatureDataset(feature_dir)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_indices, val_indices = next(sss.split(np.zeros(len(full_dataset)), full_dataset.labels))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_labels = [full_dataset.labels[i] for i in train_indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler)
    train_eval_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = CTMIL(feature_dim=feature_dim, hidden_dim=hidden_dim, n_classes=n_classes).to(device)
    pos_weight = class_weights[1] / class_weights[0]
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_auc = 0.0
    counter = 0
    best_threshold = 0.5
    metrics = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for feats, positions, label, _ in train_loader:
            feats = feats.to(device)
            positions = positions.to(device)
            label = label.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            global_logits, patch_logits, attn_weights_list = model(feats, positions, return_patch=True, return_attn=True)
            global_loss = criterion(global_logits, label)

            patch_logits = patch_logits.squeeze(0).squeeze(-1)
            attn_scores = attn_weights_list[-1]
            if attn_scores.dim() == 4:
                attn_scores = attn_scores.mean(dim=1)
            patch_weights = attn_scores[0, 0, 1:]
            patch_weights = patch_weights / (patch_weights.sum() + 1e-8)
            weighted_patch_logit = (patch_logits * patch_weights).sum().unsqueeze(0).unsqueeze(0)
            patch_loss = criterion(weighted_patch_logit, label)

            loss = global_loss + patch_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        train_preds_eval = []
        train_labels_eval = []
        with torch.no_grad():
            for feats, positions, label, _ in train_eval_loader:
                feats = feats.to(device)
                positions = positions.to(device)
                label = label.float().to(device).unsqueeze(1)
                logits = model(feats, positions)
                train_preds_eval.append(torch.sigmoid(logits).item())
                train_labels_eval.append(label.item())
        train_auc_eval = roc_auc_score(train_labels_eval, train_preds_eval)

        val_loss = 0.0
        all_preds_val = []
        all_labels_val = []
        with torch.no_grad():
            for feats, positions, label, _ in val_loader:
                feats = feats.to(device)
                positions = positions.to(device)
                label = label.float().to(device).unsqueeze(1)
                logits = model(feats, positions)
                loss = criterion(logits, label)
                val_loss += loss.item()
                all_preds_val.append(torch.sigmoid(logits).item())
                all_labels_val.append(label.item())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(all_labels_val, all_preds_val)
        fpr, tpr, thresholds = roc_curve(all_labels_val, all_preds_val)
        youden_index = tpr - fpr
        optimal_threshold = thresholds[np.argmax(youden_index)]
        scheduler.step(val_auc)

        print(f"[Seed {seed} Epoch {epoch+1}/{num_epochs}] TrainLoss={train_loss:.4f} (Train AUC {train_auc_eval:.4f}) | ValLoss={val_loss:.4f} (Val AUC {val_auc:.4f}) | OptimalTh={optimal_threshold:.3f}")

        metrics.append({'seed': seed, 'epoch': epoch+1, 'train_auc': train_auc_eval, 'val_auc': val_auc, 'threshold': optimal_threshold})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_threshold = optimal_threshold
            counter = 0
            torch.save(model.state_dict(), os.path.join(seed_dir, 'best_CTMIL_IRV2_6layers0.pt'))
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}, best val AUC: {best_val_auc:.4f}")
                break

        plt.figure()
        plt.plot(fpr, tpr, label=f'Val AUC={val_auc:.3f}')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Seed {seed} Epoch {epoch+1} ROC')
        plt.legend()
        plt.savefig(os.path.join(seed_dir, f'epoch_{epoch+1}_roc.png'))
        plt.close()

    pd.DataFrame(metrics).to_csv(os.path.join(seed_dir, 'training_metrics_6layers0.csv'), index=False)

    return {
        'seed': seed,
        'best_train_auc': max([m['train_auc'] for m in metrics]),
        'best_val_auc': best_val_auc,
        'final_threshold': best_threshold
    }

def main():
    parser = argparse.ArgumentParser(description="Train CTMIL with global and patch-level classifiers.")
    parser.add_argument("--feature_dir", default="./features_CTMIL_IRV2", help="Root directory containing class folders")
    parser.add_argument("--output_root", default="./attentionMIL_results/CTMIL_IRV2_6layers", help="Root directory for output")
    parser.add_argument("--feature_dim", type=int, default=1536)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_classes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    seed = 88
    print(f"\n{'='*30}\n=== Running seed {seed} ===\n{'='*30}")
    result = train_CTMIL(
        feature_dir=args.feature_dir,
        output_root=args.output_root,
        seed=seed,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        n_classes=args.n_classes,
        lr=args.lr,
        num_epochs=args.epochs,
        patience=args.patience
    )
    pd.DataFrame([result]).to_csv(os.path.join(args.output_root, 'CTMIL_IRV2_6layers_result.csv'), index=False)
    print("\nExperiment completed. Results saved to:", args.output_root)

if __name__ == "__main__":
    main()

