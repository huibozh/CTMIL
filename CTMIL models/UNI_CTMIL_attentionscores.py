#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# HBZH
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


seed = 88
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Dataset definition

class WSIFeatureDataset(Dataset):
    def __init__(self, feature_dir):
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
            for fn in os.listdir(class_path):
                if fn.endswith(".pt"):
                    self.files.append(os.path.join(class_path, fn))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_dict = torch.load(self.files[idx])
        feats = data_dict['features'].float()
        positions = data_dict['positions'].float()
        original_positions = positions.clone()
        epsilon = 1e-7
        positions[:, 0] /= (positions[:, 0].max() + epsilon)
        positions[:, 1] /= (positions[:, 1].max() + epsilon)
        return feats, positions, torch.tensor(self.labels[idx], dtype=torch.long), data_dict.get('wsi_id', None), original_positions


# 2. Transformer encoder layer

class TransLayer(nn.Module):
    def __init__(self, dim=512):
        super(TransLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=0.1, batch_first=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))
        self.dropout = nn.Dropout(0.05)

    def forward(self, x, return_attn=False):
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=True)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return (x, attn_weights) if return_attn else x


# 3. CTMIL model

class CTMIL(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=512, n_classes=1):
        super(CTMIL, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList([TransLayer(hidden_dim) for _ in range(6)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.patch_classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, feats, positions, return_attn=False, return_patch=False, attn_all_layers=True):
        single_sample = False
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
            positions = positions.unsqueeze(0)
            single_sample = True

        x = self.fc1(feats) + self.pos_embedding(positions)
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1).transpose(0, 1)

        attn_weights_list = []
        for i, layer in enumerate(self.layers):
            if return_attn and (attn_all_layers or i == len(self.layers) - 1):
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
                global_logits = global_logits.squeeze(0)
                patch_logits = patch_logits.squeeze(0)
            return (global_logits, patch_logits, attn_weights_list) if return_attn else (global_logits, patch_logits)
        else:
            global_logits = global_logits.squeeze(0) if single_sample else global_logits
            return (global_logits, attn_weights_list) if return_attn else global_logits



# 4. Attention scores for visualization

def save_attention_scores_rollout(data_loader, dataset_obj, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    for i, (patches, positions, label, extra, original_positions) in enumerate(data_loader):
        wsi_id = str(extra[0]) if extra and extra[0] else os.path.basename(os.path.dirname(dataset_obj.files[i]))
        output_file = os.path.join(output_dir, f"{wsi_id}_CTMIL_rollout_scores.xls")

        patches = patches.squeeze(0).to(device)
        positions = positions.squeeze(0).to(device)
        original_positions = original_positions.squeeze(0)

        try:
            with torch.no_grad():
                _, patch_logits, attn_weights_list = model(patches, positions, return_attn=True, return_patch=True)
                patch_logits = patch_logits.squeeze(0).squeeze(-1).cpu().numpy()

                rollout = None
                for attn in attn_weights_list:
                    attn_mat = attn.mean(dim=1)[0] if attn.dim() == 4 else attn[0]
                    attn_mat += torch.eye(attn_mat.size(0), device=attn_mat.device)
                    attn_mat /= attn_mat.sum(dim=-1, keepdim=True)
                    rollout = attn_mat if rollout is None else rollout @ attn_mat
                cls_rollout = rollout[0, 1:].cpu().numpy()

                last_attn = attn_weights_list[-1]
                patch_attn_scores = last_attn.mean(dim=1)[0][0, 1:].cpu().numpy() if last_attn.dim() == 4 else last_attn[0][0, 1:].cpu().numpy()

                single_logits, single_probs = [], []
                for j in range(patches.shape[0]):
                    logit = model(patches[j:j+1].unsqueeze(0), positions[j:j+1].unsqueeze(0))
                    logit = logit.squeeze().item()
                    single_logits.append(logit)
                    single_probs.append(torch.sigmoid(torch.tensor(logit)).item())

                pos_np = positions.cpu().numpy()
                orig_np = original_positions.cpu().numpy()
                patch_names = [f"{int(pos[0])}_{int(pos[1])}_{wsi_id}" for pos in orig_np]

                df = pd.DataFrame({
                    "Patch Name": patch_names,
                    "x": pos_np[:, 0], "y": pos_np[:, 1],
                    "CTMIL Rollout Weight": cls_rollout,
                    "Patch Attention Score": patch_attn_scores,
                    "Patch Classification Score": patch_logits,
                    "Single Patch Logit": single_logits,
                    "Single Patch Probability": single_probs
                })
                df.to_excel(output_file, index=False)
                print(f"Saved rollout and attention scores for {wsi_id}")
        except Exception as e:
            print(f"Error processing sample {i+1} ({wsi_id}): {e}")


# 5. Temperature scaling (calibration)

def optimize_temperature(model, val_loader, device):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for feats, positions, label, _, _ in val_loader:
            feats, positions = feats.to(device), positions.to(device)
            logits = model(feats, positions)
            logits_list.append(logits if logits.dim() > 0 else logits.unsqueeze(0))
            labels_list.append(label.float())

    logits_all = torch.cat(logits_list)
    labels_all = torch.cat(labels_list).unsqueeze(1)

    temperature = torch.tensor(1.0, requires_grad=True, device=device)
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    def eval_loss():
        optimizer.zero_grad()
        loss = criterion(logits_all / temperature, labels_all)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return temperature.item()


# 6. ROC and PRC plotting

def compute_and_plot_roc(data_loader, device, output_file, T=1.0, name="Test"):
    preds, labels = [], []
    for feats, pos, label, _, _ in data_loader:
        with torch.no_grad():
            pred = torch.sigmoid(model(feats.to(device), pos.to(device)) / T).item()
        preds.append(pred)
        labels.append(label.item())
    fpr, tpr, _ = roc_curve(labels, preds)
    auc_score = roc_auc_score(labels, preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{name} AUC={auc_score:.4f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC")
    plt.legend()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved {name} ROC to {output_file}")

def compute_and_plot_prc(data_loader, device, output_file, T=1.0, name="Test"):
    preds, labels = [], []
    for feats, pos, label, _, _ in data_loader:
        with torch.no_grad():
            pred = torch.sigmoid(model(feats.to(device), pos.to(device)) / T).item()
        preds.append(pred)
        labels.append(label.item())
    precision, recall, _ = precision_recall_curve(labels, preds)
    ap = average_precision_score(labels, preds)
    plt.figure()
    plt.plot(recall, precision, label=f'{name} AUPRC={ap:.4f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{name} PRC")
    plt.legend()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved {name} PRC to {output_file}")


# 7. Save WSI-level predictions

def save_wsi_predictions(data_loader, dataset_obj, output_file, device, T=1.0):
    results = []
    for i, (feats, pos, label, extra, _) in enumerate(data_loader):
        wsi_id = str(extra[0]) if extra and extra[0] else os.path.basename(os.path.dirname(dataset_obj.files[i]))
        with torch.no_grad():
            logit = model(feats.to(device), pos.to(device))
        score = logit.item()
        prob = torch.sigmoid(logit / T).item()
        pred = int(prob >= 0.5)
        results.append({
            "WSI_ID": wsi_id,
            "True Label": label.item(),
            "Predicted Label": pred,
            "Prediction Probability": prob,
            "Classification Score": score
        })
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"Saved WSI predictions to {output_file}")


if __name__ == '__main__':
    # Dataset & model path setup
    index_root = "./attentionMIL_results/CTMIL_UNI_6layers"
    model_path = os.path.join(index_root, "best_UNI_CTMIL.pt")

    # Load datasets
    full_dataset = WSIFeatureDataset(feature_dir="./features_transMIL_UNI")
    train_indices = np.load(os.path.join(index_root, "train_indices.npy"))
    val_indices = np.load(os.path.join(index_root, "val_indices.npy"))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    training_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataset = WSIFeatureDataset(feature_dir="./XY_cohort/features_test_transMIL_UNI")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    RM_dataset = WSIFeatureDataset(feature_dir="./RM_cohort/features_RM_transMIL_UNI")
    RM_loader = DataLoader(RM_dataset, batch_size=1, shuffle=False)

    # Load model
    model = CTMIL(feature_dim=1024, hidden_dim=512, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded and set to eval mode.")

    # Temperature scaling
    optimal_T = optimize_temperature(model, val_loader, device)
    print(f"Optimal temperature: {optimal_T:.4f}")

    # Attention rollout output directories
    out_root = "./CTMIL_attentionscores_UNI"
    os.makedirs(out_root, exist_ok=True)
    save_attention_scores_rollout(training_loader, train_dataset, os.path.join(out_root, "training"), device)
    save_attention_scores_rollout(val_loader, val_dataset, os.path.join(out_root, "validation"), device)
    save_attention_scores_rollout(test_loader, test_dataset, os.path.join(out_root, "testing"), device)
    save_attention_scores_rollout(RM_loader, RM_dataset, os.path.join(out_root, "RM"), device)

    # ROC plots
    compute_and_plot_roc(training_loader, device, os.path.join(out_root, "train_roc.pdf"), optimal_T, "Train")
    compute_and_plot_roc(val_loader, device, os.path.join(out_root, "val_roc.pdf"), optimal_T, "Validation")
    compute_and_plot_roc(test_loader, device, os.path.join(out_root, "test_roc.pdf"), optimal_T, "Test")
    compute_and_plot_roc(RM_loader, device, os.path.join(out_root, "RM_roc.pdf"), optimal_T, "RM")

    # PRC plots
    compute_and_plot_prc(training_loader, device, os.path.join(out_root, "train_prc.pdf"), optimal_T, "Train")
    compute_and_plot_prc(val_loader, device, os.path.join(out_root, "val_prc.pdf"), optimal_T, "Validation")
    compute_and_plot_prc(test_loader, device, os.path.join(out_root, "test_prc.pdf"), optimal_T, "Test")
    compute_and_plot_prc(RM_loader, device, os.path.join(out_root, "RM_prc.pdf"), optimal_T, "RM")

    # WSI-level predictions
    save_wsi_predictions(training_loader, train_dataset, os.path.join(out_root, "wsi_predictions_train.xls"), device, optimal_T)
    save_wsi_predictions(val_loader, val_dataset, os.path.join(out_root, "wsi_predictions_val.xls"), device, optimal_T)
    save_wsi_predictions(test_loader, test_dataset, os.path.join(out_root, "wsi_predictions_test.xls"), device, optimal_T)
    save_wsi_predictions(RM_loader, RM_dataset, os.path.join(out_root, "wsi_predictions_RM.xls"), device, optimal_T)
