# -*- coding: utf-8 -*-
"""

@author: HBZH
"""

import os
import argparse
import torch
import timm
from PIL import Image
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login



# ============ 1. Build pretrained UNI feature extractor ============
def build_uni_extractor(device):
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True
    )
    model.head = torch.nn.Identity()  # Remove classification head
    model.eval().to(device)
    return model


# ============ 2. Define image transforms ============
def build_transform(model):
    return transforms.Compose([
        transforms.Resize((512, 512)),
        create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    ])


# ============ 3. Feature extraction for a single WSI ============
@torch.no_grad()
def extract_features_from_patches(patch_paths, model, transform, device, chunk_size=16):
    all_features, all_positions = [], []

    for i in range(0, len(patch_paths), chunk_size):
        chunk = patch_paths[i: i + chunk_size]
        images, positions = [], []

        for path in chunk:
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"Failed to read {path}: {e}")
                continue

            images.append(transform(img))
            # Parse position from filename: "row_col_xxx.tif"
            filename = os.path.basename(path)
            try:
                row, col = map(int, filename.split("_")[:2])
                positions.append((row, col))
            except:
                print(f"Filename {filename} does not contain position info. Skipped.")
                continue

        if not images:
            continue

        batch = torch.stack(images).to(device)
        features = model(batch)
        all_features.append(features.cpu())
        all_positions.append(torch.tensor(positions, dtype=torch.float32))

    if not all_features:
        return None, None

    return torch.cat(all_features), torch.cat(all_positions)


# ============ 4. Process an entire WSI root folder ============
def process_dataset(root_dir, output_dir, model, transform, device, chunk_size=16):
    os.makedirs(output_dir, exist_ok=True)

    for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
        class_path = os.path.join(root_dir, class_dir)
        if not os.path.isdir(class_path):
            continue

        for wsi_folder in sorted(os.listdir(class_path)):
            wsi_path = os.path.join(class_path, wsi_folder)
            if not os.path.isdir(wsi_path):
                continue

            patch_paths = sorted([
                os.path.join(wsi_path, f) for f in os.listdir(wsi_path)
                if f.endswith(".tif")
            ])

            if not patch_paths:
                print(f"[Skip] No .tif files in {wsi_path}")
                continue

            features, positions = extract_features_from_patches(
                patch_paths, model, transform, device, chunk_size
            )

            if features is None:
                print(f"[Skip] No valid patches in {wsi_path}")
                continue

            save_path = os.path.join(output_dir, f"{label}_{wsi_folder}.pt")
            torch.save({
                "features": features,
                "positions": positions,
                "label": label,
                "wsi_id": wsi_folder
            }, save_path)

            print(f"[Saved] {save_path} | Features: {features.shape}, Positions: {positions.shape}")


# ============ 5. Main function ============
def main():
    parser = argparse.ArgumentParser(description="Extract WSI patch features using pretrained UNI.")
    parser.add_argument("--root_train", default="./training", help="Path to training WSI root")
    parser.add_argument("--root_val", default="./validation", help="Path to validation WSI root")
    parser.add_argument("--out_train", default="./features_train_transMIL", help="Output dir for training features")
    parser.add_argument("--out_val", default="./features_val_transMIL", help="Output dir for validation features")
    parser.add_argument("--chunk_size", type=int, default=16, help="Batch size for feature extraction")
    args = parser.parse_args()

    # Secure Hugging Face login
    login()  #token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = build_uni_extractor(device)
    transform = build_transform(model)

    # Process training set
    process_dataset(
        root_dir=args.root_train,
        output_dir=args.out_train,
        model=model,
        transform=transform,
        device=device,
        chunk_size=args.chunk_size
    )

    # Process validation set
    process_dataset(
        root_dir=args.root_val,
        output_dir=args.out_val,
        model=model,
        transform=transform,
        device=device,
        chunk_size=args.chunk_size
    )

    print("All feature extraction completed.")


if __name__ == "__main__":
    main()
