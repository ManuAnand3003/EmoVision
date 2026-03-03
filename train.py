"""
EmoVision — Custom Emotion Model Training
=========================================
Trains EfficientNet-B0 (fine-tuned) on FER2013 + optionally RAF-DB.
Exports to ONNX for fast CPU inference.

Usage:
  1. Download FER2013 from Kaggle:
     kaggle datasets download -d msambare/fer2013
     unzip fer2013.zip -d data/fer2013/

  2. (Optional) Download RAF-DB from http://www.whdeng.cn/RAF/model1.html
     Place in data/rafdb/

  3. Run training:
     python training/train.py --data data/fer2013 --epochs 40 --output models/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ── Emotion classes ────────────────────────────────────────────────────────
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
FER_IDX  = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}


# ── Dataset ────────────────────────────────────────────────────────────────
class FER2013Dataset(Dataset):
    """
    Loads FER2013 from the Kaggle CSV format.
    CSV columns: emotion (int 0-6), pixels (space-separated 48x48 values), Usage
    """

    def __init__(self, csv_path: str, usage: str = 'Training', transform=None):
        df = pd.read_csv(csv_path)
        self.data = df[df['Usage'] == usage].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row    = self.data.iloc[idx]
        label  = int(row['emotion'])
        pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
        img    = Image.fromarray(pixels, mode='L').convert('RGB')  # grayscale → RGB
        if self.transform:
            img = self.transform(img)
        return img, label


class RAFDBDataset(Dataset):
    """
    RAF-DB dataset loader.
    Expects: data/rafdb/basic/Image/aligned/  + data/rafdb/basic/EmoLabel/list_patition_label.txt
    RAF-DB label mapping: 1=surprise, 2=fear, 3=disgust, 4=happy, 5=sad, 6=angry, 7=neutral
    """
    RAF_TO_FER = {1: 5, 2: 2, 3: 1, 4: 3, 5: 4, 6: 0, 7: 6}  # remap to FER2013 indices

    def __init__(self, root: str, split: str = 'train', transform=None):
        self.root = Path(root)
        self.transform = transform
        label_file = self.root / 'basic' / 'EmoLabel' / 'list_patition_label.txt'
        self.samples = []
        with open(label_file) as f:
            for line in f:
                fname, label = line.strip().split()
                label = int(label)
                if split == 'train' and fname.startswith('train'):
                    self.samples.append((fname, self.RAF_TO_FER[label]))
                elif split == 'test' and fname.startswith('test'):
                    self.samples.append((fname, self.RAF_TO_FER[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = self.root / 'basic' / 'Image' / 'aligned' / fname
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Model ──────────────────────────────────────────────────────────────────
def build_model(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-B0 fine-tuned for emotion classification.
    - Replace classifier head with [Dropout → Linear → num_classes]
    - Unfreeze last 2 blocks for fine-tuning
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 2 feature blocks
    for block in list(model.features.children())[-3:]:
        for param in block.parameters():
            param.requires_grad = True

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


# ── Training utilities ─────────────────────────────────────────────────────
def get_class_weights(dataset) -> torch.Tensor:
    """Compute inverse-frequency weights to handle class imbalance."""
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=len(EMOTIONS)).astype(float)
    weights = 1.0 / (counts + 1e-6)
    return torch.tensor(weights / weights.sum(), dtype=torch.float32)


def get_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Create sampler that oversamples underrepresented classes."""
    labels = [dataset[i][1] for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=len(EMOTIONS)).astype(float)
    sample_weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


# ── Main training loop ─────────────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[EmoVision Trainer] Device: {device}")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Transforms ────────────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Datasets ──────────────────────────────────────────────────────────
    data_path = Path(args.data)
    csv_path  = data_path / 'fer2013.csv'

    train_ds = FER2013Dataset(csv_path, usage='Training',   transform=train_tf)
    val_ds   = FER2013Dataset(csv_path, usage='PublicTest', transform=val_tf)
    test_ds  = FER2013Dataset(csv_path, usage='PrivateTest', transform=val_tf)

    # Optionally merge RAF-DB training data
    if args.rafdb and Path(args.rafdb).exists():
        rafdb_train = RAFDBDataset(args.rafdb, split='train', transform=train_tf)
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset([train_ds, rafdb_train])
        print(f"[Data] Merged FER2013 + RAF-DB: {len(train_ds)} training samples")
    else:
        print(f"[Data] FER2013 only: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    sampler = get_weighted_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(num_classes=7).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] EfficientNet-B0 | Trainable params: {total_params:,}")

    # ── Loss (weighted for class imbalance) ───────────────────────────────
    class_weights = get_class_weights(FER2013Dataset(csv_path, usage='Training')).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc  = 0.0
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                val_loss    += loss.item() * imgs.size(0)
                preds        = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

    # ── Final test evaluation ──────────────────────────────────────────────
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n[Final Test Accuracy: {test_acc:.4f}]")
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS))

    # ── Save confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                cmap='Blues', linewidths=0.5)
    plt.title('EmoVision — Confusion Matrix')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    print(f"[Saved] confusion_matrix.png")

    # ── Export to ONNX for fast inference ─────────────────────────────────
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    onnx_path   = output_dir / 'emotion_model.onnx'
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'], output_names=['logits'],
        dynamic_axes={'input': {0: 'batch_size'}},
        opset_version=13,
    )
    print(f"[Saved] ONNX model: {onnx_path}")

    # ── Save training history ──────────────────────────────────────────────
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump({**history, 'test_accuracy': test_acc, 'best_val_accuracy': best_val_acc}, f, indent=2)

    print(f"\n✅ Training complete. Best val acc: {best_val_acc:.4f} | Test acc: {test_acc:.4f}")
    print(f"   Models saved to: {output_dir}/")


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EmoVision Model Trainer')
    parser.add_argument('--data',       default='data/fer2013',  help='Path to FER2013 CSV directory')
    parser.add_argument('--rafdb',      default=None,            help='Path to RAF-DB root (optional)')
    parser.add_argument('--output',     default='models',        help='Output directory for model files')
    parser.add_argument('--epochs',     type=int, default=40,    help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,    help='Batch size')
    parser.add_argument('--lr',         type=float, default=3e-4, help='Initial learning rate')
    args = parser.parse_args()
    train(args)
