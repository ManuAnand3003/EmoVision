"""
EmoVision — Local Fine-Tuning Script
=====================================
Fine-tune DeepFace's emotion model on your own images to improve accuracy
on Indian faces (or any custom demographic/dataset).

HOW TO USE:
-----------
1. Collect ~50-200 images per emotion and organize them like this:

   my_dataset/
     angry/      ← images of angry faces
     disgust/
     fear/
     happy/
     sad/
     surprise/
     neutral/

   Tips for collecting Indian face data:
   - Use your own photos / photos of friends/family (with consent)
   - Download from: https://www.kaggle.com/datasets/shuvoalok/raf-db
   - Use: https://github.com/microsoft/FaceSynthetics (synthetic diverse faces)
   - AffectNet has some Indian faces (research license required)
   - Scrape ethically from public Flickr/Unsplash with diverse face filters

2. Run fine-tuning:
   python finetune_local.py --data my_dataset --output models/ --epochs 20

3. Done! The script saves a fine-tuned model to models/finetuned_model.pth
   and a fast ONNX version to models/finetuned_model.onnx

REQUIREMENTS (add to venv):
   pip install torch torchvision pillow scikit-learn matplotlib seaborn tqdm
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Emotion classes (must match DeepFace order) ────────────────────────────
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


# ── Dataset ────────────────────────────────────────────────────────────────
class LocalFaceDataset(Dataset):
    """
    Loads images from a folder structure:
        root/
          emotion_name/
            image1.jpg
            image2.png
            ...

    Supports: jpg, jpeg, png, bmp, webp
    """
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, root: str, transform=None, split='train', val_split=0.15):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.class_counts = {}

        all_samples = []
        for emotion in EMOTIONS:
            folder = self.root / emotion
            if not folder.exists():
                print(f"  [warn] Missing folder: {folder} — skipping")
                continue
            imgs = [p for p in folder.iterdir() if p.suffix.lower() in self.EXTENSIONS]
            self.class_counts[emotion] = len(imgs)
            label = EMOTIONS.index(emotion)
            all_samples.extend([(str(p), label) for p in imgs])

        # Reproducible train/val split
        np.random.seed(42)
        idx = np.random.permutation(len(all_samples))
        split_pt = int(len(idx) * (1 - val_split))
        chosen = idx[:split_pt] if split == 'train' else idx[split_pt:]
        self.samples = [all_samples[i] for i in chosen]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, label

    def summary(self):
        print("\n  Dataset composition:")
        for emo, count in self.class_counts.items():
            bar = '█' * min(count // 5, 40)
            print(f"    {emo:10s} {count:4d}  {bar}")
        print(f"    Total: {sum(self.class_counts.values())} images")


# ── Model ──────────────────────────────────────────────────────────────────
def build_model(checkpoint: str = None) -> nn.Module:
    """
    EfficientNet-B0 base. Optionally load from a previous checkpoint.
    Fine-tuning strategy:
      - If starting fresh: freeze early layers, train only classifier + last 3 blocks
      - If loading checkpoint: unfreeze all layers for full fine-tuning
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace head first so checkpoints from this script align cleanly.
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, len(EMOTIONS)),
    )

    if checkpoint and Path(checkpoint).exists():
        print(f"  [+] Loading base checkpoint: {checkpoint}")
        state = torch.load(checkpoint, map_location='cpu')
        try:
            model.load_state_dict(state, strict=False)
            print("  [+] Checkpoint loaded successfully")
        except Exception as e:
            print(f"  [warn] Partial load: {e}")

        # For small adaptation sets, keep most backbone frozen to avoid overfitting/instability.
        for param in model.parameters():
            param.requires_grad = False
        for block in list(model.features.children())[-3:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        # Freeze early feature extraction layers
        for i, block in enumerate(model.features.children()):
            for param in block.parameters():
                param.requires_grad = i >= 5  # unfreeze last 3 blocks

    return model


# ── Weighted sampler for imbalanced data ───────────────────────────────────
def make_sampler(dataset: Dataset) -> WeightedRandomSampler:
    labels = [dataset.samples[i][1] for i in range(len(dataset))]
    counts = np.bincount(labels, minlength=len(EMOTIONS)).astype(float)
    weights = [1.0 / (counts[l] + 1e-6) for l in labels]
    return WeightedRandomSampler(weights, len(weights))


# ── Training ───────────────────────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[EmoVision Fine-Tuner]")
    print(f"  Device  : {device}")
    print(f"  Data    : {args.data}")
    print(f"  Output  : {args.output}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  LR      : {args.lr}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Transforms ────────────────────────────────────────────────────────
    # Stronger augmentation for small datasets
    train_tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), shear=8),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = LocalFaceDataset(args.data, transform=train_tf, split='train')
    val_ds   = LocalFaceDataset(args.data, transform=val_tf,   split='val')

    if len(train_ds) == 0:
        print("\n[ERROR] No images found. Check your folder structure:")
        print("  my_dataset/")
        print("    happy/  image1.jpg  image2.jpg  ...")
        print("    angry/  ...")
        return

    train_ds.summary()
    print(f"\n  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    sampler = make_sampler(train_ds)
    workers = min(4, args.workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(checkpoint=args.checkpoint).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    # ── Loss ──────────────────────────────────────────────────────────────
    labels_all = [train_ds.samples[i][1] for i in range(len(train_ds))]
    counts = np.bincount(labels_all, minlength=len(EMOTIONS)).astype(float)
    cw = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    cw = cw / cw.sum() * len(EMOTIONS)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=max(args.lr * 0.1, 1e-7),
    )

    # ── Loop ──────────────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  {'Val Loss':>9}  {'Val Acc':>9}")
    print("─" * 56)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        tl = tc = tt = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * imgs.size(0)
            tc += (outputs.argmax(1) == lbls).sum().item()
            tt += imgs.size(0)

        # Val
        model.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                vl += criterion(out, lbls).item() * imgs.size(0)
                vc += (out.argmax(1) == lbls).sum().item()
                vt += imgs.size(0)

        ta = tc/tt if tt>0 else 0
        va = vc/vt if vt>0 else 0
        history['train_loss'].append(tl/tt if tt>0 else 0)
        history['val_loss'].append(vl/vt if vt>0 else 0)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)

        marker = ' ✓ best' if va > best_val_acc else ''
        print(f"{epoch:>6}  {tl/tt:>11.4f}  {ta:>10.4f}  {vl/vt:>9.4f}  {va:>9.4f}{marker}")

        scheduler.step()

        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), out_dir / 'finetuned_model.pth')

    # ── Eval on val set ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(out_dir / 'finetuned_model.pth', map_location='cpu'))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
            trues.extend(lbls.numpy())

    print(f"\n  Best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")

    if len(set(trues)) > 1:
        labels = list(range(len(EMOTIONS)))
        print("\n" + classification_report(
            trues,
            preds,
            labels=labels,
            target_names=EMOTIONS,
            zero_division=0,
        ))

        # Confusion matrix
        cm = confusion_matrix(trues, preds, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=EMOTIONS, yticklabels=EMOTIONS,
                    cmap='Blues', linewidths=0.5)
        plt.title('EmoVision Fine-Tune — Confusion Matrix')
        plt.tight_layout()
        plt.savefig(out_dir / 'finetune_confusion.png', dpi=150)
        print(f"  [Saved] finetune_confusion.png")

    # ── Training curves ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train'); ax1.plot(history['val_loss'], label='Val')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history['train_acc'], label='Train'); ax2.plot(history['val_acc'], label='Val')
    ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'finetune_curves.png', dpi=150)
    print(f"  [Saved] finetune_curves.png")

    # ── Export ONNX ────────────────────────────────────────────────────────
    onnx_saved = False
    try:
        dummy = torch.randn(1, 3, 112, 112).to(device)
        onnx_path = out_dir / 'finetuned_model.onnx'
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=['input'], output_names=['logits'],
            dynamic_axes={'input': {0: 'batch_size'}},
            opset_version=18,
        )
        onnx_saved = True
        print(f"  [Saved] finetuned_model.onnx")
    except Exception as e:
        print(f"  [warn] ONNX export failed: {e}")

    # ── Save history ───────────────────────────────────────────────────────
    with open(out_dir / 'finetune_history.json', 'w') as f:
        json.dump({**history, 'best_val_accuracy': best_val_acc, 'onnx_saved': onnx_saved}, f, indent=2)

    print(f"\n✅ Fine-tuning complete!")
    print(f"   Model: {out_dir}/finetuned_model.pth")
    print(f"   ONNX:  {out_dir}/finetuned_model.onnx")
    print(f"\nNext step: Load the fine-tuned model in pipeline.py")
    print(f"  See FINETUNING.md for integration instructions.")


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='EmoVision Local Fine-Tuner')
    p.add_argument('--data',       required=True,          help='Path to dataset folder (emotion subfolders)')
    p.add_argument('--output',     default='models',       help='Output directory')
    p.add_argument('--checkpoint', default=None,           help='Optional: path to existing .pth checkpoint to continue from')
    p.add_argument('--epochs',     type=int,   default=20, help='Training epochs (20-40 recommended)')
    p.add_argument('--batch',      type=int,   default=16, help='Batch size (reduce to 8 if GPU OOM)')
    p.add_argument('--lr',         type=float, default=1e-4, help='Learning rate')
    p.add_argument('--workers',    type=int,   default=2,  help='DataLoader workers')
    args = p.parse_args()
    train(args)
