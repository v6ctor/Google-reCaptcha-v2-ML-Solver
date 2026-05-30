"""
Train a MobileNetV3-Small tile classifier.

Usage:
    python train.py
    python train.py --data ../data-collection/images --epochs 30 --batch 64
    python train.py --arch efficientnet_b0
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V3_Small_Weights, EfficientNet_B0_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(augment: bool):
    if augment:
        return transforms.Compose([
            # Spatial
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.RandomRotation(20),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.3),
            # Color / appearance
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
            transforms.RandomGrayscale(p=0.08),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            # Erase a random patch (simulates partial occlusion)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.0)),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_model(arch: str, num_classes: int):
    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    total_loss = correct = total = 0
    # per-class tracking
    class_correct = torch.zeros(num_classes)
    class_total   = torch.zeros(num_classes)
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        total_loss += criterion(out, labels).item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        for c in range(num_classes):
            mask = labels == c
            class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            class_total[c]   += mask.sum().item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    per_class_acc = (class_correct / class_total.clamp(min=1)).tolist()
    return total_loss / total, correct / total, per_class_acc, all_preds, all_labels


def print_header():
    print(f"\n{'Epoch':>5} {'Train Loss':>10} {'Train Acc':>9} {'Val Loss':>8} {'Val Acc':>7} {'LR':>8} {'Time':>6}")
    print("─" * 62)


def print_epoch(epoch, epochs, train_loss, train_acc, val_loss, val_acc, lr, elapsed, improved):
    star = " ★" if improved else ""
    print(f"{epoch:>5}/{epochs:<4} {train_loss:>10.4f} {train_acc:>8.2%} "
          f"{val_loss:>8.4f} {val_acc:>7.2%} {lr:>8.2e} {elapsed:>5.1f}s{star}")


def print_per_class(classes, per_class_acc, class_totals):
    print("\n  Per-class val accuracy:")
    print(f"  {'Class':<15} {'Acc':>6}  {'Samples':>7}")
    print("  " + "─" * 32)
    for i, cls in enumerate(classes):
        acc = per_class_acc[i]
        n   = int(class_totals[i])
        bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
        print(f"  {cls:<15} {acc:>5.1%}  {n:>7}  {bar}")


def confusion_matrix_text(classes, all_preds, all_labels):
    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    for p, l in zip(all_preds, all_labels):
        cm[l][p] += 1

    col_w = max(len(c) for c in classes) + 1
    header = " " * col_w + "".join(f"{c[:6]:>7}" for c in classes)
    lines  = ["\n  Confusion matrix (rows=actual, cols=predicted):", "  " + header]
    for i, row in enumerate(cm):
        total = max(sum(row), 1)
        cells = "".join(f"{v:>7}" for v in row)
        lines.append(f"  {classes[i]:<{col_w}}{cells}   ({cm[i][i]/total:.0%} correct)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      type=Path,  default=Path("../data-collection/images"))
    parser.add_argument("--out",       type=Path,  default=Path("model.pt"))
    parser.add_argument("--arch",      default="mobilenet_v3_small",
                        choices=["mobilenet_v3_small", "efficientnet_b0"])
    parser.add_argument("--epochs",    type=int,   default=25)
    parser.add_argument("--batch",     type=int,   default=64)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--workers",   type=int,   default=4)
    parser.add_argument("--per-class-every", type=int, default=5,
                        help="Print per-class accuracy every N epochs (default 5)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = datasets.ImageFolder(args.data, transform=build_transforms(augment=True))
    classes      = full_dataset.classes
    num_classes  = len(classes)

    # Dataset summary
    class_counts = [0] * num_classes
    for _, label in full_dataset.samples:
        class_counts[label] += 1
    total_images = sum(class_counts)

    print("═" * 62)
    print(f"  Device:      {device}")
    print(f"  Arch:        {args.arch}")
    print(f"  Classes:     {num_classes}")
    print(f"  Total imgs:  {total_images:,}")
    print(f"  Train/Val:   {int(total_images*(1-args.val_split)):,} / {int(total_images*args.val_split):,}")
    print(f"  Batch:       {args.batch}  |  LR: {args.lr}  |  Epochs: {args.epochs}")
    print("─" * 62)
    print(f"  {'Class':<15} {'Images':>8}")
    for i, cls in enumerate(classes):
        print(f"  {cls:<15} {class_counts[i]:>8,}")
    print("═" * 62)

    map_path = args.out.parent / "class_map.json"
    map_path.write_text(json.dumps({i: n for i, n in enumerate(classes)}, indent=2))
    print(f"  Class map → {map_path}\n")

    n_val   = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform = build_transforms(augment=False)

    # Track per-class totals in val set
    val_class_totals = torch.zeros(num_classes)
    for idx in val_ds.indices:
        val_class_totals[full_dataset.samples[idx][1]] += 1

    # ── class-balanced sampler ────────────────────────────────────────────────
    # Count how many training samples each class has, then give each sample a
    # weight inversely proportional to its class frequency so minority classes
    # are oversampled to match the majority class.
    train_labels = [full_dataset.samples[i][1] for i in train_ds.indices]
    train_counts = torch.zeros(num_classes)
    for lbl in train_labels:
        train_counts[lbl] += 1
    class_weights   = 1.0 / train_counts.clamp(min=1)
    sample_weights  = torch.tensor([class_weights[lbl] for lbl in train_labels])
    target_n        = int(train_counts.max().item()) * num_classes  # one full pass per class
    sampler         = WeightedRandomSampler(sample_weights, num_samples=target_n, replacement=True)

    max_cls   = classes[train_counts.argmax()]
    min_cls   = classes[train_counts.argmin()]
    print(f"  Sampler: {target_n:,} draws/epoch  "
          f"(majority={max_cls} {int(train_counts.max())}, "
          f"minority={min_cls} {int(train_counts.min())})")

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model     = build_model(args.arch, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val_acc = 0.0
    print_header()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, per_class_acc, all_preds, all_labels = evaluate(
            model, val_loader, criterion, device, num_classes)
        scheduler.step()

        lr      = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        improved = val_acc > best_val_acc

        print_epoch(epoch, args.epochs, train_loss, train_acc,
                    val_loss, val_acc, lr, elapsed, improved)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if improved:
            best_val_acc = val_acc
            torch.save({"arch": args.arch, "classes": classes,
                        "state_dict": model.state_dict()}, args.out)

        if epoch % args.per_class_every == 0 or epoch == args.epochs:
            print_per_class(classes, per_class_acc, val_class_totals)

    # Final confusion matrix
    _, _, _, all_preds, all_labels = evaluate(
        model, val_loader, criterion, device, num_classes)
    print(confusion_matrix_text(classes, all_preds, all_labels))

    # Save training history
    hist_path = args.out.parent / "history.json"
    hist_path.write_text(json.dumps(history, indent=2))

    print(f"\n{'═'*62}")
    print(f"  Best val accuracy: {best_val_acc:.2%}")
    print(f"  Model   → {args.out}")
    print(f"  History → {hist_path}")
    print(f"{'═'*62}")


if __name__ == "__main__":
    main()
