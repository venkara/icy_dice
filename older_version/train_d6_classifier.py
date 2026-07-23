from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

CLASS_NAMES = ("1", "2", "3", "4", "5", "6")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SESSION_RE = re.compile(r"^(\d{8}_\d{6}_\d{6})_\d+\.[^.]+$", re.I)
IMAGE_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    session: str


class DiceDataset(Dataset):
    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = self.transform(image.convert("RGB"))
        return image, sample.label, str(sample.path), sample.session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Icy Dice d6 classifier.")
    parser.add_argument("--data", type=Path, default=Path("dataset/d6"))
    parser.add_argument("--output", type=Path, default=Path("models/d6_baseline"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def session_from_name(path: Path) -> str:
    match = SESSION_RE.match(path.name)
    return match.group(1) if match else path.stem


def load_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for label, class_name in enumerate(CLASS_NAMES):
        folder = root / class_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(Sample(path, label, session_from_name(path)))
    if not samples:
        raise RuntimeError(f"No images found under {root}")
    return samples


def class_counts(samples: list[Sample]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for sample in samples:
        counts[CLASS_NAMES[sample.label]] += 1
    return counts


def all_classes_present(samples: list[Sample]) -> bool:
    return set(sample.label for sample in samples) == set(range(len(CLASS_NAMES)))


def grouped_split(samples: list[Sample], base_seed: int):
    labels = np.array([sample.label for sample in samples])
    groups = np.array([sample.session for sample in samples])
    indices = np.arange(len(samples))

    # Try several seeds because a purely grouped split can occasionally leave a
    # rare class out of validation or test when the dataset is still small.
    for offset in range(500):
        seed = base_seed + offset
        first = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
        train_idx, temp_idx = next(first.split(indices, labels, groups))

        second = GroupShuffleSplit(
            n_splits=1, train_size=0.50, random_state=seed + 10000
        )
        val_rel, test_rel = next(
            second.split(temp_idx, labels[temp_idx], groups[temp_idx])
        )
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]

        splits = {
            "train": [samples[i] for i in train_idx],
            "validation": [samples[i] for i in val_idx],
            "test": [samples[i] for i in test_idx],
        }
        if all(all_classes_present(split) for split in splits.values()):
            return splits, seed

    raise RuntimeError(
        "Could not create session-grouped train/validation/test splits containing "
        "all six classes. Collect more independent rolls or change --seed."
    )


def build_transforms():
    train = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(
                180,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=128,
            ),
            transforms.RandomAffine(
                0,
                translate=(0.06, 0.06),
                scale=(0.90, 1.10),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=128,
            ),
            transforms.ColorJitter(0.15, 0.15, 0.10, 0.03),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    evaluate = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train, evaluate


def build_model(pretrained: bool) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 6)
    return model


def make_loader(dataset, batch_size: int, shuffle: bool, cuda: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=cuda,
    )


def epoch_pass(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels, _paths, _sessions in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total += labels.size(0)
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((logits.argmax(1) == labels).sum().item())
    return total_loss / total, total_correct / total


def predictions(model, loader, device):
    model.eval()
    truth, predicted, confidence, paths, sessions = [], [], [], [], []
    with torch.inference_mode():
        for images, labels, batch_paths, batch_sessions in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            conf, pred = probs.max(1)
            truth.extend(labels.numpy().tolist())
            predicted.extend(pred.cpu().numpy().tolist())
            confidence.extend(conf.cpu().numpy().tolist())
            paths.extend(batch_paths)
            sessions.extend(batch_sessions)
    return (
        np.asarray(truth),
        np.asarray(predicted),
        np.asarray(confidence),
        paths,
        sessions,
    )


def save_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm)
    ax.set(
        xticks=np.arange(6),
        yticks=np.arange(6),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title="d6 test-set confusion matrix",
    )
    threshold = cm.max() / 2 if cm.size else 0
    for row in range(6):
        for col in range(6):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_curves(history: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [h["epoch"] for h in history],
        [h["train_accuracy"] for h in history],
        label="train",
    )
    ax.plot(
        [h["epoch"] for h in history],
        [h["validation_accuracy"] for h in history],
        label="validation",
    )
    ax.set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02), title="Learning curves")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_errors(truth, predicted, confidence, paths, output: Path) -> None:
    errors = [i for i in range(len(paths)) if truth[i] != predicted[i]]
    errors.sort(key=lambda i: float(confidence[i]), reverse=True)
    errors = errors[:60]
    if not errors:
        image = Image.new("RGB", (700, 160), "white")
        ImageDraw.Draw(image).text(
            (30, 65), "No test-set misclassifications.", fill="black"
        )
        image.save(output)
        return
    columns, tile_w, tile_h = 5, 180, 205
    rows = (len(errors) + columns - 1) // columns
    montage = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(montage)
    for position, index in enumerate(errors):
        row, col = divmod(position, columns)
        x, y = col * tile_w, row * tile_h
        with Image.open(paths[index]) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((150, 150))
        montage.paste(thumb, (x + (tile_w - thumb.width) // 2, y + 5))
        draw.text(
            (x + 8, y + 160),
            f"true {CLASS_NAMES[truth[index]]}  pred {CLASS_NAMES[predicted[index]]}",
            fill="black",
        )
        draw.text((x + 8, y + 180), f"confidence {confidence[index]:.3f}", fill="black")
    montage.save(output)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.data)
    splits, split_seed = grouped_split(samples, args.seed)

    print("Session-grouped dataset split:")
    split_summary = {}
    for name in ("train", "validation", "test"):
        split = splits[name]
        summary = {
            "images": len(split),
            "sessions": len({s.session for s in split}),
            "class_counts": class_counts(split),
        }
        split_summary[name] = summary
        print(f"  {name:10s}: {summary}")

    train_transform, eval_transform = build_transforms()
    datasets = {
        "train": DiceDataset(splits["train"], train_transform),
        "validation": DiceDataset(splits["validation"], eval_transform),
        "test": DiceDataset(splits["test"], eval_transform),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    loaders = {
        name: make_loader(ds, args.batch_size, name == "train", device.type == "cuda")
        for name, ds in datasets.items()
    }

    model = build_model(not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_path = args.output / "d6_mobilenet_v3_small.pt"
    best_val = -1.0
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = epoch_pass(
            model, loaders["train"], criterion, device, optimizer
        )
        val_loss, val_acc = epoch_pass(model, loaders["validation"], criterion, device)
        scheduler.step(val_acc)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:02d}: train loss {train_loss:.4f}, acc {train_acc:.3f}; "
            f"val loss {val_loss:.4f}, acc {val_acc:.3f}"
        )
        if val_acc > best_val:
            best_val = val_acc
            stale_epochs = 0
            torch.save(
                {
                    "model_name": "mobilenet_v3_small",
                    "state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "image_size": IMAGE_SIZE,
                    "mean": MEAN,
                    "std": STD,
                    "validation_accuracy": best_val,
                },
                best_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    truth, predicted, confidence, paths, sessions = predictions(
        model, loaders["test"], device
    )

    accuracy = float(np.mean(truth == predicted))
    cm = confusion_matrix(truth, predicted, labels=list(range(6)))
    report_text = classification_report(
        truth,
        predicted,
        labels=list(range(6)),
        target_names=CLASS_NAMES,
        digits=3,
        zero_division=0,
    )
    print(f"\nFinal test accuracy: {accuracy:.3f}\n")
    print(report_text)

    save_confusion(cm, args.output / "confusion_matrix.png")
    save_curves(history, args.output / "learning_curves.png")
    save_errors(
        truth,
        predicted,
        confidence,
        paths,
        args.output / "misclassified_test_images.png",
    )

    report = {
        "data": str(args.data.resolve()),
        "model": str(best_path.resolve()),
        "device": str(device),
        "split_seed": split_seed,
        "splits": split_summary,
        "best_validation_accuracy": best_val,
        "test_accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
        "history": history,
        "test_predictions": [
            {
                "path": path,
                "session": session,
                "true": CLASS_NAMES[int(t)],
                "predicted": CLASS_NAMES[int(p)],
                "confidence": float(c),
            }
            for path, session, t, p, c in zip(
                paths, sessions, truth, predicted, confidence, strict=True
            )
        ],
    }
    with (args.output / "training_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("Saved outputs in:", args.output.resolve())
    print("  d6_mobilenet_v3_small.pt")
    print("  confusion_matrix.png")
    print("  learning_curves.png")
    print("  misclassified_test_images.png")
    print("  training_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
