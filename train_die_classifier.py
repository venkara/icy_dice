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
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

from icy_dice.config import get_profile


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SESSION_RE = re.compile(
    r"^(\d{8}_\d{6}_\d{6})_\d+\.[^.]+$",
    re.IGNORECASE,
)
IMAGE_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CONFIDENCE_THRESHOLDS = tuple(
    value / 100
    for value in range(50, 100, 5)
)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    session: str


class CenterFractionCrop:
    def __init__(self, fraction: float) -> None:
        if not 0.25 <= fraction <= 1.0:
            raise ValueError(
                "crop fraction must be between 0.25 and 1.0"
            )
        self.fraction = float(fraction)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.fraction >= 0.9999:
            return image
        width, height = image.size
        crop_width = max(1, int(round(width * self.fraction)))
        crop_height = max(1, int(round(height * self.fraction)))
        left = max(0, (width - crop_width) // 2)
        top = max(0, (height - crop_height) // 2)
        return image.crop(
            (
                left,
                top,
                left + crop_width,
                top + crop_height,
            )
        )


class DiceDataset(Dataset):
    def __init__(self, samples: list[Sample], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, sample.label, str(sample.path), sample.session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a configured Icy Dice classifier."
    )
    parser.add_argument(
        "--die-type",
        required=True,
        help="Configured die type such as d6, d8, or d10.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Defaults to dataset/<die-type>.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Defaults to models/<die-type>_centerNN where NN is "
            "the crop percentage."
        ),
    )
    parser.add_argument(
        "--crop-fraction",
        type=float,
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    profile = get_profile(args.die_type)
    args.die_type = profile.die_type
    args.class_names = profile.class_names

    if args.data is None:
        args.data = Path("dataset") / profile.die_type
    if args.output is None:
        percent = int(round(args.crop_fraction * 100))
        args.output = Path("models") / (
            f"{profile.die_type}_center{percent}"
        )

    if not 0.25 <= args.crop_fraction <= 1.0:
        parser.error("--crop-fraction must be between 0.25 and 1.0")
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")
    if args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("batch size and learning rate must be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def session_from_name(path: Path) -> str:
    match = SESSION_RE.match(path.name)
    return match.group(1) if match else path.stem


def load_samples(
    root: Path,
    class_names: tuple[str, ...],
) -> list[Sample]:
    samples: list[Sample] = []
    for label, class_name in enumerate(class_names):
        folder = root / class_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(
                    Sample(
                        path=path,
                        label=label,
                        session=session_from_name(path),
                    )
                )
    if not samples:
        raise RuntimeError(f"No images found under {root}")
    return samples


def class_counts(
    samples: list[Sample],
    class_names: tuple[str, ...],
) -> dict[str, int]:
    counts = {name: 0 for name in class_names}
    for sample in samples:
        counts[class_names[sample.label]] += 1
    return counts


def session_count(samples: list[Sample]) -> int:
    return len({sample.session for sample in samples})


def all_classes_present(
    samples: list[Sample],
    class_count: int,
) -> bool:
    return {
        sample.label for sample in samples
    } == set(range(class_count))


def grouped_split(
    samples: list[Sample],
    class_count: int,
    base_seed: int,
):
    labels = np.asarray(
        [sample.label for sample in samples],
        dtype=np.int64,
    )
    groups = np.asarray([sample.session for sample in samples])
    indices = np.arange(len(samples))

    if len(np.unique(groups)) < 3:
        raise RuntimeError(
            "At least three independent capture sessions are required."
        )

    for offset in range(1000):
        seed = base_seed + offset
        first = GroupShuffleSplit(
            n_splits=1,
            train_size=0.70,
            random_state=seed,
        )
        train_idx, temporary_idx = next(
            first.split(indices, labels, groups)
        )
        temporary_groups = groups[temporary_idx]
        if len(np.unique(temporary_groups)) < 2:
            continue

        second = GroupShuffleSplit(
            n_splits=1,
            train_size=0.50,
            random_state=seed + 10000,
        )
        validation_relative, test_relative = next(
            second.split(
                temporary_idx,
                labels[temporary_idx],
                temporary_groups,
            )
        )
        validation_idx = temporary_idx[validation_relative]
        test_idx = temporary_idx[test_relative]

        splits = {
            "train": [samples[index] for index in train_idx],
            "validation": [
                samples[index] for index in validation_idx
            ],
            "test": [samples[index] for index in test_idx],
        }
        if all(
            all_classes_present(split, class_count)
            for split in splits.values()
        ):
            return splits, seed

    raise RuntimeError(
        "Could not create session-grouped splits containing every class. "
        "Collect more independent sessions for sparse classes."
    )


def build_transforms(crop_fraction: float):
    center_crop = CenterFractionCrop(crop_fraction)
    train_transform = transforms.Compose(
        [
            center_crop,
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomRotation(
                180,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=128,
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.06, 0.06),
                scale=(0.90, 1.10),
                shear=(-4.0, 4.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=128,
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            center_crop,
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_transform, evaluation_transform


def build_model(
    class_count: int,
    pretrained: bool,
) -> nn.Module:
    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
        if pretrained
        else None
    )
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        class_count,
    )
    return model


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    cuda: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=cuda,
    )


def epoch_pass(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total = 0
    context = (
        torch.enable_grad()
        if training
        else torch.inference_mode()
    )

    with context:
        for images, labels, _paths, _sessions in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total += batch_size
            total_loss += float(loss.item()) * batch_size
            total_correct += int(
                (logits.argmax(dim=1) == labels).sum().item()
            )

    return total_loss / total, total_correct / total


def predictions(model, loader, device):
    model.eval()
    truth, predicted, confidence = [], [], []
    paths, sessions = [], []

    with torch.inference_mode():
        for images, labels, batch_paths, batch_sessions in loader:
            probabilities = torch.softmax(
                model(images.to(device, non_blocking=True)),
                dim=1,
            )
            batch_confidence, batch_prediction = probabilities.max(dim=1)
            truth.extend(labels.numpy().tolist())
            predicted.extend(batch_prediction.cpu().numpy().tolist())
            confidence.extend(batch_confidence.cpu().numpy().tolist())
            paths.extend(batch_paths)
            sessions.extend(batch_sessions)

    return (
        np.asarray(truth, dtype=np.int64),
        np.asarray(predicted, dtype=np.int64),
        np.asarray(confidence, dtype=np.float32),
        paths,
        sessions,
    )


def confidence_analysis(truth, predicted, confidence):
    rows = []
    total = len(truth)
    for threshold in CONFIDENCE_THRESHOLDS:
        accepted = confidence >= threshold
        accepted_count = int(accepted.sum())
        accuracy = (
            float(np.mean(truth[accepted] == predicted[accepted]))
            if accepted_count
            else 0.0
        )
        rows.append(
            {
                "threshold": threshold,
                "accepted": accepted_count,
                "rejected": total - accepted_count,
                "coverage": accepted_count / total if total else 0.0,
                "accepted_accuracy": accuracy,
            }
        )
    return rows


def save_confusion(matrix, class_names, die_type, path):
    figure, axes = plt.subplots(
        figsize=(
            max(7, len(class_names) * 0.85),
            max(6, len(class_names) * 0.75),
        )
    )
    image = axes.imshow(matrix)
    axes.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=f"{die_type} test-set confusion matrix",
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            axes.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                ),
            )
    figure.colorbar(image, ax=axes)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_curves(history, die_type, path):
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(
        [row["epoch"] for row in history],
        [row["train_accuracy"] for row in history],
        label="train",
    )
    axes.plot(
        [row["epoch"] for row in history],
        [row["validation_accuracy"] for row in history],
        label="validation",
    )
    axes.set(
        xlabel="Epoch",
        ylabel="Accuracy",
        ylim=(0, 1.02),
        title=f"{die_type} learning curves",
    )
    axes.grid(True, alpha=0.25)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_errors(
    truth,
    predicted,
    confidence,
    paths,
    class_names,
    crop_fraction,
    output,
):
    errors = [
        index
        for index in range(len(paths))
        if truth[index] != predicted[index]
    ]
    errors.sort(
        key=lambda index: float(confidence[index]),
        reverse=True,
    )
    errors = errors[:64]

    if not errors:
        image = Image.new("RGB", (720, 170), "white")
        ImageDraw.Draw(image).text(
            (30, 70),
            "No test-set misclassifications.",
            fill="black",
        )
        image.save(output)
        return

    columns = 5
    tile_width = 185
    tile_height = 210
    rows = (len(errors) + columns - 1) // columns
    montage = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        "white",
    )
    draw = ImageDraw.Draw(montage)
    center_crop = CenterFractionCrop(crop_fraction)

    for position, index in enumerate(errors):
        row, column = divmod(position, columns)
        x, y = column * tile_width, row * tile_height
        with Image.open(paths[index]) as source:
            thumbnail = ImageOps.contain(
                center_crop(source.convert("RGB")),
                (155, 155),
            )
        montage.paste(
            thumbnail,
            (
                x + (tile_width - thumbnail.width) // 2,
                y + 5,
            ),
        )
        draw.text(
            (x + 8, y + 162),
            (
                f"true {class_names[truth[index]]}  "
                f"pred {class_names[predicted[index]]}"
            ),
            fill="black",
        )
        draw.text(
            (x + 8, y + 183),
            f"confidence {confidence[index]:.3f}",
            fill="black",
        )
    montage.save(output)


def main() -> int:
    args = parse_args()
    class_names = tuple(args.class_names)
    class_count = len(class_names)
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.data, class_names)
    splits, split_seed = grouped_split(
        samples,
        class_count,
        args.seed,
    )
    print(
        f"Loaded {len(samples)} {args.die_type} images from "
        f"{session_count(samples)} sessions."
    )

    split_summary = {}
    for name in ("train", "validation", "test"):
        split = splits[name]
        summary = {
            "images": len(split),
            "sessions": session_count(split),
            "class_counts": class_counts(split, class_names),
        }
        split_summary[name] = summary
        print(f"  {name:10s}: {summary}")

    training_transform, evaluation_transform = build_transforms(
        args.crop_fraction
    )
    datasets = {
        "train": DiceDataset(splits["train"], training_transform),
        "validation": DiceDataset(
            splits["validation"],
            evaluation_transform,
        ),
        "test": DiceDataset(splits["test"], evaluation_transform),
    }

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    loaders = {
        name: make_loader(
            dataset,
            args.batch_size,
            name == "train",
            device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }

    model = build_model(
        class_count,
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    model_path = args.output / (
        f"{args.die_type}_mobilenet_v3_small.pt"
    )
    best_validation_accuracy = -1.0
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = epoch_pass(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
        )
        validation_loss, validation_accuracy = epoch_pass(
            model,
            loaders["validation"],
            criterion,
            device,
        )
        scheduler.step(validation_accuracy)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}: train loss {train_loss:.4f}, "
            f"acc {train_accuracy:.3f}; val loss "
            f"{validation_loss:.4f}, acc {validation_accuracy:.3f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            stale_epochs = 0
            torch.save(
                {
                    "die_type": args.die_type,
                    "model_name": "mobilenet_v3_small",
                    "state_dict": model.state_dict(),
                    "class_names": list(class_names),
                    "image_size": IMAGE_SIZE,
                    "mean": MEAN,
                    "std": STD,
                    "crop_fraction": args.crop_fraction,
                    "preprocess": {
                        "center_crop_fraction": args.crop_fraction,
                        "resize": [IMAGE_SIZE, IMAGE_SIZE],
                        "normalization_mean": list(MEAN),
                        "normalization_std": list(STD),
                    },
                    "validation_accuracy": best_validation_accuracy,
                    "split_seed": split_seed,
                },
                model_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    truth, predicted, confidence, paths, sessions = predictions(
        model,
        loaders["test"],
        device,
    )
    test_accuracy = float(np.mean(truth == predicted))
    matrix = confusion_matrix(
        truth,
        predicted,
        labels=list(range(class_count)),
    )
    report_text = classification_report(
        truth,
        predicted,
        labels=list(range(class_count)),
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
    threshold_rows = confidence_analysis(
        truth,
        predicted,
        confidence,
    )

    print(
        f"\nBest validation accuracy: "
        f"{best_validation_accuracy:.3f}"
    )
    print(f"Final test accuracy: {test_accuracy:.3f}\n")
    print(report_text)

    save_confusion(
        matrix,
        class_names,
        args.die_type,
        args.output / "confusion_matrix.png",
    )
    save_curves(
        history,
        args.die_type,
        args.output / "learning_curves.png",
    )
    save_errors(
        truth,
        predicted,
        confidence,
        paths,
        class_names,
        args.crop_fraction,
        args.output / "misclassified_test_images.png",
    )

    report = {
        "die_type": args.die_type,
        "class_names": list(class_names),
        "data": str(args.data.resolve()),
        "model": str(model_path.resolve()),
        "crop_fraction": args.crop_fraction,
        "image_size": IMAGE_SIZE,
        "split_seed": split_seed,
        "splits": split_summary,
        "best_validation_accuracy": best_validation_accuracy,
        "test_accuracy": test_accuracy,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_text,
        "confidence_analysis": threshold_rows,
        "history": history,
        "test_predictions": [
            {
                "path": path,
                "session": session,
                "true": class_names[int(true_label)],
                "predicted": class_names[int(predicted_label)],
                "confidence": float(sample_confidence),
            }
            for (
                path,
                session,
                true_label,
                predicted_label,
                sample_confidence,
            ) in zip(
                paths,
                sessions,
                truth,
                predicted,
                confidence,
                strict=True,
            )
        ],
    }
    (args.output / "training_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print("\nSaved outputs in:", args.output.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
