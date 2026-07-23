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


DIE_TYPE = "d8"
CLASS_NAMES = tuple(str(value) for value in range(1, 9))
CLASS_COUNT = len(CLASS_NAMES)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Collector filenames:
#     YYYYMMDD_HHMMSS_microseconds_XX.png
# All crops from the same physical roll therefore share one group identifier.
SESSION_RE = re.compile(
    r"^(\d{8}_\d{6}_\d{6})_\d+\.[^.]+$",
    re.IGNORECASE,
)

IMAGE_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

CONFIDENCE_THRESHOLDS = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    session: str


class CenterFractionCrop:
    """
    Retain the centered fraction of an image.

    A value of 1.0 leaves the image unchanged. Because collector crops are
    normally square, this removes the same percentage from every edge. The
    implementation also handles non-square source images safely.
    """

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
    def __init__(
        self,
        samples: list[Sample],
        transform,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        with Image.open(sample.path) as image:
            image = self.transform(image.convert("RGB"))

        return (
            image,
            sample.label,
            str(sample.path),
            sample.session,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Icy Dice d8 classifier."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/d8"),
        help="Folder containing class directories 1 through 8.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/d8_center"),
        help="Directory for the checkpoint and diagnostic files.",
    )
    parser.add_argument(
        "--crop-fraction",
        type=float,
        default=0.72,
        help=(
            "Centered fraction retained before resizing. "
            "Use 1.0 for the complete candidate image."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260723,
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize MobileNetV3-Small without ImageNet weights.",
    )

    args = parser.parse_args()

    if not 0.25 <= args.crop_fraction <= 1.0:
        parser.error("--crop-fraction must be between 0.25 and 1.0")

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    if args.patience < 1:
        parser.error("--patience must be at least 1")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

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


def load_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []

    for label, class_name in enumerate(CLASS_NAMES):
        folder = root / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Missing class folder: {folder}"
            )

        for path in sorted(folder.iterdir()):
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_SUFFIXES
            ):
                samples.append(
                    Sample(
                        path=path,
                        label=label,
                        session=session_from_name(path),
                    )
                )

    if not samples:
        raise RuntimeError(
            f"No images found under {root}"
        )

    return samples


def class_counts(
    samples: list[Sample],
) -> dict[str, int]:
    counts = {
        name: 0
        for name in CLASS_NAMES
    }

    for sample in samples:
        counts[CLASS_NAMES[sample.label]] += 1

    return counts


def session_count(samples: list[Sample]) -> int:
    return len(
        {
            sample.session
            for sample in samples
        }
    )


def all_classes_present(
    samples: list[Sample],
) -> bool:
    return {
        sample.label
        for sample in samples
    } == set(range(CLASS_COUNT))


def grouped_split(
    samples: list[Sample],
    base_seed: int,
):
    """
    Create a 70/15/15 split without placing crops from one physical roll into
    more than one partition.
    """
    labels = np.asarray(
        [sample.label for sample in samples],
        dtype=np.int64,
    )
    groups = np.asarray(
        [sample.session for sample in samples]
    )
    indices = np.arange(len(samples))

    if len(np.unique(groups)) < 3:
        raise RuntimeError(
            "At least three independent capture sessions are required."
        )

    # GroupShuffleSplit is not class-stratified. Try several deterministic
    # seeds until every split contains all eight classes.
    for offset in range(1000):
        seed = base_seed + offset

        first = GroupShuffleSplit(
            n_splits=1,
            train_size=0.70,
            random_state=seed,
        )
        train_idx, temporary_idx = next(
            first.split(
                indices,
                labels,
                groups,
            )
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

        validation_idx = temporary_idx[
            validation_relative
        ]
        test_idx = temporary_idx[
            test_relative
        ]

        splits = {
            "train": [
                samples[index]
                for index in train_idx
            ],
            "validation": [
                samples[index]
                for index in validation_idx
            ],
            "test": [
                samples[index]
                for index in test_idx
            ],
        }

        if all(
            all_classes_present(split)
            for split in splits.values()
        ):
            return splits, seed

    raise RuntimeError(
        "Could not create session-grouped train/validation/test "
        "splits containing all eight classes. Collect more independent "
        "rolls, especially for classes represented in few sessions, or "
        "change --seed."
    )


def build_transforms(
    crop_fraction: float,
):
    center_crop = CenterFractionCrop(
        crop_fraction
    )

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
            transforms.Normalize(
                MEAN,
                STD,
            ),
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
            transforms.Normalize(
                MEAN,
                STD,
            ),
        ]
    )

    return (
        train_transform,
        evaluation_transform,
    )


def build_model(
    pretrained: bool,
) -> nn.Module:
    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.mobilenet_v3_small(
        weights=weights
    )
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        CLASS_COUNT,
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
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
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
        for (
            images,
            labels,
            _paths,
            _sessions,
        ) in loader:
            images = images.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(images)
            loss = criterion(
                logits,
                labels,
            )

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total += batch_size
            total_loss += (
                float(loss.item())
                * batch_size
            )
            total_correct += int(
                (
                    logits.argmax(dim=1)
                    == labels
                ).sum().item()
            )

    return (
        total_loss / total,
        total_correct / total,
    )


def predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()

    truth: list[int] = []
    predicted: list[int] = []
    confidence: list[float] = []
    paths: list[str] = []
    sessions: list[str] = []

    with torch.inference_mode():
        for (
            images,
            labels,
            batch_paths,
            batch_sessions,
        ) in loader:
            probabilities = torch.softmax(
                model(
                    images.to(
                        device,
                        non_blocking=True,
                    )
                ),
                dim=1,
            )
            batch_confidence, batch_prediction = (
                probabilities.max(dim=1)
            )

            truth.extend(
                labels.numpy().tolist()
            )
            predicted.extend(
                batch_prediction.cpu().numpy().tolist()
            )
            confidence.extend(
                batch_confidence.cpu().numpy().tolist()
            )
            paths.extend(batch_paths)
            sessions.extend(batch_sessions)

    return (
        np.asarray(
            truth,
            dtype=np.int64,
        ),
        np.asarray(
            predicted,
            dtype=np.int64,
        ),
        np.asarray(
            confidence,
            dtype=np.float32,
        ),
        paths,
        sessions,
    )


def confidence_analysis(
    truth: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
) -> list[dict[str, float | int]]:
    rows: list[
        dict[str, float | int]
    ] = []
    total = len(truth)

    for threshold in CONFIDENCE_THRESHOLDS:
        accepted = confidence >= threshold
        accepted_count = int(
            accepted.sum()
        )

        if accepted_count:
            accepted_accuracy = float(
                np.mean(
                    truth[accepted]
                    == predicted[accepted]
                )
            )
        else:
            accepted_accuracy = 0.0

        rows.append(
            {
                "threshold": threshold,
                "accepted": accepted_count,
                "rejected": total - accepted_count,
                "coverage": (
                    accepted_count / total
                    if total
                    else 0.0
                ),
                "accepted_accuracy": accepted_accuracy,
            }
        )

    return rows


def save_confusion(
    matrix: np.ndarray,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        figsize=(8.5, 7.5)
    )
    image = axes.imshow(matrix)

    axes.set(
        xticks=np.arange(CLASS_COUNT),
        yticks=np.arange(CLASS_COUNT),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title="d8 test-set confusion matrix",
    )

    threshold = (
        matrix.max() / 2
        if matrix.size
        else 0
    )

    for row in range(CLASS_COUNT):
        for column in range(CLASS_COUNT):
            axes.text(
                column,
                row,
                str(matrix[row, column]),
                horizontalalignment="center",
                verticalalignment="center",
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                ),
            )

    figure.colorbar(
        image,
        ax=axes,
    )
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=180,
    )
    plt.close(figure)


def save_curves(
    history: list[dict[str, float | int]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        figsize=(8, 5)
    )

    axes.plot(
        [
            row["epoch"]
            for row in history
        ],
        [
            row["train_accuracy"]
            for row in history
        ],
        label="train",
    )
    axes.plot(
        [
            row["epoch"]
            for row in history
        ],
        [
            row["validation_accuracy"]
            for row in history
        ],
        label="validation",
    )

    axes.set(
        xlabel="Epoch",
        ylabel="Accuracy",
        ylim=(0, 1.02),
        title="d8 learning curves",
    )
    axes.grid(
        True,
        alpha=0.25,
    )
    axes.legend()

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=180,
    )
    plt.close(figure)


def center_crop_for_display(
    image: Image.Image,
    crop_fraction: float,
) -> Image.Image:
    return CenterFractionCrop(
        crop_fraction
    )(
        image
    )


def save_errors(
    truth: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
    paths: list[str],
    output: Path,
    crop_fraction: float,
) -> None:
    errors = [
        index
        for index in range(len(paths))
        if truth[index] != predicted[index]
    ]
    errors.sort(
        key=lambda index: float(
            confidence[index]
        ),
        reverse=True,
    )
    errors = errors[:64]

    if not errors:
        image = Image.new(
            "RGB",
            (720, 170),
            "white",
        )
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
    rows = (
        len(errors)
        + columns
        - 1
    ) // columns

    montage = Image.new(
        "RGB",
        (
            columns * tile_width,
            rows * tile_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(montage)

    for position, index in enumerate(errors):
        row, column = divmod(
            position,
            columns,
        )
        x = column * tile_width
        y = row * tile_height

        with Image.open(paths[index]) as source:
            thumbnail = center_crop_for_display(
                source.convert("RGB"),
                crop_fraction,
            )
            thumbnail = ImageOps.contain(
                thumbnail,
                (155, 155),
            )

        montage.paste(
            thumbnail,
            (
                x
                + (
                    tile_width
                    - thumbnail.width
                ) // 2,
                y + 5,
            ),
        )
        draw.text(
            (x + 8, y + 162),
            (
                f"true {CLASS_NAMES[truth[index]]}  "
                f"pred {CLASS_NAMES[predicted[index]]}"
            ),
            fill="black",
        )
        draw.text(
            (x + 8, y + 183),
            (
                f"confidence "
                f"{confidence[index]:.3f}"
            ),
            fill="black",
        )

    montage.save(output)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = load_samples(
        args.data
    )
    splits, split_seed = grouped_split(
        samples,
        args.seed,
    )

    print(
        f"Loaded {len(samples)} d8 images "
        f"from {session_count(samples)} sessions."
    )
    print(
        f"Center crop fraction: "
        f"{args.crop_fraction:.2f}"
    )
    print(
        "\nSession-grouped dataset split:"
    )

    split_summary: dict[
        str,
        dict[str, object],
    ] = {}

    for name in (
        "train",
        "validation",
        "test",
    ):
        split = splits[name]
        summary = {
            "images": len(split),
            "sessions": session_count(split),
            "class_counts": class_counts(split),
        }
        split_summary[name] = summary
        print(
            f"  {name:10s}: {summary}"
        )

    (
        training_transform,
        evaluation_transform,
    ) = build_transforms(
        args.crop_fraction
    )

    datasets = {
        "train": DiceDataset(
            splits["train"],
            training_transform,
        ),
        "validation": DiceDataset(
            splits["validation"],
            evaluation_transform,
        ),
        "test": DiceDataset(
            splits["test"],
            evaluation_transform,
        ),
    }

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")

    loaders = {
        name: make_loader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            cuda=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }

    model = build_model(
        pretrained=not args.no_pretrained
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )

    best_path = (
        args.output
        / "d8_mobilenet_v3_small.pt"
    )
    best_validation_accuracy = -1.0
    stale_epochs = 0
    history: list[
        dict[str, float | int]
    ] = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        (
            training_loss,
            training_accuracy,
        ) = epoch_pass(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
        )
        (
            validation_loss,
            validation_accuracy,
        ) = epoch_pass(
            model,
            loaders["validation"],
            criterion,
            device,
        )

        scheduler.step(
            validation_accuracy
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": training_loss,
            "train_accuracy": training_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}: "
            f"train loss {training_loss:.4f}, "
            f"acc {training_accuracy:.3f}; "
            f"val loss {validation_loss:.4f}, "
            f"acc {validation_accuracy:.3f}"
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_accuracy
            )
            stale_epochs = 0

            torch.save(
                {
                    "die_type": DIE_TYPE,
                    "model_name": "mobilenet_v3_small",
                    "state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "image_size": IMAGE_SIZE,
                    "mean": MEAN,
                    "std": STD,
                    "crop_fraction": args.crop_fraction,
                    "preprocess": {
                        "center_crop_fraction": args.crop_fraction,
                        "resize": [
                            IMAGE_SIZE,
                            IMAGE_SIZE,
                        ],
                        "normalization_mean": list(MEAN),
                        "normalization_std": list(STD),
                    },
                    "validation_accuracy": (
                        best_validation_accuracy
                    ),
                    "split_seed": split_seed,
                },
                best_path,
            )
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print(
                f"Early stopping after "
                f"{epoch} epochs."
            )
            break

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(
        checkpoint["state_dict"]
    )

    (
        truth,
        predicted,
        confidence,
        paths,
        sessions,
    ) = predictions(
        model,
        loaders["test"],
        device,
    )

    test_accuracy = float(
        np.mean(
            truth == predicted
        )
    )

    matrix = confusion_matrix(
        truth,
        predicted,
        labels=list(
            range(CLASS_COUNT)
        ),
    )
    report_text = classification_report(
        truth,
        predicted,
        labels=list(
            range(CLASS_COUNT)
        ),
        target_names=CLASS_NAMES,
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
    print(
        f"Final test accuracy: "
        f"{test_accuracy:.3f}\n"
    )
    print(report_text)

    print("Confidence threshold analysis:")
    print(
        " threshold  accepted/total  coverage  "
        "accuracy among accepted"
    )
    for row in threshold_rows:
        print(
            f"   {row['threshold']:.2f}"
            f"       {row['accepted']:3d}/{len(truth):3d}"
            f"       {row['coverage']:.3f}"
            f"          "
            f"{row['accepted_accuracy']:.3f}"
        )

    save_confusion(
        matrix,
        args.output
        / "confusion_matrix.png",
    )
    save_curves(
        history,
        args.output
        / "learning_curves.png",
    )
    save_errors(
        truth,
        predicted,
        confidence,
        paths,
        args.output
        / "misclassified_test_images.png",
        args.crop_fraction,
    )

    report = {
        "die_type": DIE_TYPE,
        "data": str(
            args.data.resolve()
        ),
        "model": str(
            best_path.resolve()
        ),
        "device": str(device),
        "crop_fraction": args.crop_fraction,
        "image_size": IMAGE_SIZE,
        "split_seed": split_seed,
        "splits": split_summary,
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
        "test_accuracy": test_accuracy,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_text,
        "confidence_analysis": threshold_rows,
        "history": history,
        "test_predictions": [
            {
                "path": path,
                "session": session,
                "true": CLASS_NAMES[int(true_label)],
                "predicted": CLASS_NAMES[
                    int(predicted_label)
                ],
                "confidence": float(
                    sample_confidence
                ),
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

    with (
        args.output
        / "training_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print(
        "\nSaved outputs in:",
        args.output.resolve(),
    )
    print(
        "  d8_mobilenet_v3_small.pt"
    )
    print(
        "  confusion_matrix.png"
    )
    print(
        "  learning_curves.png"
    )
    print(
        "  misclassified_test_images.png"
    )
    print(
        "  training_report.json"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
