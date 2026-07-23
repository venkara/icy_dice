from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from .config import DieProfile, ModelSpec
from . import vision


@dataclass(frozen=True)
class ModelBundle:
    name: str
    path: Path
    model: nn.Module
    transform: transforms.Compose
    class_names: tuple[str, ...]
    crop_fraction: float
    image_size: int
    device: torch.device
    validation_accuracy: float | None


def center_crop_fraction(
    image: np.ndarray,
    fraction: float,
    output_size: int = vision.CROP_SIZE,
) -> np.ndarray:
    if not 0.25 <= fraction <= 1.0:
        raise ValueError(f"Invalid crop fraction: {fraction}")

    height, width = image.shape[:2]
    crop_width = max(2, int(round(width * fraction)))
    crop_height = max(2, int(round(height * fraction)))
    x1 = max(0, (width - crop_width) // 2)
    y1 = max(0, (height - crop_height) // 2)
    x2 = min(width, x1 + crop_width)
    y2 = min(height, y1 + crop_height)
    x1 = max(0, x2 - crop_width)
    y1 = max(0, y2 - crop_height)
    crop = image[y1:y2, x1:x2]

    return cv2.resize(
        crop,
        (output_size, output_size),
        interpolation=cv2.INTER_CUBIC,
    )


def _checkpoint_value(
    checkpoint: dict,
    primary: str,
    nested: str,
    fallback,
):
    if primary in checkpoint:
        return checkpoint[primary]
    return checkpoint.get("preprocess", {}).get(nested, fallback)


def load_model_bundle(
    spec: ModelSpec,
    expected_classes: tuple[str, ...],
    device: torch.device | None = None,
) -> ModelBundle:
    if not spec.path.exists():
        raise FileNotFoundError(f"Model not found: {spec.path}")

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    try:
        checkpoint = torch.load(
            spec.path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(spec.path, map_location=device)

    class_names = tuple(
        str(value)
        for value in checkpoint.get("class_names", expected_classes)
    )
    if class_names != expected_classes:
        raise RuntimeError(
            f"{spec.name} classes {class_names} do not match "
            f"the profile classes {expected_classes}."
        )

    crop_fraction = float(
        _checkpoint_value(
            checkpoint,
            "crop_fraction",
            "center_crop_fraction",
            spec.crop_fraction,
        )
    )
    image_size = int(checkpoint.get("image_size", 160))
    mean = tuple(
        float(value)
        for value in _checkpoint_value(
            checkpoint,
            "mean",
            "normalization_mean",
            (0.485, 0.456, 0.406),
        )
    )
    std = tuple(
        float(value)
        for value in _checkpoint_value(
            checkpoint,
            "std",
            "normalization_std",
            (0.229, 0.224, 0.225),
        )
    )

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(class_names),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return ModelBundle(
        name=spec.name,
        path=spec.path,
        model=model,
        transform=evaluation_transform,
        class_names=class_names,
        crop_fraction=crop_fraction,
        image_size=image_size,
        device=device,
        validation_accuracy=checkpoint.get("validation_accuracy"),
    )


class ModelEnsemble:
    def __init__(self, profile: DieProfile) -> None:
        if not profile.models:
            raise RuntimeError(
                f"No trained models are configured for {profile.die_type}."
            )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.profile = profile
        self.bundles = tuple(
            load_model_bundle(spec, profile.class_names, device)
            for spec in profile.models
        )
        self.device = device

    def classify(
        self,
        crops: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        return {
            bundle.name: self._classify_bundle(crops, bundle)
            for bundle in self.bundles
        }

    def _classify_bundle(
        self,
        crops: list[np.ndarray],
        bundle: ModelBundle,
    ) -> np.ndarray:
        if not crops:
            return np.empty(
                (0, len(bundle.class_names)),
                dtype=np.float32,
            )

        tensors: list[torch.Tensor] = []
        for crop in crops:
            model_crop = center_crop_fraction(
                crop,
                bundle.crop_fraction,
            )
            rgb = cv2.cvtColor(model_crop, cv2.COLOR_BGR2RGB)
            tensors.append(
                bundle.transform(Image.fromarray(rgb))
            )

        batch = torch.stack(tensors).to(bundle.device)
        with torch.inference_mode():
            probabilities = torch.softmax(
                bundle.model(batch),
                dim=1,
            )
        return probabilities.cpu().numpy()
