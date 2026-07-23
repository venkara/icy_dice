from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

try:
    import icy_dice_dataset_collector_v4_2 as collector
except ImportError as error:
    raise ImportError(
        "icy_dice_d8_reader.py must be in the same directory as "
        "icy_dice_dataset_collector_v4_2.py."
    ) from error


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_55_PATH = Path(
    "models/d8_center55/d8_mobilenet_v3_small.pt"
)
DEFAULT_MODEL_60_PATH = Path(
    "models/d8_center60/d8_mobilenet_v3_small.pt"
)

DEFAULT_MODEL_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_FRAME_VOTE_THRESHOLD = 0.75

# A selected burst frame is discarded if candidate centroids cannot be matched
# to the representative frame within this many rectified-tray pixels.
MAX_CANDIDATE_MATCH_DISTANCE = 70.0
MIN_AGGREGATION_FRAMES = 8

# Keep the result detail window away from the main window. These are easy to
# change for a different monitor arrangement.
RESULT_WINDOW_X = 425
RESULT_WINDOW_Y = 1150

DEBUG_DIRECTORY = Path("captures/d8_reader")
CLASS_NAMES = tuple(str(value) for value in range(1, 9))


# ---------------------------------------------------------------------------
# Model and result types
# ---------------------------------------------------------------------------

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


@dataclass(frozen=True)
class ModelAggregate:
    value: int
    confidence: float
    vote_fraction: float
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class D8Prediction:
    value: int
    accepted: bool
    combined_confidence: float
    model_55: ModelAggregate
    model_60: ModelAggregate
    display_crop: np.ndarray


@dataclass(frozen=True)
class AggregationResult:
    predictions: list[D8Prediction]
    representative_tray: np.ndarray
    representative_mask: np.ndarray
    representative_candidates: list[collector.Candidate]
    representative_component_labels: np.ndarray
    representative_split_count: int
    frames_used: int
    selected_frames: int


# ---------------------------------------------------------------------------
# Arguments and count handling
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read physical d8 rolls using burst aggregation and the "
            "55%/60% center-crop model pair."
        )
    )
    parser.add_argument(
        "--model55",
        type=Path,
        default=DEFAULT_MODEL_55_PATH,
        help="Path to the trained 55%% center-crop d8 model.",
    )
    parser.add_argument(
        "--model60",
        type=Path,
        default=DEFAULT_MODEL_60_PATH,
        help="Path to the trained 60%% center-crop d8 model.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Initial number of d8 dice. Prompts when omitted.",
    )
    parser.add_argument(
        "--model-threshold",
        type=float,
        default=DEFAULT_MODEL_CONFIDENCE_THRESHOLD,
        help=(
            "Minimum burst-averaged confidence required from each model."
        ),
    )
    parser.add_argument(
        "--vote-threshold",
        type=float,
        default=DEFAULT_FRAME_VOTE_THRESHOLD,
        help=(
            "Minimum fraction of selected frames that must vote for each "
            "model's final class."
        ),
    )

    args = parser.parse_args()

    for name in ("model_threshold", "vote_threshold"):
        value = float(getattr(args, name))

        if not 0.0 < value <= 1.0:
            parser.error(
                f"--{name.replace('_', '-')} must be greater than 0 "
                "and at most 1."
            )

    return args


def parse_d8_count(text: str) -> int:
    normalized = text.strip().lower().replace(" ", "")

    if normalized.endswith("d8"):
        count_text = normalized[:-2]
        count = int(count_text) if count_text else 1
    else:
        count = int(normalized)

    if not 1 <= count <= 30:
        raise ValueError(
            "The d8 count must be between 1 and 30."
        )

    return count


def prompt_for_d8_count(
    current: int | None = None,
) -> int:
    while True:
        if current is None:
            response = input(
                "How many d8 dice? Examples: 4 or 4d8: "
            ).strip()
        else:
            response = input(
                f"New d8 count [{current}] "
                "(Enter keeps current): "
            ).strip()

            if not response:
                return current

        try:
            return parse_d8_count(response)
        except (TypeError, ValueError):
            print(
                "Enter a count such as 4 or an expression "
                "such as 4d8."
            )


def make_d8_request(
    count: int,
) -> collector.RollRequest:
    return collector.RollRequest(
        count=count,
        die_type="d8",
    )


# ---------------------------------------------------------------------------
# Model loading and preprocessing
# ---------------------------------------------------------------------------

def load_model_bundle(
    name: str,
    path: Path,
) -> ModelBundle:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} model not found: {path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    class_names = tuple(
        str(value)
        for value in checkpoint.get(
            "class_names",
            CLASS_NAMES,
        )
    )

    if class_names != CLASS_NAMES:
        raise RuntimeError(
            f"{name} is not an eight-class d8 model. "
            f"Classes found: {class_names}"
        )

    checkpoint_die_type = checkpoint.get(
        "die_type"
    )

    if (
        checkpoint_die_type is not None
        and checkpoint_die_type != "d8"
    ):
        raise RuntimeError(
            f"{name} identifies itself as "
            f"{checkpoint_die_type!r}, not 'd8'."
        )

    crop_fraction = float(
        checkpoint.get(
            "crop_fraction",
            checkpoint.get(
                "preprocess",
                {},
            ).get(
                "center_crop_fraction",
                1.0,
            ),
        )
    )
    image_size = int(
        checkpoint.get(
            "image_size",
            160,
        )
    )
    mean = tuple(
        float(value)
        for value in checkpoint.get(
            "mean",
            checkpoint.get(
                "normalization_mean",
                (0.485, 0.456, 0.406),
            ),
        )
    )
    std = tuple(
        float(value)
        for value in checkpoint.get(
            "std",
            checkpoint.get(
                "normalization_std",
                (0.229, 0.224, 0.225),
            ),
        )
    )

    model = models.mobilenet_v3_small(
        weights=None
    )
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(class_names),
    )
    model.load_state_dict(
        checkpoint["state_dict"]
    )
    model.to(device)
    model.eval()

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=(
                    transforms.InterpolationMode.BILINEAR
                ),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean,
                std,
            ),
        ]
    )

    return ModelBundle(
        name=name,
        path=path,
        model=model,
        transform=evaluation_transform,
        class_names=class_names,
        crop_fraction=crop_fraction,
        image_size=image_size,
        device=device,
        validation_accuracy=checkpoint.get(
            "validation_accuracy"
        ),
    )


def center_crop_fraction(
    image: np.ndarray,
    fraction: float,
) -> np.ndarray:
    if not 0.25 <= fraction <= 1.0:
        raise ValueError(
            f"Invalid crop fraction: {fraction}"
        )

    if fraction >= 0.9999:
        return image.copy()

    height, width = image.shape[:2]
    crop_width = max(
        2,
        int(round(width * fraction)),
    )
    crop_height = max(
        2,
        int(round(height * fraction)),
    )

    x1 = max(
        0,
        (width - crop_width) // 2,
    )
    y1 = max(
        0,
        (height - crop_height) // 2,
    )
    x2 = min(
        width,
        x1 + crop_width,
    )
    y2 = min(
        height,
        y1 + crop_height,
    )

    x1 = max(
        0,
        x2 - crop_width,
    )
    y1 = max(
        0,
        y2 - crop_height,
    )

    crop = image[y1:y2, x1:x2]

    return cv2.resize(
        crop,
        (
            collector.CROP_SIZE,
            collector.CROP_SIZE,
        ),
        interpolation=cv2.INTER_CUBIC,
    )


def classify_crops(
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
        rgb = cv2.cvtColor(
            model_crop,
            cv2.COLOR_BGR2RGB,
        )
        tensors.append(
            bundle.transform(
                Image.fromarray(rgb)
            )
        )

    batch = torch.stack(
        tensors
    ).to(bundle.device)

    with torch.inference_mode():
        logits = bundle.model(batch)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    return probabilities.cpu().numpy()


# ---------------------------------------------------------------------------
# Candidate matching and burst aggregation
# ---------------------------------------------------------------------------

def candidate_match_order(
    reference: list[collector.Candidate],
    current: list[collector.Candidate],
) -> list[int] | None:
    """
    Match stationary candidates to the representative frame by centroid.

    The detector's reading-order sort is usually already sufficient. Explicit
    matching prevents a small vertical jitter from swapping two nearby dice.
    """
    if len(reference) != len(current):
        return None

    count = len(reference)

    if count == 0:
        return []

    pairs: list[
        tuple[float, int, int]
    ] = []

    for reference_index, reference_candidate in enumerate(
        reference
    ):
        reference_point = np.asarray(
            reference_candidate.centroid,
            dtype=np.float32,
        )

        for current_index, current_candidate in enumerate(
            current
        ):
            current_point = np.asarray(
                current_candidate.centroid,
                dtype=np.float32,
            )
            distance = float(
                np.linalg.norm(
                    reference_point
                    - current_point
                )
            )
            pairs.append(
                (
                    distance,
                    reference_index,
                    current_index,
                )
            )

    pairs.sort(
        key=lambda item: item[0]
    )

    assigned_reference: set[int] = set()
    assigned_current: set[int] = set()
    mapping = [-1] * count
    maximum_distance = 0.0

    for (
        distance,
        reference_index,
        current_index,
    ) in pairs:
        if reference_index in assigned_reference:
            continue

        if current_index in assigned_current:
            continue

        mapping[reference_index] = current_index
        assigned_reference.add(
            reference_index
        )
        assigned_current.add(
            current_index
        )
        maximum_distance = max(
            maximum_distance,
            distance,
        )

        if len(assigned_reference) == count:
            break

    if (
        any(index < 0 for index in mapping)
        or maximum_distance
        > MAX_CANDIDATE_MATCH_DISTANCE
    ):
        return None

    return mapping


def weighted_probability_average(
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    probabilities shape: frames x dice x classes
    """
    normalized_weights = weights / weights.sum()

    return np.tensordot(
        normalized_weights,
        probabilities,
        axes=(0, 0),
    )


def model_aggregates(
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> list[ModelAggregate]:
    averaged = weighted_probability_average(
        probabilities,
        weights,
    )
    frame_votes = np.argmax(
        probabilities,
        axis=2,
    )

    aggregates: list[ModelAggregate] = []

    for die_index in range(
        averaged.shape[0]
    ):
        class_index = int(
            np.argmax(
                averaged[die_index]
            )
        )
        confidence = float(
            averaged[
                die_index,
                class_index,
            ]
        )
        vote_fraction = float(
            np.mean(
                frame_votes[:, die_index]
                == class_index
            )
        )

        aggregates.append(
            ModelAggregate(
                value=int(
                    CLASS_NAMES[
                        class_index
                    ]
                ),
                confidence=confidence,
                vote_fraction=vote_fraction,
                probabilities=tuple(
                    float(value)
                    for value in averaged[
                        die_index
                    ]
                ),
            )
        )

    return aggregates


def aggregate_selected_burst(
    selection: collector.BurstSelection,
    background: np.ndarray,
    request: collector.RollRequest,
    bundle_55: ModelBundle,
    bundle_60: ModelBundle,
    model_threshold: float,
    vote_threshold: float,
) -> AggregationResult:
    representative = selection.representative
    representative_mask = collector.foreground_mask(
        representative.rectified,
        background,
    )
    (
        reference_candidates,
        reference_labels,
        representative_split_count,
    ) = collector.detect_die_candidates(
        representative_mask,
        request.count,
    )

    if len(reference_candidates) != request.count:
        raise RuntimeError(
            "The representative frame no longer contains "
            f"{request.count} detectable dice."
        )

    frame_crops: list[
        list[np.ndarray]
    ] = []
    frame_weights: list[float] = []

    for frame in selection.selected_frames:
        mask = collector.foreground_mask(
            frame.rectified,
            background,
        )
        (
            candidates,
            component_labels,
            _split_count,
        ) = collector.detect_die_candidates(
            mask,
            request.count,
        )

        mapping = candidate_match_order(
            reference_candidates,
            candidates,
        )

        if mapping is None:
            continue

        ordered_crops: list[np.ndarray] = []

        for current_index in mapping:
            candidate = candidates[
                current_index
            ]
            images = collector.extract_candidate_images(
                frame.rectified,
                component_labels,
                candidate,
            )
            ordered_crops.append(
                images.masked
            )

        frame_crops.append(
            ordered_crops
        )
        frame_weights.append(
            max(
                0.05,
                float(
                    frame.quality_score
                ),
            )
        )

    if len(frame_crops) < MIN_AGGREGATION_FRAMES:
        raise RuntimeError(
            "Too few burst frames could be matched consistently: "
            f"{len(frame_crops)} available, "
            f"{MIN_AGGREGATION_FRAMES} required."
        )

    frame_count = len(frame_crops)
    die_count = request.count
    flat_crops = [
        crop
        for crops in frame_crops
        for crop in crops
    ]

    probabilities_55 = classify_crops(
        flat_crops,
        bundle_55,
    ).reshape(
        frame_count,
        die_count,
        len(CLASS_NAMES),
    )
    probabilities_60 = classify_crops(
        flat_crops,
        bundle_60,
    ).reshape(
        frame_count,
        die_count,
        len(CLASS_NAMES),
    )

    weights = np.asarray(
        frame_weights,
        dtype=np.float64,
    )
    aggregates_55 = model_aggregates(
        probabilities_55,
        weights,
    )
    aggregates_60 = model_aggregates(
        probabilities_60,
        weights,
    )

    representative_images = [
        collector.extract_candidate_images(
            representative.rectified,
            reference_labels,
            candidate,
        )
        for candidate in reference_candidates
    ]

    predictions: list[D8Prediction] = []

    for (
        aggregate_55,
        aggregate_60,
        representative_images_for_die,
    ) in zip(
        aggregates_55,
        aggregates_60,
        representative_images,
        strict=True,
    ):
        agreement = (
            aggregate_55.value
            == aggregate_60.value
        )
        accepted = (
            agreement
            and aggregate_55.confidence
            >= model_threshold
            and aggregate_60.confidence
            >= model_threshold
            and aggregate_55.vote_fraction
            >= vote_threshold
            and aggregate_60.vote_fraction
            >= vote_threshold
        )

        combined_probabilities = (
            np.asarray(
                aggregate_55.probabilities,
                dtype=np.float32,
            )
            + np.asarray(
                aggregate_60.probabilities,
                dtype=np.float32,
            )
        ) / 2.0
        combined_index = int(
            np.argmax(
                combined_probabilities
            )
        )

        if agreement:
            value = aggregate_55.value
            combined_confidence = float(
                combined_probabilities[
                    value - 1
                ]
            )
        else:
            value = int(
                CLASS_NAMES[
                    combined_index
                ]
            )
            combined_confidence = float(
                combined_probabilities[
                    combined_index
                ]
            )

        predictions.append(
            D8Prediction(
                value=value,
                accepted=accepted,
                combined_confidence=(
                    combined_confidence
                ),
                model_55=aggregate_55,
                model_60=aggregate_60,
                display_crop=center_crop_fraction(
                    representative_images_for_die.masked,
                    bundle_55.crop_fraction,
                ),
            )
        )

    return AggregationResult(
        predictions=predictions,
        representative_tray=(
            representative.rectified.copy()
        ),
        representative_mask=(
            representative_mask
        ),
        representative_candidates=(
            reference_candidates
        ),
        representative_component_labels=(
            reference_labels
        ),
        representative_split_count=(
            representative_split_count
        ),
        frames_used=frame_count,
        selected_frames=len(
            selection.selected_frames
        ),
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def annotate_predictions(
    result: AggregationResult,
) -> np.ndarray:
    output = (
        result.representative_tray.copy()
    )

    all_accepted = all(
        prediction.accepted
        for prediction in result.predictions
    )
    total = sum(
        prediction.value
        for prediction in result.predictions
    )

    for index, (
        candidate,
        prediction,
    ) in enumerate(
        zip(
            result.representative_candidates,
            result.predictions,
            strict=True,
        ),
        start=1,
    ):
        x, y, width, height = (
            candidate.bbox
        )
        color = (
            (0, 255, 0)
            if prediction.accepted
            else (0, 0, 255)
        )

        cv2.rectangle(
            output,
            (x, y),
            (
                x + width,
                y + height,
            ),
            color,
            3,
        )

        label = (
            f"#{index}: {prediction.value} "
            f"{prediction.combined_confidence:.0%}"
        )
        text_y = max(
            24,
            y - 10,
        )

        cv2.putText(
            output,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    values_text = " + ".join(
        str(prediction.value)
        for prediction in result.predictions
    )
    split_text = (
        f" | auto-split "
        f"{result.representative_split_count}"
        if result.representative_split_count
        else ""
    )

    if all_accepted:
        status = (
            f"ACCEPTED | {values_text} = {total}"
            f" | {result.frames_used} burst frames"
            f"{split_text}"
        )
        status_color = (0, 255, 0)
    else:
        uncertain_count = sum(
            not prediction.accepted
            for prediction in result.predictions
        )
        status = (
            f"UNCERTAIN ({uncertain_count}) | "
            f"provisional {values_text} = {total}"
            f" | {result.frames_used} burst frames"
            f"{split_text}"
        )
        status_color = (0, 0, 255)

    return collector.add_status_banner(
        output,
        status,
        status_color,
        height=64,
    )


def build_prediction_sheet(
    result: AggregationResult,
) -> np.ndarray:
    tile_width = 235
    tile_height = 235
    columns = min(
        5,
        max(
            1,
            len(result.predictions),
        ),
    )
    rows = math.ceil(
        len(result.predictions)
        / columns
    )

    sheet = np.full(
        (
            rows * tile_height,
            columns * tile_width,
            3,
        ),
        45,
        dtype=np.uint8,
    )

    for index, prediction in enumerate(
        result.predictions,
        start=1,
    ):
        row = (
            index - 1
        ) // columns
        column = (
            index - 1
        ) % columns

        tile_x = column * tile_width
        tile_y = row * tile_height
        image_x = tile_x + 53
        image_y = tile_y + 37

        sheet[
            image_y:
            image_y + collector.CROP_SIZE,
            image_x:
            image_x + collector.CROP_SIZE,
        ] = prediction.display_crop

        color = (
            (0, 255, 0)
            if prediction.accepted
            else (0, 0, 255)
        )

        cv2.putText(
            sheet,
            (
                f"Die {index}: "
                f"{prediction.value}"
            ),
            (
                tile_x + 12,
                tile_y + 24,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

        model_55 = prediction.model_55
        model_60 = prediction.model_60

        cv2.putText(
            sheet,
            (
                f"55%: {model_55.value} "
                f"{model_55.confidence:.0%} "
                f"votes {model_55.vote_fraction:.0%}"
            ),
            (
                tile_x + 12,
                tile_y + 187,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            (
                f"60%: {model_60.value} "
                f"{model_60.confidence:.0%} "
                f"votes {model_60.vote_fraction:.0%}"
            ),
            (
                tile_x + 12,
                tile_y + 209,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            (
                "ACCEPTED"
                if prediction.accepted
                else "UNCERTAIN"
            ),
            (
                tile_x + 12,
                tile_y + 231,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    return sheet


def raw_marker_view(
    frame: np.ndarray,
    corners,
    ids,
    marker_count: int,
) -> np.ndarray:
    output = frame.copy()

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(
            output,
            corners,
            ids,
        )

    return collector.add_status_banner(
        output,
        (
            f"ARUCO MARKERS {marker_count}/4 | "
            "adjust camera, lighting, or marker visibility"
        ),
        (0, 0, 255),
        height=58,
    )


def waiting_for_background_view(
    tray: np.ndarray,
    request: collector.RollRequest,
) -> np.ndarray:
    return collector.add_status_banner(
        tray,
        (
            f"REMOVE ALL DICE | press B for fresh background | "
            f"next {request.expression}"
        ),
        (0, 255, 255),
        height=58,
    )


def show_result_window(
    window_name: str,
    sheet: np.ndarray,
) -> None:
    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    maximum_width = 1180
    maximum_height = 760

    scale = min(
        1.0,
        maximum_width
        / max(
            sheet.shape[1],
            1,
        ),
        maximum_height
        / max(
            sheet.shape[0],
            1,
        ),
    )

    width = max(
        420,
        int(
            round(
                sheet.shape[1]
                * scale
            )
        ),
    )
    height = max(
        260,
        int(
            round(
                sheet.shape[0]
                * scale
            )
        ),
    )

    cv2.resizeWindow(
        window_name,
        width,
        height,
    )
    cv2.moveWindow(
        window_name,
        RESULT_WINDOW_X,
        RESULT_WINDOW_Y,
    )
    cv2.imshow(
        window_name,
        sheet,
    )
    cv2.waitKey(1)


def close_window_if_open(
    window_name: str,
) -> None:
    try:
        cv2.destroyWindow(
            window_name
        )
        cv2.waitKey(1)
    except cv2.error:
        pass


# ---------------------------------------------------------------------------
# Debug output
# ---------------------------------------------------------------------------

def prediction_record(
    prediction: D8Prediction,
    index: int,
) -> dict[str, object]:
    return {
        "die": index,
        "value": prediction.value,
        "accepted": prediction.accepted,
        "combined_confidence": (
            prediction.combined_confidence
        ),
        "model55": {
            "value": prediction.model_55.value,
            "confidence": (
                prediction.model_55.confidence
            ),
            "vote_fraction": (
                prediction.model_55.vote_fraction
            ),
            "probabilities": list(
                prediction.model_55.probabilities
            ),
        },
        "model60": {
            "value": prediction.model_60.value,
            "confidence": (
                prediction.model_60.confidence
            ),
            "vote_fraction": (
                prediction.model_60.vote_fraction
            ),
            "probabilities": list(
                prediction.model_60.probabilities
            ),
        },
    }


def save_debug(
    raw_frame: np.ndarray | None,
    tray: np.ndarray | None,
    mask: np.ndarray | None,
    annotated: np.ndarray | None,
    result: AggregationResult | None,
    result_sheet: np.ndarray | None,
    request: collector.RollRequest,
    bundle_55: ModelBundle,
    bundle_60: ModelBundle,
    model_threshold: float,
    vote_threshold: float,
) -> None:
    DEBUG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    stem = DEBUG_DIRECTORY / timestamp

    if raw_frame is not None:
        cv2.imwrite(
            str(
                Path(
                    f"{stem}_camera.png"
                )
            ),
            raw_frame,
        )

    if tray is not None:
        cv2.imwrite(
            str(
                Path(
                    f"{stem}_tray.png"
                )
            ),
            tray,
        )

    if mask is not None:
        cv2.imwrite(
            str(
                Path(
                    f"{stem}_mask.png"
                )
            ),
            mask,
        )

    if annotated is not None:
        cv2.imwrite(
            str(
                Path(
                    f"{stem}_annotated.png"
                )
            ),
            annotated,
        )

    if result_sheet is not None:
        cv2.imwrite(
            str(
                Path(
                    f"{stem}_predictions.png"
                )
            ),
            result_sheet,
        )

    metadata = {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "roll": request.expression,
        "model55": {
            "path": str(
                bundle_55.path
            ),
            "crop_fraction": (
                bundle_55.crop_fraction
            ),
        },
        "model60": {
            "path": str(
                bundle_60.path
            ),
            "crop_fraction": (
                bundle_60.crop_fraction
            ),
        },
        "model_threshold": model_threshold,
        "vote_threshold": vote_threshold,
        "frames_used": (
            result.frames_used
            if result is not None
            else None
        ),
        "selected_frames": (
            result.selected_frames
            if result is not None
            else None
        ),
        "predictions": (
            [
                prediction_record(
                    prediction,
                    index,
                )
                for index, prediction in enumerate(
                    result.predictions,
                    start=1,
                )
            ]
            if result is not None
            else []
        ),
    }

    Path(
        f"{stem}_result.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Debug files saved in:",
        DEBUG_DIRECTORY.resolve(),
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    count = (
        prompt_for_d8_count()
        if args.count is None
        else parse_d8_count(
            str(args.count)
        )
    )
    request = make_d8_request(
        count
    )

    bundle_55 = load_model_bundle(
        "55% model",
        args.model55,
    )
    bundle_60 = load_model_bundle(
        "60% model",
        args.model60,
    )

    print("Icy Dice d8 live reader")
    print("-----------------------")
    print(
        f"55% model: {bundle_55.path.resolve()}"
    )
    print(
        f"60% model: {bundle_60.path.resolve()}"
    )
    print(
        f"Device: {bundle_55.device}"
    )
    print(
        f"Per-model confidence threshold: "
        f"{args.model_threshold:.0%}"
    )
    print(
        f"Per-model frame-vote threshold: "
        f"{args.vote_threshold:.0%}"
    )
    print(
        "\nRemove all dice and press B for a fresh background."
    )
    print(
        f"Then roll {request.expression}; "
        "after the dice stop, press C."
    )

    camera = collector.open_camera()
    (
        detector,
        dictionary,
        parameters,
    ) = collector.create_aruco_detector()

    main_window = (
        "Icy Dice - d8 Live Reader"
    )
    result_window = (
        "Icy Dice - d8 Result Details"
    )

    cv2.namedWindow(
        main_window,
        cv2.WINDOW_NORMAL,
    )
    cv2.resizeWindow(
        main_window,
        1000,
        715,
    )

    background: np.ndarray | None = None
    background_ready = False
    result_active = False

    last_frame: np.ndarray | None = None
    last_rectified: np.ndarray | None = None
    last_mask: np.ndarray | None = None
    last_main_image: np.ndarray | None = None
    last_candidates: list[
        collector.Candidate
    ] = []
    last_result: AggregationResult | None = None
    last_result_sheet: np.ndarray | None = None

    print("\nControls")
    print(
        "  B  capture fresh empty-tray background"
    )
    print(
        "  C  capture and classify a one-second burst"
    )
    print(
        "  N  choose a new d8 count"
    )
    print(
        "  S  save current debug images and results"
    )
    print(
        "  Q  quit"
    )

    try:
        while True:
            ok, frame = camera.read()

            if not ok or frame is None:
                print(
                    "Camera frame read failed."
                )
                return 1

            last_frame = frame.copy()
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )
            (
                corners,
                ids,
                _rejected,
            ) = collector.detect_markers(
                gray,
                detector,
                dictionary,
                parameters,
            )
            marker_count = (
                0
                if ids is None
                else len(ids)
            )

            if marker_count == 4:
                source_points = (
                    collector.find_inward_marker_corners(
                        corners
                    )
                )
                last_rectified = (
                    collector.rectify_tray(
                        frame,
                        source_points,
                    )
                )

            if (
                result_active
                and last_main_image is not None
            ):
                cv2.imshow(
                    main_window,
                    last_main_image,
                )
            elif marker_count != 4:
                last_main_image = raw_marker_view(
                    frame,
                    corners,
                    ids,
                    marker_count,
                )
                cv2.imshow(
                    main_window,
                    last_main_image,
                )
            elif (
                background_ready
                and background is not None
                and last_rectified is not None
            ):
                last_mask = collector.foreground_mask(
                    last_rectified,
                    background,
                )
                (
                    last_candidates,
                    _component_labels,
                    split_count,
                ) = collector.detect_die_candidates(
                    last_mask,
                    request.count,
                )
                last_main_image = (
                    collector.annotate_candidates(
                        last_rectified,
                        last_candidates,
                        request,
                        split_count=split_count,
                    )
                )
                cv2.imshow(
                    main_window,
                    last_main_image,
                )
            elif last_rectified is not None:
                last_mask = None
                last_candidates = []
                last_main_image = (
                    waiting_for_background_view(
                        last_rectified,
                        request,
                    )
                )
                cv2.imshow(
                    main_window,
                    last_main_image,
                )

            command = collector.poll_command()

            if command == "q":
                return 0

            if command == "b":
                if (
                    marker_count != 4
                    or last_rectified is None
                ):
                    print(
                        "Cannot capture background: "
                        "all four markers are required."
                    )
                    continue

                if background is not None:
                    removal_mask = (
                        collector.foreground_mask(
                            last_rectified,
                            background,
                        )
                    )
                    remaining_regions = (
                        collector.connected_regions(
                            removal_mask
                        )
                    )

                    if remaining_regions:
                        print(
                            "Background capture blocked: "
                            "foreground objects are still visible. "
                            "Remove all dice first."
                        )
                        continue

                background = (
                    last_rectified.copy()
                )
                background_ready = True
                result_active = False
                last_result = None
                last_result_sheet = None
                last_mask = None
                last_candidates = []
                close_window_if_open(
                    result_window
                )

                collector.save_background(
                    background
                )
                print(
                    "Fresh background captured. "
                    f"Cue: roll {request.expression}; "
                    "after the dice stop, press C."
                )

            if command == "n":
                print(
                    "\nClick PowerShell to enter "
                    "the new d8 count."
                )
                count = prompt_for_d8_count(
                    request.count
                )
                request = make_d8_request(
                    count
                )

                background = None
                background_ready = False
                result_active = False
                last_result = None
                last_result_sheet = None
                last_mask = None
                last_candidates = []
                close_window_if_open(
                    result_window
                )

                print(
                    "\nRemove all dice and press B "
                    "for a fresh background."
                )
                print(
                    f"Next roll: {request.expression}."
                )

            if command == "s":
                save_debug(
                    raw_frame=last_frame,
                    tray=last_rectified,
                    mask=last_mask,
                    annotated=last_main_image,
                    result=last_result,
                    result_sheet=(
                        last_result_sheet
                    ),
                    request=request,
                    bundle_55=bundle_55,
                    bundle_60=bundle_60,
                    model_threshold=(
                        args.model_threshold
                    ),
                    vote_threshold=(
                        args.vote_threshold
                    ),
                )

            if command == "c":
                if result_active:
                    print(
                        "A result is already displayed. "
                        "Remove the dice and press B "
                        "for the next roll."
                    )
                    continue

                if (
                    not background_ready
                    or background is None
                ):
                    print(
                        "Read blocked: remove all dice "
                        "and press B for a fresh "
                        "background first."
                    )
                    continue

                print(
                    f"\nCapturing approximately "
                    f"{collector.BURST_DURATION_SECONDS:.1f} "
                    f"second for {request.expression}..."
                )

                selection, failure = (
                    collector.capture_ranked_burst(
                        camera=camera,
                        detector=detector,
                        dictionary=dictionary,
                        parameters=parameters,
                        background=background,
                        request=request,
                        burst_window=main_window,
                    )
                )

                if selection is None:
                    print(failure)
                    print(
                        "Leave the dice in place and "
                        "press C to try again."
                    )
                    continue

                try:
                    result = (
                        aggregate_selected_burst(
                            selection=selection,
                            background=background,
                            request=request,
                            bundle_55=bundle_55,
                            bundle_60=bundle_60,
                            model_threshold=(
                                args.model_threshold
                            ),
                            vote_threshold=(
                                args.vote_threshold
                            ),
                        )
                    )
                except RuntimeError as error:
                    print(
                        f"Classification failed: {error}"
                    )
                    print(
                        "Leave the dice in place and "
                        "press C to try again."
                    )
                    continue

                annotated = annotate_predictions(
                    result
                )
                result_sheet = (
                    build_prediction_sheet(
                        result
                    )
                )

                last_result = result
                last_result_sheet = result_sheet
                last_rectified = (
                    result.representative_tray
                )
                last_mask = (
                    result.representative_mask
                )
                last_main_image = annotated

                cv2.imshow(
                    main_window,
                    annotated,
                )
                show_result_window(
                    result_window,
                    result_sheet,
                )

                values = [
                    prediction.value
                    for prediction in (
                        result.predictions
                    )
                ]
                total = sum(values)
                all_accepted = all(
                    prediction.accepted
                    for prediction in (
                        result.predictions
                    )
                )

                print(
                    "\nRoll read:"
                )
                print(
                    f"  Burst frames selected: "
                    f"{result.selected_frames}"
                )
                print(
                    f"  Burst frames used: "
                    f"{result.frames_used}"
                )

                for index, prediction in enumerate(
                    result.predictions,
                    start=1,
                ):
                    state = (
                        "accepted"
                        if prediction.accepted
                        else "UNCERTAIN"
                    )
                    print(
                        f"  Die {index}: "
                        f"{prediction.value} "
                        f"({state})"
                    )
                    print(
                        f"    55% model: "
                        f"{prediction.model_55.value}, "
                        f"confidence "
                        f"{prediction.model_55.confidence:.1%}, "
                        f"frame votes "
                        f"{prediction.model_55.vote_fraction:.1%}"
                    )
                    print(
                        f"    60% model: "
                        f"{prediction.model_60.value}, "
                        f"confidence "
                        f"{prediction.model_60.confidence:.1%}, "
                        f"frame votes "
                        f"{prediction.model_60.vote_fraction:.1%}"
                    )

                print(
                    f"  Values: {values}"
                )
                print(
                    f"  Total: {total}"
                )

                if all_accepted:
                    print(
                        "  Result accepted."
                    )
                else:
                    print(
                        "  Result is provisional because "
                        "at least one die failed model "
                        "agreement, confidence, or "
                        "frame-vote requirements."
                    )

                print(
                    "\nRemove all dice and press B "
                    "before the next roll."
                )

                result_active = True
                background_ready = False

    finally:
        camera.release()
        close_window_if_open(
            result_window
        )
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
