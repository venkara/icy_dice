from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CLASS_NAMES = ("1", "2", "3", "4", "5", "6")

DEFAULT_INPUT = Path("dataset/d6")
DEFAULT_OUTPUT = Path("dataset_variants")

OUTPUT_SIZE = 128
CENTER_CROP_FRACTION = 0.68

# Die-mask extraction from the collector's gray-background crops.
BACKGROUND_BORDER_WIDTH = 7
MIN_DIE_COLOR_DISTANCE = 16.0
MIN_DIE_AREA = 250

# Numeral detection.
DIE_MASK_EROSION_ITERATIONS = 1
MIN_NUMERAL_COMPONENT_AREA = 8
MAX_NUMERAL_COMPONENT_FRACTION = 0.22
NUMERAL_CROP_MIN_FRACTION = 0.48
NUMERAL_CROP_MAX_FRACTION = 0.82
NUMERAL_BOX_SCALE = 3.0

SESSION_PATTERN = re.compile(
    r"^(?P<session>\d{8}_\d{6}_\d{6})_\d+\.[^.]+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumeralDetection:
    center_x: float
    center_y: float
    bbox: tuple[int, int, int, int] | None
    score: float
    confidence: float
    status: str


@dataclass(frozen=True)
class CropResult:
    image: np.ndarray
    bounds: tuple[int, int, int, int]
    detection: NumeralDetection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate center-cropped and numeral-centered d6 dataset variants "
            "from Icy Dice collector crops."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input dataset root containing class folders 1 through 6.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Root directory for generated dataset variants.",
    )
    parser.add_argument(
        "--center-fraction",
        type=float,
        default=CENTER_CROP_FRACTION,
        help="Fraction of the original crop retained by the center variant.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=72,
        help="Maximum number of images included in preview sheets.",
    )
    parser.add_argument(
        "--preview-page-size",
        type=int,
        default=24,
        help="Number of samples on each preview page.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260722,
        help="Random seed used to choose preview examples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing generated images.",
    )
    return parser.parse_args()


def session_id_from_path(path: Path) -> str:
    match = SESSION_PATTERN.match(path.name)

    if match is not None:
        return match.group("session")

    return path.stem


def list_dataset_images(root: Path) -> list[tuple[str, Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {root}")

    items: list[tuple[str, Path]] = []

    for label in CLASS_NAMES:
        class_directory = root / label

        if not class_directory.exists():
            raise FileNotFoundError(f"Missing class directory: {class_directory}")

        for path in sorted(class_directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                items.append((label, path))

    if not items:
        raise RuntimeError(f"No images found beneath {root}")

    return items


def estimate_background_color(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    border = min(
        BACKGROUND_BORDER_WIDTH,
        max(1, min(height, width) // 6),
    )

    pixels = np.concatenate(
        [
            image[:border, :, :].reshape(-1, 3),
            image[-border:, :, :].reshape(-1, 3),
            image[:, :border, :].reshape(-1, 3),
            image[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )

    return np.median(pixels.astype(np.float32), axis=0)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if count <= 1:
        return np.zeros_like(mask)

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(component_areas))

    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def build_die_mask(
    image: np.ndarray,
    background_color: np.ndarray,
) -> np.ndarray:
    difference = image.astype(np.float32) - background_color.reshape(1, 1, 3)
    color_distance = np.sqrt(np.sum(difference * difference, axis=2))

    # Adapt slightly to noisier borders, while retaining a useful minimum.
    border_width = min(
        BACKGROUND_BORDER_WIDTH,
        max(1, min(image.shape[:2]) // 6),
    )

    border_distance = np.concatenate(
        [
            color_distance[:border_width, :].ravel(),
            color_distance[-border_width:, :].ravel(),
            color_distance[:, :border_width].ravel(),
            color_distance[:, -border_width:].ravel(),
        ]
    )

    adaptive_threshold = float(np.percentile(border_distance, 95)) + 5.0
    threshold = max(MIN_DIE_COLOR_DISTANCE, adaptive_threshold)

    mask = np.where(
        color_distance >= threshold,
        255,
        0,
    ).astype(np.uint8)

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2,
    )

    mask = largest_component(mask)

    if cv2.countNonZero(mask) < MIN_DIE_AREA:
        # Conservative fallback: treat the central region as the die area.
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        margin_y = image.shape[0] // 5
        margin_x = image.shape[1] // 5
        mask[
            margin_y : image.shape[0] - margin_y,
            margin_x : image.shape[1] - margin_x,
        ] = 255

    return mask


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(mask, binaryImage=True)

    if moments["m00"] <= 0:
        height, width = mask.shape[:2]
        return width / 2.0, height / 2.0

    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


def detect_numeral(
    image: np.ndarray,
    die_mask: np.ndarray,
) -> NumeralDetection:
    """
    Detect the most likely top-face numeral.

    White numerals are identified using a combination of:
      * high brightness,
      * low saturation,
      * positive local contrast,
      * proximity to the die centroid,
      * and apparent component size.

    The centrality term is deliberately strong so a smaller side-face numeral
    is less likely to beat the top numeral.
    """
    die_area = cv2.countNonZero(die_mask)
    die_center_x, die_center_y = mask_centroid(die_mask)

    if die_area <= 0:
        return NumeralDetection(
            center_x=die_center_x,
            center_y=die_center_y,
            bbox=None,
            score=0.0,
            confidence=0.0,
            status="fallback-no-die-mask",
        )

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    saturation = hsv[:, :, 1].astype(np.float32)
    lightness = lab[:, :, 0].astype(np.float32)

    blurred = cv2.GaussianBlur(
        lightness,
        (0, 0),
        sigmaX=4.0,
        sigmaY=4.0,
    )
    local_contrast = lightness - blurred

    interior = die_mask.copy()

    if DIE_MASK_EROSION_ITERATIONS > 0:
        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )
        interior = cv2.erode(
            interior,
            erosion_kernel,
            iterations=DIE_MASK_EROSION_ITERATIONS,
        )

    die_pixels = interior > 0

    if not np.any(die_pixels):
        die_pixels = die_mask > 0

    light_values = lightness[die_pixels]
    saturation_values = saturation[die_pixels]
    contrast_values = local_contrast[die_pixels]

    bright_threshold = max(
        145.0,
        float(np.percentile(light_values, 72)),
    )
    low_saturation_threshold = float(
        np.clip(
            np.percentile(saturation_values, 40) + 35.0,
            55.0,
            145.0,
        )
    )
    contrast_threshold = max(
        7.0,
        float(np.percentile(contrast_values, 82)),
    )
    mid_light_threshold = float(np.percentile(light_values, 50))

    bright_neutral = (lightness >= bright_threshold) & (
        saturation <= low_saturation_threshold
    )
    locally_bright = (
        (local_contrast >= contrast_threshold)
        & (lightness >= mid_light_threshold)
        & (saturation <= 165.0)
    )

    numeral_mask = np.where(
        die_pixels & (bright_neutral | locally_bright),
        255,
        0,
    ).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )
    numeral_mask = cv2.morphologyEx(
        numeral_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )
    numeral_mask = cv2.dilate(
        numeral_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        numeral_mask,
        connectivity=8,
    )

    die_scale = math.sqrt(float(die_area))
    maximum_area = max(
        MIN_NUMERAL_COMPONENT_AREA + 1,
        int(MAX_NUMERAL_COMPONENT_FRACTION * die_area),
    )

    candidates: list[
        tuple[
            float,
            float,
            tuple[int, int, int, int],
            float,
            float,
        ]
    ] = []

    for component_label in range(1, count):
        area = int(stats[component_label, cv2.CC_STAT_AREA])

        if not (MIN_NUMERAL_COMPONENT_AREA <= area <= maximum_area):
            continue

        x = int(stats[component_label, cv2.CC_STAT_LEFT])
        y = int(stats[component_label, cv2.CC_STAT_TOP])
        width = int(stats[component_label, cv2.CC_STAT_WIDTH])
        height = int(stats[component_label, cv2.CC_STAT_HEIGHT])

        if width < 3 or height < 3:
            continue

        center_x, center_y = centroids[component_label]
        center_x = float(center_x)
        center_y = float(center_y)

        distance = math.hypot(
            center_x - die_center_x,
            center_y - die_center_y,
        )
        normalized_distance = distance / max(die_scale, 1.0)
        centrality = math.exp(-4.0 * normalized_distance * normalized_distance)

        component_pixels = labels == component_label
        mean_contrast = float(
            np.mean(np.maximum(local_contrast[component_pixels], 0.0))
        )
        mean_lightness = float(np.mean(lightness[component_pixels]))

        area_fraction = area / max(float(die_area), 1.0)
        size_score = math.sqrt(area_fraction)
        fill_ratio = area / max(float(width * height), 1.0)

        # Centrality receives the greatest weight because the project needs the
        # uppermost face, not simply the clearest visible numeral.
        score = (
            4.5 * centrality
            + 7.0 * size_score
            + 0.025 * mean_contrast
            + 0.35 * fill_ratio
            + 0.0015 * mean_lightness
        )

        candidates.append(
            (
                score,
                centrality,
                (x, y, width, height),
                center_x,
                center_y,
            )
        )

    if not candidates:
        return NumeralDetection(
            center_x=die_center_x,
            center_y=die_center_y,
            bbox=None,
            score=0.0,
            confidence=0.0,
            status="fallback-no-numeral",
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_centrality, best_bbox, best_x, best_y = candidates[0]

    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    score_margin = max(0.0, best_score - second_score)

    confidence = float(
        np.clip(
            0.55 * best_centrality
            + 0.30 * math.tanh(score_margin)
            + 0.15 * math.tanh(best_score / 5.0),
            0.0,
            1.0,
        )
    )

    status = "detected" if confidence >= 0.45 else "detected-low-confidence"

    return NumeralDetection(
        center_x=best_x,
        center_y=best_y,
        bbox=best_bbox,
        score=float(best_score),
        confidence=confidence,
        status=status,
    )


def padded_square_crop(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    side: int,
    fill_color: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    side = max(2, int(round(side)))

    x1 = int(round(center_x - side / 2))
    y1 = int(round(center_y - side / 2))
    x2 = x1 + side
    y2 = y1 + side

    output = np.empty(
        (side, side, image.shape[2]),
        dtype=image.dtype,
    )
    output[:, :] = np.clip(
        np.round(fill_color),
        0,
        255,
    ).astype(image.dtype)

    image_height, image_width = image.shape[:2]

    source_x1 = max(0, x1)
    source_y1 = max(0, y1)
    source_x2 = min(image_width, x2)
    source_y2 = min(image_height, y2)

    if source_x1 < source_x2 and source_y1 < source_y2:
        destination_x1 = source_x1 - x1
        destination_y1 = source_y1 - y1
        destination_x2 = destination_x1 + (source_x2 - source_x1)
        destination_y2 = destination_y1 + (source_y2 - source_y1)

        output[
            destination_y1:destination_y2,
            destination_x1:destination_x2,
        ] = image[
            source_y1:source_y2,
            source_x1:source_x2,
        ]

    return output, (x1, y1, x2, y2)


def resize_crop(crop: np.ndarray) -> np.ndarray:
    interpolation = (
        cv2.INTER_AREA if max(crop.shape[:2]) >= OUTPUT_SIZE else cv2.INTER_CUBIC
    )

    return cv2.resize(
        crop,
        (OUTPUT_SIZE, OUTPUT_SIZE),
        interpolation=interpolation,
    )


def make_center_crop(
    image: np.ndarray,
    background_color: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    side = max(
        2,
        int(round(min(height, width) * fraction)),
    )

    crop, bounds = padded_square_crop(
        image,
        center_x=width / 2.0,
        center_y=height / 2.0,
        side=side,
        fill_color=background_color,
    )

    return resize_crop(crop), bounds


def make_numeral_crop(
    image: np.ndarray,
    background_color: np.ndarray,
    die_mask: np.ndarray,
) -> CropResult:
    height, width = image.shape[:2]
    minimum_dimension = min(height, width)

    detection = detect_numeral(image, die_mask)
    die_center_x, die_center_y = mask_centroid(die_mask)

    if detection.bbox is not None:
        _x, _y, numeral_width, numeral_height = detection.bbox
        proposed_side = int(
            round(max(numeral_width, numeral_height) * NUMERAL_BOX_SCALE)
        )

        minimum_side = int(round(minimum_dimension * NUMERAL_CROP_MIN_FRACTION))
        maximum_side = int(round(minimum_dimension * NUMERAL_CROP_MAX_FRACTION))
        side = int(
            np.clip(
                proposed_side,
                minimum_side,
                maximum_side,
            )
        )

        # Pull the crop slightly toward the die centroid. This retains top-face
        # context and reduces the influence of an off-center side candidate.
        center_x = 0.75 * detection.center_x + 0.25 * die_center_x
        center_y = 0.75 * detection.center_y + 0.25 * die_center_y
    else:
        side = int(round(minimum_dimension * NUMERAL_CROP_MAX_FRACTION))
        center_x = die_center_x
        center_y = die_center_y

    crop, bounds = padded_square_crop(
        image,
        center_x=center_x,
        center_y=center_y,
        side=side,
        fill_color=background_color,
    )

    return CropResult(
        image=resize_crop(crop),
        bounds=bounds,
        detection=detection,
    )


def safe_write_image(
    path: Path,
    image: np.ndarray,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}\n"
            "Use --overwrite to replace generated files."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save image: {path}")


def draw_debug_overlay(
    image: np.ndarray,
    die_mask: np.ndarray,
    center_bounds: tuple[int, int, int, int],
    numeral_result: CropResult,
) -> np.ndarray:
    overlay = image.copy()

    contours, _hierarchy = cv2.findContours(
        die_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 255),
        1,
    )

    cx1, cy1, cx2, cy2 = center_bounds
    cv2.rectangle(
        overlay,
        (cx1, cy1),
        (cx2 - 1, cy2 - 1),
        (255, 0, 0),
        1,
    )

    nx1, ny1, nx2, ny2 = numeral_result.bounds
    cv2.rectangle(
        overlay,
        (nx1, ny1),
        (nx2 - 1, ny2 - 1),
        (0, 255, 0),
        2,
    )

    detection = numeral_result.detection

    if detection.bbox is not None:
        x, y, width, height = detection.bbox
        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            1,
        )
        cv2.circle(
            overlay,
            (
                int(round(detection.center_x)),
                int(round(detection.center_y)),
            ),
            3,
            (0, 0, 255),
            -1,
        )

    return overlay


def add_label_bar(
    image: np.ndarray,
    text: str,
    bar_height: int = 24,
) -> np.ndarray:
    output = np.full(
        (image.shape[0] + bar_height, image.shape[1], 3),
        245,
        dtype=np.uint8,
    )
    output[bar_height:] = image

    cv2.putText(
        output,
        text,
        (4, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return output


def build_preview_tile(
    label: str,
    filename: str,
    original: np.ndarray,
    overlay: np.ndarray,
    center_crop: np.ndarray,
    numeral_crop: np.ndarray,
    detection: NumeralDetection,
) -> np.ndarray:
    display_size = 128

    original_display = cv2.resize(
        original,
        (display_size, display_size),
        interpolation=cv2.INTER_AREA,
    )
    overlay_display = cv2.resize(
        overlay,
        (display_size, display_size),
        interpolation=cv2.INTER_AREA,
    )

    panels = [
        add_label_bar(original_display, "original"),
        add_label_bar(overlay_display, "selection"),
        add_label_bar(center_crop, "center"),
        add_label_bar(numeral_crop, "numeral"),
    ]

    strip = cv2.hconcat(panels)

    header_height = 38
    tile = np.full(
        (strip.shape[0] + header_height, strip.shape[1], 3),
        255,
        dtype=np.uint8,
    )

    tile[header_height:] = strip

    description = (
        f"true {label} | {detection.status} | " f"confidence {detection.confidence:.2f}"
    )
    cv2.putText(
        tile,
        description,
        (5, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    short_name = filename[:55]
    cv2.putText(
        tile,
        short_name,
        (5, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    return tile


def save_preview_pages(
    previews: list[np.ndarray],
    output_directory: Path,
    page_size: int,
) -> list[Path]:
    if not previews:
        return []

    preview_directory = output_directory / "previews"
    preview_directory.mkdir(parents=True, exist_ok=True)

    columns = 2
    page_paths: list[Path] = []

    for page_index, page_start in enumerate(
        range(0, len(previews), page_size),
        start=1,
    ):
        page_tiles = previews[page_start : page_start + page_size]

        tile_height = max(tile.shape[0] for tile in page_tiles)
        tile_width = max(tile.shape[1] for tile in page_tiles)
        rows = math.ceil(len(page_tiles) / columns)

        page = np.full(
            (
                rows * (tile_height + 12) + 12,
                columns * (tile_width + 12) + 12,
                3,
            ),
            235,
            dtype=np.uint8,
        )

        for tile_index, tile in enumerate(page_tiles):
            row = tile_index // columns
            column = tile_index % columns

            x = 12 + column * (tile_width + 12)
            y = 12 + row * (tile_height + 12)

            page[
                y : y + tile.shape[0],
                x : x + tile.shape[1],
            ] = tile

        page_path = preview_directory / f"crop_preview_{page_index:02d}.jpg"

        if not cv2.imwrite(
            str(page_path),
            page,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        ):
            raise RuntimeError(f"Could not save preview page: {page_path}")

        page_paths.append(page_path)

    return page_paths


def main() -> int:
    args = parse_args()

    if not (0.35 <= args.center_fraction <= 0.95):
        raise ValueError("--center-fraction must be between 0.35 and 0.95.")

    if args.preview_page_size < 1:
        raise ValueError("--preview-page-size must be at least 1.")

    items = list_dataset_images(args.input)

    center_root = args.output / "d6_center"
    numeral_root = args.output / "d6_numeral"
    debug_root = args.output / "debug_overlays"

    if not args.overwrite:
        for root in (center_root, numeral_root, debug_root):
            if root.exists() and any(root.rglob("*")):
                raise FileExistsError(
                    f"Output directory is not empty: {root}\n"
                    "Choose another --output directory or use --overwrite."
                )

    args.output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    preview_indices = set(
        rng.sample(
            range(len(items)),
            k=min(args.preview_count, len(items)),
        )
    )

    manifest_rows: list[dict[str, object]] = []
    preview_tiles: list[np.ndarray] = []
    status_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()

    for index, (label, source_path) in enumerate(items):
        image = cv2.imread(
            str(source_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(f"Could not read image: {source_path}")

        background_color = estimate_background_color(image)
        die_mask = build_die_mask(
            image,
            background_color,
        )

        center_crop, center_bounds = make_center_crop(
            image,
            background_color,
            args.center_fraction,
        )
        numeral_result = make_numeral_crop(
            image,
            background_color,
            die_mask,
        )

        center_path = center_root / label / source_path.name
        numeral_path = numeral_root / label / source_path.name

        safe_write_image(
            center_path,
            center_crop,
            overwrite=args.overwrite,
        )
        safe_write_image(
            numeral_path,
            numeral_result.image,
            overwrite=args.overwrite,
        )

        overlay = draw_debug_overlay(
            image,
            die_mask,
            center_bounds,
            numeral_result,
        )

        if index in preview_indices:
            debug_path = debug_root / label / source_path.name
            safe_write_image(
                debug_path,
                overlay,
                overwrite=args.overwrite,
            )

            preview_tiles.append(
                build_preview_tile(
                    label=label,
                    filename=source_path.name,
                    original=image,
                    overlay=overlay,
                    center_crop=center_crop,
                    numeral_crop=numeral_result.image,
                    detection=numeral_result.detection,
                )
            )

        detection = numeral_result.detection
        status_counts[detection.status] += 1
        class_counts[label] += 1

        manifest_rows.append(
            {
                "label": label,
                "session_id": session_id_from_path(source_path),
                "filename": source_path.name,
                "source_path": str(source_path),
                "center_path": str(center_path),
                "numeral_path": str(numeral_path),
                "center_bounds": list(center_bounds),
                "numeral_bounds": list(numeral_result.bounds),
                "numeral_bbox": (
                    list(detection.bbox) if detection.bbox is not None else None
                ),
                "numeral_status": detection.status,
                "numeral_score": detection.score,
                "numeral_confidence": detection.confidence,
            }
        )

        if (index + 1) % 100 == 0 or index + 1 == len(items):
            print(f"Processed {index + 1}/{len(items)} images...")

    manifest_path = args.output / "crop_manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as manifest_file:
        fieldnames = [
            "label",
            "session_id",
            "filename",
            "source_path",
            "center_path",
            "numeral_path",
            "center_bounds",
            "numeral_bounds",
            "numeral_bbox",
            "numeral_status",
            "numeral_score",
            "numeral_confidence",
        ]

        writer = csv.DictWriter(
            manifest_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    preview_paths = save_preview_pages(
        preview_tiles,
        args.output,
        page_size=args.preview_page_size,
    )

    summary = {
        "input_root": str(args.input.resolve()),
        "output_root": str(args.output.resolve()),
        "image_count": len(items),
        "class_counts": dict(sorted(class_counts.items())),
        "center_fraction": args.center_fraction,
        "center_dataset": str(center_root.resolve()),
        "numeral_dataset": str(numeral_root.resolve()),
        "status_counts": dict(sorted(status_counts.items())),
        "manifest": str(manifest_path.resolve()),
        "preview_pages": [str(path.resolve()) for path in preview_paths],
    }

    summary_path = args.output / "crop_summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(summary, summary_file, indent=2)

    print("\nGenerated dataset variants:")
    print(f"  Center crop:   {center_root.resolve()}")
    print(f"  Numeral crop:  {numeral_root.resolve()}")
    print(f"  Manifest:      {manifest_path.resolve()}")
    print(f"  Summary:       {summary_path.resolve()}")

    if preview_paths:
        print("  Preview pages:")

        for path in preview_paths:
            print(f"    {path.resolve()}")

    print("\nNumeral-detection status:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:26s} {count}")

    print("\nTrain both variants with:")
    print(
        "  py train_d6_classifier.py "
        "--data dataset_variants/d6_center "
        "--output models/d6_center"
    )
    print(
        "  py train_d6_classifier.py "
        "--data dataset_variants/d6_numeral "
        "--output models/d6_numeral"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
