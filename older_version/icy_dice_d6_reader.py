from __future__ import annotations

import argparse
import json
import math
import re
import sys

try:
    import msvcrt
except ImportError:
    msvcrt = None
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 60

OUTPUT_WIDTH = 1200
OUTPUT_HEIGHT = 800

BACKGROUND_PATH = Path("calibration/empty_tray.png")
DATASET_DIRECTORY = Path("dataset")
SESSION_DIRECTORY = Path("sessions")
DEBUG_DIRECTORY = Path("captures")

# These margins are used only by motion detection. They must not be
# applied to the foreground mask, because doing so cuts off dice near a wall.
MOTION_MARGIN_X = 45
MOTION_MARGIN_Y = 45

FOREGROUND_THRESHOLD = 28
MIN_DIE_AREA = 500
MAX_DIE_AREA = 40_000

# A candidate centroid may be very near the tray wall, but not literally on
# the rectified image boundary.
CANDIDATE_CENTER_MARGIN = 3

# Gentle morphology removes isolated felt noise without joining nearby dice.
FOREGROUND_OPEN_KERNEL = 3
FOREGROUND_CLOSE_KERNEL = 5
FOREGROUND_CLOSE_ITERATIONS = 1

# If fewer components are found than the requested die count, spatial k-means
# is used to split the largest merged foreground region.
KMEANS_ATTEMPTS = 10
MERGED_REGION_AREA_RATIO = 1.35

CROP_SIZE = 128
CROP_PADDING_FRACTION = 0.32
MASK_BACKGROUND_BGR = (128, 128, 128)

# The winning preprocessing experiment retained the central 68% of each
# masked die crop before classification.
CENTER_CROP_FRACTION = 0.68
DEFAULT_MODEL_PATH = Path("models/d6_center/d6_mobilenet_v3_small.pt")
DEFAULT_CONFIDENCE_THRESHOLD = 0.75

STABLE_MOTION_THRESHOLD = 1.8
STABLE_FRAMES_REQUIRED = 18
MOTION_IMAGE_WIDTH = 300

ROLL_PATTERN = re.compile(
    r"^\s*(?P<count>\d*)\s*d\s*(?P<die>4|6|8|10|12|20|00|%)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RollRequest:
    count: int
    die_type: str

    @property
    def expression(self) -> str:
        return f"{self.count}{self.die_type}"

    @property
    def allowed_labels(self) -> tuple[str, ...]:
        if self.die_type == "d00":
            return tuple(f"{value:02d}" for value in range(0, 100, 10))

        sides = int(self.die_type[1:])

        # Train a d10 on its printed markings, 0 through 9.
        if sides == 10:
            return tuple(str(value) for value in range(10))

        return tuple(str(value) for value in range(1, sides + 1))


@dataclass(frozen=True)
class Candidate:
    component_label: int
    bbox: tuple[int, int, int, int]
    area: int
    centroid: tuple[float, float]


@dataclass(frozen=True)
class CandidateImages:
    raw: np.ndarray
    masked: np.ndarray
    component_mask: np.ndarray


@dataclass(frozen=True)
class DiePrediction:
    value: int
    confidence: float
    probabilities: tuple[float, ...]
    center_crop: np.ndarray


def open_camera() -> cv2.VideoCapture:
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open camera.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return camera


def create_aruco_detector():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This OpenCV installation does not include cv2.aruco.\n"
            "Install opencv-contrib-python."
        )

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return detector, None, None

    return None, dictionary, parameters


def detect_markers(gray, detector, dictionary, parameters):
    if detector is not None:
        return detector.detectMarkers(gray)

    return cv2.aruco.detectMarkers(
        gray,
        dictionary,
        parameters=parameters,
    )


def order_points(points: np.ndarray) -> np.ndarray:
    """Return points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).ravel()

    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[2] = points[np.argmax(coordinate_sum)]
    ordered[1] = points[np.argmin(coordinate_difference)]
    ordered[3] = points[np.argmax(coordinate_difference)]

    return ordered


def find_inward_marker_corners(marker_corners: list[np.ndarray]) -> np.ndarray:
    centers = np.array(
        [corners.reshape(4, 2).mean(axis=0) for corners in marker_corners],
        dtype=np.float32,
    )
    arrangement_center = centers.mean(axis=0)

    inward_corners: list[np.ndarray] = []

    for corners in marker_corners:
        points = corners.reshape(4, 2)
        distances = np.linalg.norm(points - arrangement_center, axis=1)
        inward_corners.append(points[np.argmin(distances)])

    return order_points(np.array(inward_corners))


def rectify_tray(frame: np.ndarray, source_points: np.ndarray) -> np.ndarray:
    destination_points = np.array(
        [
            [0, 0],
            [OUTPUT_WIDTH - 1, 0],
            [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
            [0, OUTPUT_HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(source_points, destination_points)

    return cv2.warpPerspective(
        frame,
        transform,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )


def save_background(rectified: np.ndarray) -> None:
    BACKGROUND_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(BACKGROUND_PATH), rectified):
        raise RuntimeError("Could not save empty-tray background.")

    print(f"Background saved: {BACKGROUND_PATH.resolve()}")


def load_background() -> np.ndarray | None:
    if not BACKGROUND_PATH.exists():
        return None

    background = cv2.imread(str(BACKGROUND_PATH))

    if background is None:
        raise RuntimeError(f"Could not read background image: {BACKGROUND_PATH}")

    return background


def foreground_mask(tray: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Find pixels that changed relative to the freshly captured empty tray.

    The full rectified frame is retained. In particular, no hard border is
    erased here; the old 45-pixel ROI was the reason edge dice were cropped.
    """
    if tray.shape != background.shape:
        raise ValueError("Current tray and background have different dimensions.")

    tray_lab = cv2.cvtColor(tray, cv2.COLOR_BGR2LAB)
    background_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)

    difference = cv2.absdiff(tray_lab, background_lab).astype(np.float32)
    magnitude = np.sqrt(np.sum(difference * difference, axis=2))

    mask = np.where(
        magnitude >= FOREGROUND_THRESHOLD,
        255,
        0,
    ).astype(np.uint8)

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (FOREGROUND_OPEN_KERNEL, FOREGROUND_OPEN_KERNEL),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (FOREGROUND_CLOSE_KERNEL, FOREGROUND_CLOSE_KERNEL),
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
        iterations=FOREGROUND_CLOSE_ITERATIONS,
    )

    return mask


def region_is_plausible(
    coordinates: np.ndarray,
    image_shape: tuple[int, ...],
) -> bool:
    """
    Reject obvious edge artifacts while allowing a real die very near a wall.
    """
    if len(coordinates) < MIN_DIE_AREA:
        return False

    image_height, image_width = image_shape[:2]

    center_y, center_x = coordinates.mean(axis=0)

    if not (
        CANDIDATE_CENTER_MARGIN <= center_x < image_width - CANDIDATE_CENTER_MARGIN
    ):
        return False

    if not (
        CANDIDATE_CENTER_MARGIN <= center_y < image_height - CANDIDATE_CENTER_MARGIN
    ):
        return False

    y_values = coordinates[:, 0]
    x_values = coordinates[:, 1]
    width = int(x_values.max() - x_values.min() + 1)
    height = int(y_values.max() - y_values.min() + 1)

    # A changed border or lighting failure can create a frame-sized component.
    if width > 0.85 * image_width and height > 0.85 * image_height:
        return False

    return True


def connected_regions(mask: np.ndarray) -> list[np.ndarray]:
    """
    Return each plausible connected foreground component as [y, x] pixels.

    Large components are retained because they may contain multiple dice that
    need to be separated.
    """
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    regions: list[np.ndarray] = []

    for component_label in range(1, count):
        area = int(stats[component_label, cv2.CC_STAT_AREA])

        if area < MIN_DIE_AREA:
            continue

        coordinates = np.column_stack(np.where(labels == component_label)).astype(
            np.int32
        )

        if region_is_plausible(coordinates, mask.shape):
            regions.append(coordinates)

    return regions


def split_region_spatially(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Divide one fused foreground region into two spatial clusters.

    The foreground pixels themselves are clustered, so a narrow shadow bridge
    between two nearby dice no longer forces them to be one candidate.
    """
    if len(coordinates) < 2 * MIN_DIE_AREA:
        return None

    # OpenCV k-means expects samples as [x, y] float32 rows.
    samples = coordinates[:, [1, 0]].astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.25,
    )

    _compactness, cluster_ids, _centers = cv2.kmeans(
        samples,
        2,
        None,
        criteria,
        KMEANS_ATTEMPTS,
        cv2.KMEANS_PP_CENTERS,
    )

    cluster_ids = cluster_ids.ravel()
    first = coordinates[cluster_ids == 0]
    second = coordinates[cluster_ids == 1]

    if len(first) < MIN_DIE_AREA or len(second) < MIN_DIE_AREA:
        return None

    return first, second


def split_merged_regions(
    regions: list[np.ndarray],
    expected_count: int,
) -> tuple[list[np.ndarray], int]:
    """
    Split the largest likely merged region until the requested count is met.

    Because the roll expression gives us the expected number of dice, we can
    use that information instead of accepting one large connected blob.
    """
    regions = list(regions)
    split_count = 0

    while regions and len(regions) < expected_count:
        areas = np.asarray([len(region) for region in regions], dtype=np.float32)
        largest_index = int(np.argmax(areas))
        largest_area = float(areas[largest_index])

        if len(regions) > 1:
            other_areas = np.delete(areas, largest_index)
            typical_area = float(np.median(other_areas))

            # Avoid splitting a normal die merely because another die failed
            # to segment at all.
            if largest_area < MERGED_REGION_AREA_RATIO * typical_area:
                break
        else:
            # With only one component, require enough foreground for at least
            # two plausible dice.
            if largest_area < 2 * MIN_DIE_AREA:
                break

        result = split_region_spatially(regions[largest_index])

        if result is None:
            break

        first, second = result
        regions.pop(largest_index)
        regions.extend((first, second))
        split_count += 1

    return regions, split_count


def build_candidates_from_regions(
    regions: list[np.ndarray],
    image_shape: tuple[int, ...],
) -> tuple[list[Candidate], np.ndarray]:
    """
    Convert pixel-coordinate regions into the label map used for saved crops.
    """
    component_labels = np.zeros(image_shape[:2], dtype=np.int32)
    candidates: list[Candidate] = []

    for component_label, coordinates in enumerate(regions, start=1):
        area = len(coordinates)

        if area < MIN_DIE_AREA:
            continue

        y_values = coordinates[:, 0]
        x_values = coordinates[:, 1]

        x = int(x_values.min())
        y = int(y_values.min())
        width = int(x_values.max() - x + 1)
        height = int(y_values.max() - y + 1)
        center_x = float(x_values.mean())
        center_y = float(y_values.mean())

        component_labels[y_values, x_values] = component_label

        candidates.append(
            Candidate(
                component_label=component_label,
                bbox=(x, y, width, height),
                area=area,
                centroid=(center_x, center_y),
            )
        )

    # Number candidates top-to-bottom and then left-to-right.
    candidates.sort(
        key=lambda candidate: (
            round(candidate.centroid[1] / 80),
            candidate.centroid[0],
        )
    )

    return candidates, component_labels


def detect_die_candidates(
    mask: np.ndarray,
    expected_count: int,
) -> tuple[list[Candidate], np.ndarray, int]:
    regions = connected_regions(mask)
    regions, split_count = split_merged_regions(
        regions,
        expected_count,
    )

    candidates, component_labels = build_candidates_from_regions(
        regions,
        mask.shape,
    )

    return candidates, component_labels, split_count


def annotate_candidates(
    tray: np.ndarray,
    candidates: list[Candidate],
    request: RollRequest,
    stable: bool,
    motion_score: float,
    split_count: int = 0,
) -> np.ndarray:
    output = tray.copy()

    count_matches = len(candidates) == request.count
    box_color = (0, 255, 0) if count_matches else (0, 165, 255)

    for index, candidate in enumerate(candidates, start=1):
        x, y, width, height = candidate.bbox

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            box_color,
            3,
        )

        cv2.circle(output, (x + 18, y + 18), 16, (0, 0, 0), -1)
        cv2.putText(
            output,
            str(index),
            (x + 9, y + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"area {candidate.area}",
            (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
            cv2.LINE_AA,
        )

    ready = count_matches and stable
    status_color = (0, 255, 0) if ready else (0, 165, 255)
    split_text = f" | auto-split {split_count}" if split_count else ""
    status_line = (
        f"Target {request.expression} | "
        f"detected {len(candidates)}/{request.count}"
        f"{split_text} | "
        f"motion {motion_score:.2f} | "
        f"{'READY - press C' if ready else 'waiting'}"
    )

    cv2.rectangle(output, (0, 0), (OUTPUT_WIDTH, 54), (0, 0, 0), -1)
    cv2.putText(
        output,
        status_line,
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )

    return output


def motion_frame(tray: np.ndarray) -> np.ndarray:
    roi = tray[
        MOTION_MARGIN_Y : tray.shape[0] - MOTION_MARGIN_Y,
        MOTION_MARGIN_X : tray.shape[1] - MOTION_MARGIN_X,
    ]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    output_height = max(
        1,
        int(gray.shape[0] * MOTION_IMAGE_WIDTH / gray.shape[1]),
    )
    small = cv2.resize(
        gray,
        (MOTION_IMAGE_WIDTH, output_height),
        interpolation=cv2.INTER_AREA,
    )

    return cv2.GaussianBlur(small, (5, 5), 0)


def calculate_motion_score(
    current: np.ndarray,
    previous: np.ndarray | None,
) -> float:
    if previous is None or current.shape != previous.shape:
        return float("inf")

    return float(cv2.absdiff(current, previous).mean())


def parse_roll_request(text: str) -> RollRequest:
    normalized = text.strip().lower().replace("percentile", "d00")
    match = ROLL_PATTERN.fullmatch(normalized)

    if match is None:
        raise ValueError(
            "Use notation such as 4d6, d20, 2d10, or 3d00 "
            "(d00 means a percentile die)."
        )

    count_text = match.group("count")
    count = int(count_text) if count_text else 1

    if count < 1 or count > 30:
        raise ValueError("The die count must be between 1 and 30.")

    die_text = match.group("die")
    die_type = "d00" if die_text in {"00", "%"} else f"d{die_text}"

    return RollRequest(count=count, die_type=die_type)


def prompt_for_roll_request(current: RollRequest | None = None) -> RollRequest:
    while True:
        if current is None:
            prompt = "Roll to collect (examples: 4d6, 2d20, 3d00): "
        else:
            prompt = f"New roll [{current.expression}] (Enter keeps current): "

        response = input(prompt).strip()

        if not response and current is not None:
            return current

        try:
            request = parse_roll_request(response)
        except ValueError as error:
            print(error)
            continue

        print(
            f"\nCue: roll {request.count} {request.die_type} "
            f"{'die' if request.count == 1 else 'dice'}."
        )
        print("The app will show READY when the count matches and they are still.")
        return request


def normalize_label(raw_label: str, request: RollRequest) -> str:
    label = raw_label.strip()

    if request.die_type == "d00":
        if not label.isdigit():
            raise ValueError(f"{label!r} is not a percentile marking.")

        numeric = int(label)
        if numeric not in range(0, 100, 10):
            raise ValueError(f"{label!r} is not one of 00, 10, 20, ..., 90.")

        return f"{numeric:02d}"

    if not label.isdigit():
        raise ValueError(f"{label!r} is not a numeric die marking.")

    normalized = str(int(label))

    if normalized not in request.allowed_labels:
        raise ValueError(
            f"{label!r} is not valid for {request.die_type}. "
            f"Allowed: {', '.join(request.allowed_labels)}"
        )

    return normalized


def prompt_for_labels(request: RollRequest) -> list[str] | None:
    print("\nCheck the numbered candidates in the verification window.")
    print("Enter their visible markings in candidate-number order.")

    if request.die_type == "d10":
        print("For d10, enter the printed 0 rather than semantic value 10.")
    elif request.die_type == "d00":
        print("For d00, use 00, 10, 20, ..., 90.")

    print("Enter R to reject this capture.")

    while True:
        response = input("Visible markings: ").strip()

        if response.lower() == "r":
            print("Capture rejected. Reroll or reposition the dice.")
            return None

        pieces = response.replace(",", " ").split()

        if len(pieces) != request.count:
            print(f"Expected {request.count} markings; received {len(pieces)}.")
            continue

        try:
            return [normalize_label(piece, request) for piece in pieces]
        except ValueError as error:
            print(error)


def square_bounds(
    candidate: Candidate,
) -> tuple[int, int, int, int]:
    """
    Return the desired square crop, even when it extends beyond the image.

    The extraction routine pads the unavailable part instead of shifting the
    crop and pushing an edge die off-center.
    """
    _x, _y, width, height = candidate.bbox
    center_x, center_y = candidate.centroid

    side = max(width, height)
    padding = max(8, int(round(side * CROP_PADDING_FRACTION)))
    side += 2 * padding

    x1 = int(round(center_x - side / 2))
    y1 = int(round(center_y - side / 2))
    x2 = x1 + side
    y2 = y1 + side

    return x1, y1, x2, y2


def extract_with_padding(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    fill_value,
) -> np.ndarray:
    """
    Extract a crop that may cross the image boundary and pad the remainder.
    """
    x1, y1, x2, y2 = bounds
    output_width = x2 - x1
    output_height = y2 - y1

    if image.ndim == 2:
        output = np.full(
            (output_height, output_width),
            fill_value,
            dtype=image.dtype,
        )
    else:
        output = np.full(
            (output_height, output_width, image.shape[2]),
            fill_value,
            dtype=image.dtype,
        )

    image_height, image_width = image.shape[:2]

    source_x1 = max(0, x1)
    source_y1 = max(0, y1)
    source_x2 = min(image_width, x2)
    source_y2 = min(image_height, y2)

    if source_x1 >= source_x2 or source_y1 >= source_y2:
        return output

    destination_x1 = source_x1 - x1
    destination_y1 = source_y1 - y1
    destination_x2 = destination_x1 + (source_x2 - source_x1)
    destination_y2 = destination_y1 + (source_y2 - source_y1)

    output[
        destination_y1:destination_y2,
        destination_x1:destination_x2,
    ] = image[source_y1:source_y2, source_x1:source_x2]

    return output


def extract_candidate_images(
    tray: np.ndarray,
    component_labels: np.ndarray,
    candidate: Candidate,
) -> CandidateImages:
    bounds = square_bounds(candidate)

    raw = extract_with_padding(
        tray,
        bounds,
        MASK_BACKGROUND_BGR,
    )

    label_crop = extract_with_padding(
        component_labels,
        bounds,
        0,
    )

    component_mask = np.where(
        label_crop == candidate.component_label,
        255,
        0,
    ).astype(np.uint8)

    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )
    expanded_mask = cv2.dilate(
        component_mask,
        dilation_kernel,
        iterations=1,
    )

    masked = np.full_like(raw, MASK_BACKGROUND_BGR)
    masked[expanded_mask > 0] = raw[expanded_mask > 0]

    raw = cv2.resize(
        raw,
        (CROP_SIZE, CROP_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    masked = cv2.resize(
        masked,
        (CROP_SIZE, CROP_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    component_mask = cv2.resize(
        component_mask,
        (CROP_SIZE, CROP_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    return CandidateImages(
        raw=raw,
        masked=masked,
        component_mask=component_mask,
    )


def build_contact_sheet(candidate_images: list[CandidateImages]) -> np.ndarray:
    tile_size = CROP_SIZE + 44
    columns = min(5, max(1, len(candidate_images)))
    rows = (len(candidate_images) + columns - 1) // columns

    sheet = np.full(
        (rows * tile_size, columns * tile_size, 3),
        40,
        dtype=np.uint8,
    )

    for index, images in enumerate(candidate_images, start=1):
        row = (index - 1) // columns
        column = (index - 1) % columns
        x = column * tile_size + 22
        y = row * tile_size + 34

        sheet[y : y + CROP_SIZE, x : x + CROP_SIZE] = images.masked
        cv2.putText(
            sheet,
            f"Candidate {index}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return sheet


def save_capture(
    request: RollRequest,
    labels: list[str],
    tray: np.ndarray,
    foreground: np.ndarray,
    candidates: list[Candidate],
    annotated: np.ndarray,
    candidate_images: list[CandidateImages],
) -> Path:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_path = SESSION_DIRECTORY / session_id
    raw_crop_path = session_path / "crops_raw"
    masked_crop_path = session_path / "crops_masked"
    mask_crop_path = session_path / "component_masks"

    raw_crop_path.mkdir(parents=True, exist_ok=True)
    masked_crop_path.mkdir(parents=True, exist_ok=True)
    mask_crop_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(session_path / "tray.png"), tray)
    cv2.imwrite(str(session_path / "foreground_mask.png"), foreground)
    cv2.imwrite(str(session_path / "annotated.png"), annotated)

    metadata_candidates: list[dict[str, object]] = []

    for index, (label, candidate, images) in enumerate(
        zip(labels, candidates, candidate_images, strict=True),
        start=1,
    ):
        image_name = f"{session_id}_{index:02d}.png"

        raw_file = raw_crop_path / image_name
        masked_file = masked_crop_path / image_name
        component_mask_file = mask_crop_path / image_name

        cv2.imwrite(str(raw_file), images.raw)
        cv2.imwrite(str(masked_file), images.masked)
        cv2.imwrite(str(component_mask_file), images.component_mask)

        dataset_path = DATASET_DIRECTORY / request.die_type / label
        dataset_path.mkdir(parents=True, exist_ok=True)
        dataset_file = dataset_path / image_name
        cv2.imwrite(str(dataset_file), images.masked)

        metadata_candidates.append(
            {
                "candidate_index": index,
                "visible_marking": label,
                "bbox": list(candidate.bbox),
                "area": candidate.area,
                "centroid": list(candidate.centroid),
                "dataset_file": str(dataset_file),
                "raw_crop_file": str(raw_file),
                "masked_crop_file": str(masked_file),
                "component_mask_file": str(component_mask_file),
            }
        )

    metadata = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "roll_expression": request.expression,
        "die_type": request.die_type,
        "expected_count": request.count,
        "visible_markings": labels,
        "rectified_size": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
        "crop_size": [CROP_SIZE, CROP_SIZE],
        "foreground_threshold": FOREGROUND_THRESHOLD,
        "foreground_close_kernel": FOREGROUND_CLOSE_KERNEL,
        "foreground_close_iterations": FOREGROUND_CLOSE_ITERATIONS,
        "edge_mask_removed": True,
        "candidates": metadata_candidates,
    }

    with (session_path / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return session_path


def save_debug_images(
    frame: np.ndarray,
    tray: np.ndarray | None,
    mask: np.ndarray | None,
    annotated: np.ndarray | None,
) -> None:
    DEBUG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    cv2.imwrite(str(DEBUG_DIRECTORY / f"{timestamp}_camera.png"), frame)

    if tray is not None:
        cv2.imwrite(str(DEBUG_DIRECTORY / f"{timestamp}_tray.png"), tray)
    if mask is not None:
        cv2.imwrite(str(DEBUG_DIRECTORY / f"{timestamp}_mask.png"), mask)
    if annotated is not None:
        cv2.imwrite(str(DEBUG_DIRECTORY / f"{timestamp}_annotated.png"), annotated)

    print(f"Debug images saved in: {DEBUG_DIRECTORY.resolve()}")


def placeholder_image(message: str) -> np.ndarray:
    image = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        image,
        message,
        (45, OUTPUT_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def poll_command() -> str | None:
    """
    Read one command from either an OpenCV window or the Windows console.

    OpenCV normally receives keys only while one of its windows has focus.
    msvcrt lets the same B/C/N/S/Q commands work while PowerShell has focus.
    """
    opencv_key = cv2.waitKeyEx(1)

    if opencv_key == 27:
        return "q"

    if opencv_key != -1:
        character_code = opencv_key & 0xFF

        if character_code:
            character = chr(character_code).lower()

            if character in {"b", "c", "n", "s", "q"}:
                return character

    if msvcrt is not None and msvcrt.kbhit():
        character = msvcrt.getwch()

        # Discard the second byte of Windows special-key sequences.
        if character in {"\x00", "\xe0"}:
            if msvcrt.kbhit():
                msvcrt.getwch()
            return None

        if character == "\x1b":
            return "q"

        character = character.lower()

        if character in {"b", "c", "n", "s", "q"}:
            return character

    return None


def parse_reader_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read physical d6 rolls from the Icy Dice camera tray."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained d6 center-crop model.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Initial number of d6 dice. Prompts when omitted.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum confidence required to accept every die.",
    )
    return parser.parse_args()


def parse_d6_count(text: str) -> int:
    normalized = text.strip().lower().replace(" ", "")

    if normalized.endswith("d6"):
        count_text = normalized[:-2]
        count = int(count_text) if count_text else 1
    else:
        count = int(normalized)

    if not 1 <= count <= 30:
        raise ValueError("The d6 count must be between 1 and 30.")

    return count


def prompt_for_d6_count(current: int | None = None) -> int:
    while True:
        if current is None:
            response = input("How many d6 dice? Examples: 4 or 4d6: ").strip()
        else:
            response = input(
                f"New d6 count [{current}] " "(Enter keeps current): "
            ).strip()

            if not response:
                return current

        try:
            return parse_d6_count(response)
        except (ValueError, TypeError):
            print("Enter a count such as 4 or an expression such as 4d6.")


def make_d6_request(count: int) -> RollRequest:
    return RollRequest(count=count, die_type="d6")


def load_d6_model(
    model_path: Path,
) -> tuple[
    nn.Module,
    transforms.Compose,
    tuple[str, ...],
    torch.device,
    dict[str, object],
]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Expected the center-crop model produced by "
            "train_d6_classifier.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            model_path,
            map_location=device,
        )

    class_names = tuple(
        str(value)
        for value in checkpoint.get(
            "class_names",
            ["1", "2", "3", "4", "5", "6"],
        )
    )

    if class_names != ("1", "2", "3", "4", "5", "6"):
        raise RuntimeError(
            "The selected model is not a six-class d6 model. "
            f"Classes found: {class_names}"
        )

    image_size = int(checkpoint.get("image_size", 160))
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
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    details = {
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "validation_accuracy": checkpoint.get("validation_accuracy"),
    }

    return (
        model,
        evaluation_transform,
        class_names,
        device,
        details,
    )


def center_crop_for_model(
    masked_crop: np.ndarray,
) -> np.ndarray:
    """
    Reproduce dataset_variants/d6_center preprocessing.

    The collector crop is 128 x 128. Retain the central 68%, enlarge it back
    to 128 x 128, then let the model transform resize it to its input size.
    """
    height, width = masked_crop.shape[:2]
    side = max(
        2,
        int(round(min(height, width) * CENTER_CROP_FRACTION)),
    )

    center_x = width // 2
    center_y = height // 2

    x1 = max(0, center_x - side // 2)
    y1 = max(0, center_y - side // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)

    # Keep the requested side length when rounding against an edge.
    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)

    crop = masked_crop[y1:y2, x1:x2]

    return cv2.resize(
        crop,
        (CROP_SIZE, CROP_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )


def classify_candidate_images(
    candidate_images: list[CandidateImages],
    model: nn.Module,
    evaluation_transform,
    class_names: tuple[str, ...],
    device: torch.device,
) -> list[DiePrediction]:
    if not candidate_images:
        return []

    center_crops = [center_crop_for_model(images.masked) for images in candidate_images]

    tensors = []

    for crop in center_crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        tensors.append(evaluation_transform(pil_image))

    batch = torch.stack(tensors).to(device)

    with torch.inference_mode():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    predictions: list[DiePrediction] = []

    for crop, probability_vector in zip(
        center_crops,
        probabilities,
        strict=True,
    ):
        class_index = int(np.argmax(probability_vector))
        predictions.append(
            DiePrediction(
                value=int(class_names[class_index]),
                confidence=float(probability_vector[class_index]),
                probabilities=tuple(float(value) for value in probability_vector),
                center_crop=crop,
            )
        )

    return predictions


def annotate_predictions(
    tray: np.ndarray,
    candidates: list[Candidate],
    predictions: list[DiePrediction],
    threshold: float,
    split_count: int,
) -> np.ndarray:
    output = tray.copy()
    all_accepted = all(prediction.confidence >= threshold for prediction in predictions)
    total = sum(prediction.value for prediction in predictions)

    for index, (candidate, prediction) in enumerate(
        zip(candidates, predictions, strict=True),
        start=1,
    ):
        x, y, width, height = candidate.bbox
        accepted = prediction.confidence >= threshold
        color = (0, 255, 0) if accepted else (0, 0, 255)

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            color,
            3,
        )

        label = f"#{index}: {prediction.value} " f"{prediction.confidence * 100:.0f}%"

        text_y = max(78, y - 10)

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

    values_text = " + ".join(str(prediction.value) for prediction in predictions)
    split_text = f" | auto-split {split_count}" if split_count else ""

    if all_accepted:
        status = f"ACCEPTED | {values_text} = {total}" f"{split_text}"
        status_color = (0, 255, 0)
    else:
        uncertain_count = sum(
            prediction.confidence < threshold for prediction in predictions
        )
        status = (
            f"UNCERTAIN ({uncertain_count}) | "
            f"provisional {values_text} = {total}"
            f"{split_text}"
        )
        status_color = (0, 0, 255)

    cv2.rectangle(
        output,
        (0, 0),
        (OUTPUT_WIDTH, 58),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        output,
        status,
        (18, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        status_color,
        2,
        cv2.LINE_AA,
    )

    return output


def build_prediction_sheet(
    predictions: list[DiePrediction],
    threshold: float,
) -> np.ndarray:
    tile_width = 190
    tile_height = 190
    columns = min(5, max(1, len(predictions)))
    rows = math.ceil(len(predictions) / columns)

    sheet = np.full(
        (rows * tile_height, columns * tile_width, 3),
        45,
        dtype=np.uint8,
    )

    for index, prediction in enumerate(predictions, start=1):
        row = (index - 1) // columns
        column = (index - 1) % columns

        x = column * tile_width + 31
        y = row * tile_height + 42

        sheet[
            y : y + CROP_SIZE,
            x : x + CROP_SIZE,
        ] = prediction.center_crop

        accepted = prediction.confidence >= threshold
        color = (0, 255, 0) if accepted else (0, 0, 255)

        cv2.putText(
            sheet,
            f"Die {index}: {prediction.value}",
            (column * tile_width + 12, row * tile_height + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            f"{prediction.confidence * 100:.1f}%",
            (column * tile_width + 54, row * tile_height + 184),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return sheet


def save_reader_debug(
    raw_frame: np.ndarray,
    tray: np.ndarray | None,
    mask: np.ndarray | None,
    annotated: np.ndarray | None,
    predictions: list[DiePrediction] | None,
    threshold: float,
) -> None:
    DEBUG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    cv2.imwrite(
        str(DEBUG_DIRECTORY / f"{timestamp}_camera.png"),
        raw_frame,
    )

    if tray is not None:
        cv2.imwrite(
            str(DEBUG_DIRECTORY / f"{timestamp}_tray.png"),
            tray,
        )

    if mask is not None:
        cv2.imwrite(
            str(DEBUG_DIRECTORY / f"{timestamp}_mask.png"),
            mask,
        )

    if annotated is not None:
        cv2.imwrite(
            str(DEBUG_DIRECTORY / f"{timestamp}_reader.png"),
            annotated,
        )

    if predictions:
        cv2.imwrite(
            str(DEBUG_DIRECTORY / f"{timestamp}_predictions.png"),
            build_prediction_sheet(predictions, threshold),
        )

        for index, prediction in enumerate(predictions, start=1):
            cv2.imwrite(
                str(
                    DEBUG_DIRECTORY / f"{timestamp}_die_{index:02d}_"
                    f"{prediction.value}_"
                    f"{prediction.confidence:.3f}.png"
                ),
                prediction.center_crop,
            )

    print(f"Debug images saved in: {DEBUG_DIRECTORY.resolve()}")


def main() -> int:
    args = parse_reader_args()

    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("--threshold must be greater than 0 and at most 1.")

    if args.count is None:
        count = prompt_for_d6_count()
    else:
        if not 1 <= args.count <= 30:
            raise ValueError("--count must be between 1 and 30.")
        count = args.count

    request = make_d6_request(count)

    (
        model,
        evaluation_transform,
        class_names,
        device,
        model_details,
    ) = load_d6_model(args.model)

    print("Icy Dice d6 reader")
    print("------------------")
    print(f"Model: {args.model.resolve()}")
    print(f"Device: {device}")
    print("Model validation accuracy: " f"{model_details['validation_accuracy']}")
    print(f"Acceptance threshold: {args.threshold:.2f}\n")

    # The old background remains available after a result so B can verify
    # that the dice were removed before recording the next background.
    background: np.ndarray | None = None
    background_ready = False
    result_active = False

    camera = open_camera()
    detector, dictionary, parameters = create_aruco_detector()

    raw_window = "Icy Dice - Camera"
    tray_window = "Icy Dice - Rectified Tray"
    reader_window = "Icy Dice - d6 Reader"
    mask_window = "Icy Dice - Foreground Mask"
    result_window = "Icy Dice - Predictions"

    for window in (
        raw_window,
        tray_window,
        reader_window,
        mask_window,
        result_window,
    ):
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(raw_window, 960, 540)
    cv2.resizeWindow(tray_window, 900, 600)
    cv2.resizeWindow(reader_window, 1000, 670)
    cv2.resizeWindow(mask_window, 900, 600)
    cv2.resizeWindow(result_window, 950, 500)

    print("Controls")
    print("  B  capture a fresh empty-tray background")
    print("  C  read the roll when READY")
    print("  N  choose a new number of d6 dice")
    print("  S  save diagnostic images")
    print("  Q  quit")
    print(
        f"\nRemove all dice, then press B. " f"The next roll is {request.expression}."
    )

    previous_motion_frame: np.ndarray | None = None
    stable_frames = 0

    last_frame: np.ndarray | None = None
    last_rectified: np.ndarray | None = None
    last_mask: np.ndarray | None = None
    last_component_labels: np.ndarray | None = None
    last_candidates: list[Candidate] = []
    last_split_count = 0
    last_live_annotated: np.ndarray | None = None

    result_annotated: np.ndarray | None = None
    result_sheet: np.ndarray | None = None
    result_predictions: list[DiePrediction] = []

    try:
        while True:
            ok, frame = camera.read()

            if not ok or frame is None:
                print("Camera frame read failed.")
                return 1

            last_frame = frame.copy()
            camera_display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, _rejected = detect_markers(
                gray,
                detector,
                dictionary,
                parameters,
            )
            marker_count = 0 if ids is None else len(ids)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(
                    camera_display,
                    corners,
                    ids,
                )

            if marker_count == 4:
                source_points = find_inward_marker_corners(corners)

                cv2.polylines(
                    camera_display,
                    [np.round(source_points).astype(np.int32)],
                    True,
                    (255, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

                last_rectified = rectify_tray(
                    frame,
                    source_points,
                )
                cv2.imshow(tray_window, last_rectified)

                if result_active and result_annotated is not None:
                    cv2.imshow(reader_window, result_annotated)

                    if result_sheet is not None:
                        cv2.imshow(result_window, result_sheet)

                    camera_status = (
                        "RESULT SHOWN | remove dice and press B "
                        f"| next {request.expression}"
                    )
                    camera_status_color = (0, 255, 255)

                elif background_ready and background is not None:
                    current_motion_frame = motion_frame(last_rectified)
                    motion_score = calculate_motion_score(
                        current_motion_frame,
                        previous_motion_frame,
                    )
                    previous_motion_frame = current_motion_frame

                    if motion_score < STABLE_MOTION_THRESHOLD:
                        stable_frames += 1
                    else:
                        stable_frames = 0

                    stable = stable_frames >= STABLE_FRAMES_REQUIRED

                    last_mask = foreground_mask(
                        last_rectified,
                        background,
                    )
                    (
                        last_candidates,
                        last_component_labels,
                        last_split_count,
                    ) = detect_die_candidates(
                        last_mask,
                        request.count,
                    )

                    last_live_annotated = annotate_candidates(
                        last_rectified,
                        last_candidates,
                        request,
                        stable,
                        motion_score,
                        split_count=last_split_count,
                    )

                    cv2.imshow(mask_window, last_mask)
                    cv2.imshow(reader_window, last_live_annotated)

                    camera_status = f"4 markers | target {request.expression}"
                    camera_status_color = (0, 255, 0)

                else:
                    previous_motion_frame = None
                    stable_frames = 0
                    last_mask = None
                    last_component_labels = None
                    last_candidates = []
                    last_split_count = 0

                    placeholder = placeholder_image(
                        "REMOVE DICE - press B for a fresh background"
                    )
                    cv2.imshow(mask_window, placeholder)
                    cv2.imshow(reader_window, placeholder)

                    camera_status = (
                        "4 markers | remove dice and press B "
                        f"| next {request.expression}"
                    )
                    camera_status_color = (0, 255, 255)

            else:
                previous_motion_frame = None
                stable_frames = 0
                camera_status = f"Markers detected: {marker_count}/4"
                camera_status_color = (0, 0, 255)

            cv2.putText(
                camera_display,
                camera_status,
                (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                camera_status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(raw_window, camera_display)

            command = poll_command()

            if command == "q":
                return 0

            if command == "b":
                if last_rectified is None or marker_count != 4:
                    print(
                        "Cannot capture background: " "all four markers are required."
                    )
                    continue

                # When an older background exists, prevent an accidental
                # background capture while dice remain in the tray.
                if background is not None:
                    removal_mask = foreground_mask(
                        last_rectified,
                        background,
                    )
                    remaining_regions = connected_regions(removal_mask)

                    if remaining_regions:
                        print(
                            "Background capture blocked: foreground objects "
                            "are still visible. Remove all dice first."
                        )
                        continue

                background = last_rectified.copy()
                background_ready = True
                result_active = False
                result_annotated = None
                result_sheet = None
                result_predictions = []

                last_mask = None
                last_component_labels = None
                last_candidates = []
                last_split_count = 0
                previous_motion_frame = None
                stable_frames = 0

                save_background(background)
                print(f"Fresh background captured. " f"Cue: roll {request.expression}.")

            if command == "n":
                print("\nClick the PowerShell window to enter " "the new d6 count.")
                count = prompt_for_d6_count(request.count)
                request = make_d6_request(count)

                background_ready = False
                result_active = False
                result_annotated = None
                result_sheet = None
                result_predictions = []

                last_mask = None
                last_component_labels = None
                last_candidates = []
                last_split_count = 0
                previous_motion_frame = None
                stable_frames = 0

                print("\nRemove all dice and press B for a fresh background.")
                print(f"Next roll: {request.expression}.")

            if command == "s":
                if last_frame is not None:
                    annotated = (
                        result_annotated if result_active else last_live_annotated
                    )
                    save_reader_debug(
                        raw_frame=last_frame,
                        tray=last_rectified,
                        mask=last_mask,
                        annotated=annotated,
                        predictions=(result_predictions if result_active else None),
                        threshold=args.threshold,
                    )

            if command == "c":
                if result_active:
                    print(
                        "A result is already displayed. "
                        "Remove the dice and press B for the next roll."
                    )
                    continue

                if not background_ready or background is None:
                    print(
                        "Read blocked: remove the dice and press B "
                        "for a fresh background first."
                    )
                    continue

                if marker_count != 4 or last_rectified is None:
                    print("Read blocked: all four markers must be visible.")
                    continue

                if len(last_candidates) != request.count:
                    print(
                        "Read blocked: "
                        f"expected {request.count} dice but detected "
                        f"{len(last_candidates)}."
                    )
                    continue

                if stable_frames < STABLE_FRAMES_REQUIRED:
                    print("Read blocked: the dice are not yet stable.")
                    continue

                if last_component_labels is None or last_mask is None:
                    print("Read blocked: candidate component data " "are unavailable.")
                    continue

                frozen_tray = last_rectified.copy()
                frozen_candidates = list(last_candidates)
                frozen_labels = last_component_labels.copy()

                candidate_images = [
                    extract_candidate_images(
                        frozen_tray,
                        frozen_labels,
                        candidate,
                    )
                    for candidate in frozen_candidates
                ]

                predictions = classify_candidate_images(
                    candidate_images,
                    model,
                    evaluation_transform,
                    class_names,
                    device,
                )

                result_annotated = annotate_predictions(
                    frozen_tray,
                    frozen_candidates,
                    predictions,
                    threshold=args.threshold,
                    split_count=last_split_count,
                )
                result_sheet = build_prediction_sheet(
                    predictions,
                    args.threshold,
                )
                result_predictions = predictions

                cv2.imshow(reader_window, result_annotated)
                cv2.imshow(result_window, result_sheet)
                cv2.waitKey(1)

                values = [prediction.value for prediction in predictions]
                confidences = [prediction.confidence for prediction in predictions]
                total = sum(values)
                accepted = all(
                    confidence >= args.threshold for confidence in confidences
                )

                print("\nRoll read:")
                for index, prediction in enumerate(
                    predictions,
                    start=1,
                ):
                    marker = (
                        "accepted"
                        if prediction.confidence >= args.threshold
                        else "UNCERTAIN"
                    )
                    print(
                        f"  Die {index}: {prediction.value} "
                        f"({prediction.confidence:.1%}, {marker})"
                    )

                print(f"  Values: {values}")
                print(f"  Total: {total}")

                if accepted:
                    print("  Result accepted.")
                else:
                    print(
                        "  Result is provisional because at least one "
                        "die is below the confidence threshold."
                    )

                print("\nRemove all dice and press B before the next roll.")

                result_active = True
                background_ready = False
                previous_motion_frame = None
                stable_frames = 0

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
