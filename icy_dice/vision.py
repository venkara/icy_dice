"""Proven camera, ArUco, segmentation, crop, and burst-capture backend.

This module contains no roll workflow or user interaction.  It was extracted
from dataset collector v4.2 so CLI and future GUI front ends share exactly
the same image-processing implementation.
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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

# Pressing C starts a burst. Motion is used to rank frames, not to block C.
BURST_DURATION_SECONDS = 1.0
BURST_MIN_CAPTURE_FRAMES = 30
BURST_MAX_CAPTURE_FRAMES = 90
BURST_SELECTED_FRAMES = 18
BURST_MIN_VALID_FRAMES = 18

# Frame-ranking weights. They sum to 1.0.
BURST_WEIGHT_SHARPNESS = 0.35
BURST_WEIGHT_MOTION = 0.20
BURST_WEIGHT_CENTROID = 0.25
BURST_WEIGHT_AREA = 0.10
BURST_WEIGHT_MARKERS = 0.10

MOTION_IMAGE_WIDTH = 300


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


@dataclass
class BurstFrame:
    capture_index: int
    rectified: np.ndarray
    source_points: np.ndarray
    candidates: list[Candidate]
    split_count: int
    motion_score: float
    sharpness_score: float
    centroid_error: float = 0.0
    area_error: float = 0.0
    marker_error: float = 0.0
    quality_score: float = 0.0


@dataclass(frozen=True)
class BurstSelection:
    representative: BurstFrame
    selected_frames: list[BurstFrame]
    total_frames: int
    marker_valid_frames: int
    count_valid_frames: int




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
        CANDIDATE_CENTER_MARGIN
        <= center_x
        < image_width - CANDIDATE_CENTER_MARGIN
    ):
        return False

    if not (
        CANDIDATE_CENTER_MARGIN
        <= center_y
        < image_height - CANDIDATE_CENTER_MARGIN
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

        coordinates = np.column_stack(
            np.where(labels == component_label)
        ).astype(np.int32)

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
    split_count: int = 0,
    state_override: str | None = None,
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

    split_text = (
        f" | auto-split {split_count}"
        if split_count
        else ""
    )

    if state_override is not None:
        state_text = state_override
        status_color = (0, 255, 255)
    elif count_matches:
        state_text = "DICE STOPPED? PRESS C FOR BURST"
        status_color = (0, 255, 0)
    else:
        state_text = "COUNT MISMATCH - C MAY STILL RETRY"
        status_color = (0, 165, 255)

    status_line = (
        f"Target {request.expression} | "
        f"detected {len(candidates)}/{request.count}"
        f"{split_text} | "
        f"{state_text}"
    )

    return add_status_banner(
        output,
        status_line,
        status_color,
        height=58,
    )

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


def candidate_sharpness(
    tray: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Variance of the Laplacian inside foreground regions.

    Higher values indicate a sharper frame. Shadows are included, but because
    every burst frame contains the same dice, the metric remains useful for
    ranking frames within that burst.
    """
    gray = cv2.cvtColor(tray, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)

    pixels = laplacian[mask > 0]

    if pixels.size < 100:
        roi = gray[
            MOTION_MARGIN_Y : gray.shape[0] - MOTION_MARGIN_Y,
            MOTION_MARGIN_X : gray.shape[1] - MOTION_MARGIN_X,
        ]
        return float(cv2.Laplacian(roi, cv2.CV_32F).var())

    return float(pixels.var())


def normalized_rank(
    values: list[float],
    higher_is_better: bool,
) -> np.ndarray:
    """
    Convert a metric into ranks from 0 (worst) through 1 (best).
    """
    count = len(values)

    if count == 0:
        return np.empty(0, dtype=np.float32)

    if count == 1:
        return np.ones(1, dtype=np.float32)

    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)

    if not np.any(finite):
        return np.full(count, 0.5, dtype=np.float32)

    finite_values = array[finite]
    replacement = (
        float(np.max(finite_values)) + 1.0
        if not higher_is_better
        else float(np.min(finite_values)) - 1.0
    )
    array = np.where(finite, array, replacement)

    order = np.argsort(array)
    ranks = np.empty(count, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, count)

    if not higher_is_better:
        ranks = 1.0 - ranks

    return ranks.astype(np.float32)


def score_burst_frames(
    frames: list[BurstFrame],
) -> None:
    """
    Score valid frames using aggregate geometry from the entire burst.

    Candidate order is deterministic (reading order), so corresponding
    centroids and areas can be compared across frames.
    """
    if not frames:
        return

    centroid_stack = np.asarray(
        [
            [candidate.centroid for candidate in frame.candidates]
            for frame in frames
        ],
        dtype=np.float32,
    )
    area_stack = np.asarray(
        [
            [candidate.area for candidate in frame.candidates]
            for frame in frames
        ],
        dtype=np.float32,
    )
    marker_stack = np.asarray(
        [frame.source_points for frame in frames],
        dtype=np.float32,
    )

    median_centroids = np.median(centroid_stack, axis=0)
    median_areas = np.median(area_stack, axis=0)
    median_markers = np.median(marker_stack, axis=0)

    for frame_index, frame in enumerate(frames):
        frame.centroid_error = float(
            np.mean(
                np.linalg.norm(
                    centroid_stack[frame_index] - median_centroids,
                    axis=1,
                )
            )
        )
        frame.area_error = float(
            np.mean(
                np.abs(
                    np.log(
                        (area_stack[frame_index] + 1.0)
                        / (median_areas + 1.0)
                    )
                )
            )
        )
        frame.marker_error = float(
            np.mean(
                np.linalg.norm(
                    marker_stack[frame_index] - median_markers,
                    axis=1,
                )
            )
        )

    sharpness_rank = normalized_rank(
        [frame.sharpness_score for frame in frames],
        higher_is_better=True,
    )
    motion_rank = normalized_rank(
        [frame.motion_score for frame in frames],
        higher_is_better=False,
    )
    centroid_rank = normalized_rank(
        [frame.centroid_error for frame in frames],
        higher_is_better=False,
    )
    area_rank = normalized_rank(
        [frame.area_error for frame in frames],
        higher_is_better=False,
    )
    marker_rank = normalized_rank(
        [frame.marker_error for frame in frames],
        higher_is_better=False,
    )

    for index, frame in enumerate(frames):
        frame.quality_score = float(
            BURST_WEIGHT_SHARPNESS * sharpness_rank[index]
            + BURST_WEIGHT_MOTION * motion_rank[index]
            + BURST_WEIGHT_CENTROID * centroid_rank[index]
            + BURST_WEIGHT_AREA * area_rank[index]
            + BURST_WEIGHT_MARKERS * marker_rank[index]
        )


def burst_progress_image(
    tray: np.ndarray | None,
    request: RollRequest,
    elapsed: float,
    captured: int,
    valid: int,
) -> np.ndarray:
    if tray is None:
        output = np.zeros(
            (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3),
            dtype=np.uint8,
        )
    else:
        output = tray.copy()

    return add_status_banner(
        output,
        (
            f"CAPTURING {request.expression} BURST | "
            f"{elapsed:.2f}/{BURST_DURATION_SECONDS:.2f}s | "
            f"frames {captured} | valid {valid}"
        ),
        (0, 255, 255),
        height=70,
    )


def capture_ranked_burst(
    camera: cv2.VideoCapture,
    detector,
    dictionary,
    parameters,
    background: np.ndarray,
    request: RollRequest,
    burst_window: str,
) -> tuple[BurstSelection | None, str]:
    """
    Capture about one second, retain count-valid frames, and rank them.

    C means "the dice have stopped." Motion no longer blocks the request.
    Instead, low-motion frames naturally rise to the top of the burst ranking.
    """
    start_time = time.perf_counter()
    total_frames = 0
    marker_valid_frames = 0
    previous_motion: np.ndarray | None = None
    valid_frames: list[BurstFrame] = []
    latest_rectified: np.ndarray | None = None

    while total_frames < BURST_MAX_CAPTURE_FRAMES:
        elapsed = time.perf_counter() - start_time

        if (
            elapsed >= BURST_DURATION_SECONDS
            and total_frames >= BURST_MIN_CAPTURE_FRAMES
        ):
            break

        ok, frame = camera.read()

        if not ok or frame is None:
            continue

        total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = detect_markers(
            gray,
            detector,
            dictionary,
            parameters,
        )

        marker_count = 0 if ids is None else len(ids)

        if marker_count == 4:
            marker_valid_frames += 1
            source_points = find_inward_marker_corners(corners)
            rectified = rectify_tray(frame, source_points)
            latest_rectified = rectified

            current_motion = motion_frame(rectified)
            motion_score = calculate_motion_score(
                current_motion,
                previous_motion,
            )
            previous_motion = current_motion

            mask = foreground_mask(rectified, background)
            candidates, _component_labels, split_count = (
                detect_die_candidates(
                    mask,
                    request.count,
                )
            )

            if len(candidates) == request.count:
                valid_frames.append(
                    BurstFrame(
                        capture_index=total_frames,
                        rectified=rectified,
                        source_points=source_points.copy(),
                        candidates=list(candidates),
                        split_count=split_count,
                        motion_score=motion_score,
                        sharpness_score=candidate_sharpness(
                            rectified,
                            mask,
                        ),
                    )
                )

        progress = burst_progress_image(
            latest_rectified,
            request,
            elapsed=time.perf_counter() - start_time,
            captured=total_frames,
            valid=len(valid_frames),
        )
        cv2.imshow(burst_window, progress)

        # Keep the OpenCV windows responsive. Escape aborts the burst.
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return None, "Burst cancelled."

    if len(valid_frames) < BURST_MIN_VALID_FRAMES:
        return (
            None,
            (
                f"Burst failed: captured {total_frames} frames, "
                f"{marker_valid_frames} had all markers, and "
                f"{len(valid_frames)} had the expected "
                f"{request.count} dice. Need at least "
                f"{BURST_MIN_VALID_FRAMES} count-valid frames."
            ),
        )

    score_burst_frames(valid_frames)

    selected_count = min(
        BURST_SELECTED_FRAMES,
        len(valid_frames),
    )
    selected = sorted(
        valid_frames,
        key=lambda frame: frame.quality_score,
        reverse=True,
    )[:selected_count]

    # The representative image is the highest-ranked member of the selected
    # group. The ranking itself was computed using aggregate geometry from all
    # valid frames, so it reflects the burst consensus rather than one metric.
    representative = selected[0]

    return (
        BurstSelection(
            representative=representative,
            selected_frames=selected,
            total_frames=total_frames,
            marker_valid_frames=marker_valid_frames,
            count_valid_frames=len(valid_frames),
        ),
        "",
    )


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


def draw_banner_text(
    image: np.ndarray,
    text: str,
    color: tuple[int, int, int],
    banner_height: int,
) -> None:
    baseline_y = min(
        banner_height - 14,
        max(32, int(round(banner_height * 0.67))),
    )

    cv2.putText(
        image,
        text,
        (18, baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        color,
        2,
        cv2.LINE_AA,
    )


def add_status_banner(
    image: np.ndarray,
    text: str,
    color: tuple[int, int, int],
    height: int = 58,
) -> np.ndarray:
    """
    Add a black status strip above the image.

    No source pixels are covered or replaced; the returned image is taller by
    exactly `height` pixels.
    """
    if height < 1:
        raise ValueError("Status-banner height must be positive.")

    output = cv2.copyMakeBorder(
        image,
        height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    draw_banner_text(
        output,
        text,
        color,
        banner_height=height,
    )

    return output


def replace_status_banner(
    image: np.ndarray,
    text: str,
    color: tuple[int, int, int],
    default_height: int = 58,
) -> np.ndarray:
    """
    Replace an existing tray-image banner when present.

    This prevents the saved-result screen from gaining another strip each time
    its status changes. Non-padded images simply receive a new banner.
    """
    existing_height = 0

    if (
        image.shape[1] == OUTPUT_WIDTH
        and image.shape[0] > OUTPUT_HEIGHT
    ):
        existing_height = image.shape[0] - OUTPUT_HEIGHT

    if existing_height <= 0:
        return add_status_banner(
            image,
            text,
            color,
            height=default_height,
        )

    output = image.copy()
    output[:existing_height, :] = (0, 0, 0)

    draw_banner_text(
        output,
        text,
        color,
        banner_height=existing_height,
    )

    return output


def close_window_if_open(window_name: str) -> None:
    try:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
    except cv2.error:
        pass
