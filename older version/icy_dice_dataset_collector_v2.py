from __future__ import annotations

import json
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

ROI_MARGIN_X = 45
ROI_MARGIN_Y = 45

FOREGROUND_THRESHOLD = 28
MIN_DIE_AREA = 500
MAX_DIE_AREA = 40_000

CROP_SIZE = 128
CROP_PADDING_FRACTION = 0.25
MASK_BACKGROUND_BGR = (128, 128, 128)

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

    roi = np.zeros_like(mask)
    roi[
        ROI_MARGIN_Y : mask.shape[0] - ROI_MARGIN_Y,
        ROI_MARGIN_X : mask.shape[1] - ROI_MARGIN_X,
    ] = 255
    mask = cv2.bitwise_and(mask, roi)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

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

    return mask


def detect_die_candidates(mask: np.ndarray) -> tuple[list[Candidate], np.ndarray]:
    count, component_labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    candidates: list[Candidate] = []

    for component_label in range(1, count):
        x = int(stats[component_label, cv2.CC_STAT_LEFT])
        y = int(stats[component_label, cv2.CC_STAT_TOP])
        width = int(stats[component_label, cv2.CC_STAT_WIDTH])
        height = int(stats[component_label, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_label, cv2.CC_STAT_AREA])

        if area < MIN_DIE_AREA or area > MAX_DIE_AREA:
            continue

        centroid_x, centroid_y = centroids[component_label]

        candidates.append(
            Candidate(
                component_label=component_label,
                bbox=(x, y, width, height),
                area=area,
                centroid=(float(centroid_x), float(centroid_y)),
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


def annotate_candidates(
    tray: np.ndarray,
    candidates: list[Candidate],
    request: RollRequest,
    stable: bool,
    motion_score: float,
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
    status_line = (
        f"Target {request.expression} | "
        f"detected {len(candidates)}/{request.count} | "
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
        ROI_MARGIN_Y : tray.shape[0] - ROI_MARGIN_Y,
        ROI_MARGIN_X : tray.shape[1] - ROI_MARGIN_X,
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
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    _x, _y, width, height = candidate.bbox
    center_x, center_y = candidate.centroid

    side = max(width, height)
    padding = max(8, int(round(side * CROP_PADDING_FRACTION)))
    side += 2 * padding

    x1 = int(round(center_x - side / 2))
    y1 = int(round(center_y - side / 2))
    x2 = x1 + side
    y2 = y1 + side

    image_height, image_width = image_shape[:2]

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > image_width:
        x1 -= x2 - image_width
        x2 = image_width
    if y2 > image_height:
        y1 -= y2 - image_height
        y2 = image_height

    return max(0, x1), max(0, y1), x2, y2


def extract_candidate_images(
    tray: np.ndarray,
    component_labels: np.ndarray,
    candidate: Candidate,
) -> CandidateImages:
    x1, y1, x2, y2 = square_bounds(candidate, tray.shape)
    raw = tray[y1:y2, x1:x2].copy()

    component_mask = np.where(
        component_labels[y1:y2, x1:x2] == candidate.component_label,
        255,
        0,
    ).astype(np.uint8)

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expanded_mask = cv2.dilate(component_mask, dilation_kernel, iterations=1)

    masked = np.full_like(raw, MASK_BACKGROUND_BGR)
    masked[expanded_mask > 0] = raw[expanded_mask > 0]

    raw = cv2.resize(raw, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
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


def main() -> int:
    print("Icy Dice dataset collector")
    print("--------------------------")
    print("When an OpenCV window asks for input, click the PowerShell")
    print("window before typing the roll or visible markings.\n")

    request = prompt_for_roll_request()

    # A fresh background is required before every roll. An older saved image
    # remains on disk for inspection, but this run will not use it automatically.
    background: np.ndarray | None = None
    background_ready = False

    print("\nPreparation: remove all dice from the tray.")
    print("Press B after all four markers are visible.")
    print("Do not roll until the program confirms the fresh background.")

    camera = open_camera()
    detector, dictionary, parameters = create_aruco_detector()

    raw_window = "Icy Dice - Camera"
    tray_window = "Icy Dice - Rectified Tray"
    candidate_window = "Icy Dice - Dataset Collector"
    mask_window = "Icy Dice - Foreground Mask"
    verify_window = "Icy Dice - Verify Capture"
    crop_window = "Icy Dice - Candidate Crops"

    for window in (
        raw_window,
        tray_window,
        candidate_window,
        mask_window,
        verify_window,
        crop_window,
    ):
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(raw_window, 960, 540)
    cv2.resizeWindow(tray_window, 900, 600)
    cv2.resizeWindow(candidate_window, 900, 600)
    cv2.resizeWindow(mask_window, 900, 600)
    cv2.resizeWindow(verify_window, 900, 600)
    cv2.resizeWindow(crop_window, 850, 500)

    print("\nControls")
    print("  B  capture fresh empty-tray background for the next roll")
    print("  C  capture and label a ready roll")
    print("  N  choose a new number/type of dice")
    print("  S  save debug images")
    print("  Q  quit")
    print(f"\nNext roll will be {request.expression}, after background capture.")

    previous_motion_frame: np.ndarray | None = None
    stable_frames = 0

    last_rectified: np.ndarray | None = None
    last_mask: np.ndarray | None = None
    last_component_labels: np.ndarray | None = None
    last_candidates: list[Candidate] = []
    last_annotated: np.ndarray | None = None

    try:
        while True:
            ok, frame = camera.read()

            if not ok or frame is None:
                print("Camera frame read failed.")
                return 1

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
                cv2.aruco.drawDetectedMarkers(camera_display, corners, ids)

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

                last_rectified = rectify_tray(frame, source_points)
                cv2.imshow(tray_window, last_rectified)

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

                if background_ready and background is not None:
                    last_mask = foreground_mask(last_rectified, background)
                    last_candidates, last_component_labels = detect_die_candidates(
                        last_mask
                    )
                    last_annotated = annotate_candidates(
                        last_rectified,
                        last_candidates,
                        request,
                        stable,
                        motion_score,
                    )
                    cv2.imshow(mask_window, last_mask)
                    cv2.imshow(candidate_window, last_annotated)
                else:
                    last_mask = None
                    last_component_labels = None
                    last_candidates = []
                    last_annotated = placeholder_image(
                        "REMOVE DICE - press B for a fresh background"
                    )
                    cv2.imshow(mask_window, last_annotated)
                    cv2.imshow(candidate_window, last_annotated)

                if background_ready:
                    camera_status = f"4 markers | target {request.expression}"
                else:
                    camera_status = (
                        f"4 markers | remove dice and press B | "
                        f"next {request.expression}"
                    )
                camera_status_color = (0, 255, 0)
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
                0.85,
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
                    print("Cannot capture background: four markers are required.")
                else:
                    background = last_rectified.copy()
                    background_ready = True
                    save_background(background)

                    last_mask = None
                    last_component_labels = None
                    last_candidates = []
                    previous_motion_frame = None
                    stable_frames = 0

                    print("Fresh background captured.")
                    print(f"Cue: roll {request.expression}.")

            if command == "n":
                print("\nClick the PowerShell window to enter a new roll.")
                request = prompt_for_roll_request(request)

                background = None
                background_ready = False
                last_mask = None
                last_component_labels = None
                last_candidates = []

                print("\nRemove all dice from the tray.")
                print("Press B to capture a fresh background for the new roll.")
                print(f"Next roll: {request.expression}.")

                previous_motion_frame = None
                stable_frames = 0

            if command == "s":
                save_debug_images(
                    frame,
                    last_rectified,
                    last_mask,
                    last_annotated,
                )

            if command == "c":
                if not background_ready or background is None:
                    print(
                        "Capture blocked: remove the dice and press B "
                        "for a fresh background first."
                    )
                    continue
                if marker_count != 4 or last_rectified is None:
                    print("Capture blocked: all four markers must be visible.")
                    continue
                if len(last_candidates) != request.count:
                    print(
                        "Capture blocked: "
                        f"expected {request.count} candidates but detected "
                        f"{len(last_candidates)}."
                    )
                    continue
                if stable_frames < STABLE_FRAMES_REQUIRED:
                    print(
                        "Capture blocked: dice are not yet considered stable "
                        f"(threshold {STABLE_MOTION_THRESHOLD})."
                    )
                    continue
                if last_component_labels is None or last_mask is None:
                    print("Capture blocked: component data are unavailable.")
                    continue

                frozen_tray = last_rectified.copy()
                frozen_mask = last_mask.copy()
                frozen_component_labels = last_component_labels.copy()
                frozen_candidates = list(last_candidates)

                frozen_annotated = annotate_candidates(
                    frozen_tray,
                    frozen_candidates,
                    request,
                    stable=True,
                    motion_score=0.0,
                )

                candidate_images = [
                    extract_candidate_images(
                        frozen_tray,
                        frozen_component_labels,
                        candidate,
                    )
                    for candidate in frozen_candidates
                ]

                cv2.imshow(verify_window, frozen_annotated)
                cv2.imshow(crop_window, build_contact_sheet(candidate_images))
                cv2.waitKey(1)

                print(
                    "\nCapture frozen. Click the PowerShell window "
                    "to verify the markings."
                )
                labels = prompt_for_labels(request)

                if labels is None:
                    continue

                session_path = save_capture(
                    request=request,
                    labels=labels,
                    tray=frozen_tray,
                    foreground=frozen_mask,
                    candidates=frozen_candidates,
                    annotated=frozen_annotated,
                    candidate_images=candidate_images,
                )

                print(f"\nSaved session: {session_path.resolve()}")
                print(f"Saved labels: {', '.join(labels)}")

                # Every accepted roll consumes its background reference.
                background = None
                background_ready = False
                last_mask = None
                last_component_labels = None
                last_candidates = []

                print("\nRemove all dice from the tray.")
                print(
                    "Press B to capture a fresh background before "
                    f"rolling {request.expression} again."
                )
                print("Press N instead to choose a different roll.")

                previous_motion_frame = None
                stable_frames = 0

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
