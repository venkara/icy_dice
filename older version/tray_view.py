from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 60

# Normalized tray image. The tray is approximately 3:2.
OUTPUT_WIDTH = 1200
OUTPUT_HEIGHT = 800

CAPTURE_DIRECTORY = Path("captures")

BACKGROUND_PATH = Path("calibration/empty_tray.png")

# Ignore the tray walls and corner remnants.
ROI_MARGIN_X = 45
ROI_MARGIN_Y = 45

FOREGROUND_THRESHOLD = 28
MIN_DIE_AREA = 500


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

    # Improves corner stability once a marker has been found.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # Newer OpenCV interface.
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters,
        )
        return detector, None, None

    # Compatibility with older OpenCV versions.
    return None, dictionary, parameters


def detect_markers(
    gray: np.ndarray,
    detector,
    dictionary,
    parameters,
):
    if detector is not None:
        return detector.detectMarkers(gray)

    return cv2.aruco.detectMarkers(
        gray,
        dictionary,
        parameters=parameters,
    )


def order_points(points: np.ndarray) -> np.ndarray:
    """
    Return four points in this order:

        top-left, top-right, bottom-right, bottom-left
    """
    points = np.asarray(points, dtype=np.float32)

    ordered = np.zeros((4, 2), dtype=np.float32)

    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).ravel()

    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[2] = points[np.argmax(coordinate_sum)]
    ordered[1] = points[np.argmin(coordinate_difference)]
    ordered[3] = points[np.argmax(coordinate_difference)]

    return ordered


def find_inward_marker_corners(
    marker_corners: list[np.ndarray],
) -> np.ndarray:
    """
    For each marker, select the marker corner nearest the center
    of the four-marker arrangement.

    This avoids requiring a particular marker ID at each corner.
    """
    centers = np.array(
        [corners.reshape(4, 2).mean(axis=0) for corners in marker_corners],
        dtype=np.float32,
    )

    arrangement_center = centers.mean(axis=0)

    inward_corners: list[np.ndarray] = []

    for corners in marker_corners:
        points = corners.reshape(4, 2)

        distances = np.linalg.norm(
            points - arrangement_center,
            axis=1,
        )

        inward_corners.append(points[np.argmin(distances)])

    return order_points(np.array(inward_corners))


def rectify_tray(
    frame: np.ndarray,
    source_points: np.ndarray,
) -> np.ndarray:
    destination_points = np.array(
        [
            [0, 0],
            [OUTPUT_WIDTH - 1, 0],
            [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
            [0, OUTPUT_HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(
        source_points,
        destination_points,
    )

    return cv2.warpPerspective(
        frame,
        transform,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )


def save_images(
    raw_frame: np.ndarray,
    rectified_frame: np.ndarray,
) -> None:
    CAPTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    raw_path = CAPTURE_DIRECTORY / f"{timestamp}_raw.png"
    tray_path = CAPTURE_DIRECTORY / f"{timestamp}_tray.png"

    cv2.imwrite(str(raw_path), raw_frame)
    cv2.imwrite(str(tray_path), rectified_frame)

    print(f"Saved: {raw_path.resolve()}")
    print(f"Saved: {tray_path.resolve()}")


def main() -> None:
    background = load_background()
    camera = open_camera()

    detector, dictionary, parameters = create_aruco_detector()

    raw_window = "Icy Dice — Camera"
    tray_window = "Icy Dice — Rectified Tray"
    mask_window = "Icy Dice — Foreground Mask"
    candidate_window = "Icy Dice — Die Candidates"

    cv2.namedWindow(raw_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(tray_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(candidate_window, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(raw_window, 960, 540)
    cv2.resizeWindow(tray_window, 900, 600)
    cv2.resizeWindow(mask_window, 900, 600)
    cv2.resizeWindow(candidate_window, 900, 600)

    last_rectified: np.ndarray | None = None
    last_reported_ids: tuple[int, ...] | None = None

    print("B: capture the empty tray as background")
    print("S: save raw and rectified images")
    print("Q or Escape: quit")

    try:
        while True:
            ok, frame = camera.read()

            if not ok or frame is None:
                print("Camera frame read failed.")
                break

            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = detect_markers(
                gray,
                detector,
                dictionary,
                parameters,
            )

            marker_count = 0 if ids is None else len(ids)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(
                    display,
                    corners,
                    ids,
                )

                detected_ids = tuple(sorted(int(value) for value in ids.flatten()))

                if detected_ids != last_reported_ids:
                    print(f"Detected marker IDs: {detected_ids}")
                    last_reported_ids = detected_ids

            if marker_count == 4:
                source_points = find_inward_marker_corners(corners)

                # Show the four selected inward-facing corners.
                for index, point in enumerate(source_points):
                    x, y = np.round(point).astype(int)

                    cv2.circle(
                        display,
                        (x, y),
                        9,
                        (0, 0, 255),
                        -1,
                    )

                    cv2.putText(
                        display,
                        str(index),
                        (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                polygon = np.round(source_points).astype(np.int32)

                cv2.polylines(
                    display,
                    [polygon],
                    True,
                    (255, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

                last_rectified = rectify_tray(
                    frame,
                    source_points,
                )

                cv2.imshow(
                    tray_window,
                    last_rectified,
                )

                status = "4 markers — tray rectified"
                status_color = (0, 255, 0)

            else:
                status = f"Markers detected: {marker_count}/4"
                status_color = (0, 0, 255)

            cv2.putText(
                display,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(raw_window, display)

            # Update the foreground and candidate windows every frame,
            # as long as we have both a rectified tray and a background.
            if last_rectified is not None and background is not None:
                mask = foreground_mask(
                    last_rectified,
                    background,
                )

                candidate_image, boxes = find_die_candidates(
                    last_rectified,
                    mask,
                )

                cv2.putText(
                    candidate_image,
                    f"Candidates: {len(boxes)}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(mask_window, mask)
                cv2.imshow(candidate_window, candidate_image)

            elif background is None:
                placeholder = np.zeros(
                    (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3),
                    dtype=np.uint8,
                )

                cv2.putText(
                    placeholder,
                    "Press B with the tray empty",
                    (50, OUTPUT_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )

                cv2.imshow(mask_window, placeholder)
                cv2.imshow(candidate_window, placeholder)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("s"), ord("S")):
                if last_rectified is None:
                    print(
                        "Cannot save a rectified image: "
                        "four markers are not visible."
                    )
                else:
                    save_images(
                        frame,
                        last_rectified,
                    )

            if key in (ord("b"), ord("B")):
                if last_rectified is None:
                    print("Cannot capture background: " "four markers must be visible.")
                else:
                    background = last_rectified.copy()
                    save_background(background)
                    print("Background is now active.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


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


def foreground_mask(
    tray: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """
    Compare the current tray with the saved empty tray in Lab space.
    """
    if tray.shape != background.shape:
        raise ValueError("Current tray and background have different dimensions.")

    tray_lab = cv2.cvtColor(tray, cv2.COLOR_BGR2LAB)
    background_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)

    difference = cv2.absdiff(tray_lab, background_lab)

    # Euclidean-like color difference across L, a, and b.
    difference_float = difference.astype(np.float32)
    magnitude = np.sqrt(np.sum(difference_float * difference_float, axis=2))

    mask = np.where(
        magnitude >= FOREGROUND_THRESHOLD,
        255,
        0,
    ).astype(np.uint8)

    # Exclude walls and marker remnants.
    roi = np.zeros_like(mask)
    roi[
        ROI_MARGIN_Y : mask.shape[0] - ROI_MARGIN_Y,
        ROI_MARGIN_X : mask.shape[1] - ROI_MARGIN_X,
    ] = 255

    mask = cv2.bitwise_and(mask, roi)

    # Remove tiny texture differences, then fill small gaps in dice.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11),
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

    return mask


def find_die_candidates(
    tray: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Draw boxes around sufficiently large foreground components.
    """
    output = tray.copy()
    candidate_boxes: list[tuple[int, int, int, int]] = []

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < MIN_DIE_AREA:
            continue

        candidate_boxes.append((x, y, width, height))

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            3,
        )

        cv2.putText(
            output,
            f"{area}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output, candidate_boxes


if __name__ == "__main__":
    main()
