from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CAMERA_INDEX = 0
WIDTH = 1920
HEIGHT = 1080
FPS = 60

CAPTURE_DIRECTORY = Path("captures")


def fourcc_string(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * position)) & 0xFF) for position in range(4))


def open_camera() -> cv2.VideoCapture:
    # DirectShow worked during the original test, so try it first.
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()

        print("DirectShow failed; trying Microsoft Media Foundation...")
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open camera 0.\n"
            "Close Camera Hub, Zoom, Teams, browsers, or other programs "
            "that may be using the camera."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)

    return camera


def save_frame(frame: np.ndarray) -> Path:
    CAPTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = CAPTURE_DIRECTORY / f"frame_{timestamp}.png"

    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not save {path}")

    return path


def main() -> None:
    camera = open_camera()

    # Discard startup frames while exposure and white balance stabilize.
    frame: np.ndarray | None = None

    for _ in range(30):
        ok, candidate = camera.read()

        if ok and candidate is not None:
            frame = candidate

    if frame is None:
        camera.release()
        raise RuntimeError("Camera opened, but no frames were received.")

    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)
    fourcc = fourcc_string(camera.get(cv2.CAP_PROP_FOURCC))

    try:
        backend = camera.getBackendName()
    except cv2.error:
        backend = "unknown"

    print(f"Backend:    {backend}")
    print(f"Resolution: {width} × {height}")
    print(f"Requested:  {FPS} fps")
    print(f"Reported:   {fps:.1f} fps")
    print(f"Pixel mode: {fourcc!r}")
    print()
    print("S: save the current raw frame")
    print("Q or Esc: quit")

    window_name = "Icy Dice Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    try:
        while True:
            ok, frame = camera.read()

            if not ok or frame is None:
                print("Frame read failed.")
                break

            # Preserve the unmodified image for saving and later processing.
            display_frame = frame.copy()

            cv2.putText(
                display_frame,
                f"{frame.shape[1]}x{frame.shape[0]}  {backend}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("s"), ord("S")):
                path = save_frame(frame)
                print(f"Saved: {path.resolve()}")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
