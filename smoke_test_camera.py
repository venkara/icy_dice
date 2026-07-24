from __future__ import annotations

import sys

import cv2
import numpy as np

from icy_dice.camera import (
    CameraCalibrator,
    capture_snapshot,
    lock_camera_controls,
    measure_frame,
    summarize_metrics,
)


class FakeCamera:
    """Minimal VideoCapture substitute for a camera-module smoke test."""

    def __init__(self) -> None:
        self.properties = {
            cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
            cv2.CAP_PROP_FPS: 60.0,
            cv2.CAP_PROP_EXPOSURE: -5.0,
            cv2.CAP_PROP_AUTO_EXPOSURE: 0.75,
            cv2.CAP_PROP_GAIN: 0.0,
            cv2.CAP_PROP_WB_TEMPERATURE: 4600.0,
            cv2.CAP_PROP_AUTO_WB: 1.0,
            cv2.CAP_PROP_FOCUS: 35.0,
            cv2.CAP_PROP_AUTOFOCUS: 1.0,
        }
        self.counter = 0

    def getBackendName(self) -> str:
        return "FAKE_DSHOW"

    def get(self, property_id: int) -> float:
        return self.properties.get(property_id, 0.0)

    def set(self, property_id: int, value: float) -> bool:
        self.properties[property_id] = float(value)
        return True

    def read(self):
        self.counter += 1
        value = 120 + (self.counter % 3) - 1
        frame = np.full((240, 320, 3), value, dtype=np.uint8)
        cv2.putText(
            frame,
            "ICY DICE",
            (60, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return True, frame


def main() -> int:
    camera = FakeCamera()
    snapshot = capture_snapshot(camera, probe_writable=True)
    assert snapshot.backend == "FAKE_DSHOW"
    assert snapshot.value("exposure") == -5.0

    actions = lock_camera_controls(camera)
    assert any(
        action.control == "auto_exposure" and action.succeeded
        for action in actions
    )
    assert camera.get(cv2.CAP_PROP_AUTO_WB) == 0.0
    assert camera.get(cv2.CAP_PROP_AUTOFOCUS) == 0.0

    _, frame = camera.read()
    metrics = measure_frame(frame)
    summary = summarize_metrics([metrics, metrics])
    assert summary.frame_count == 2
    assert metrics.sharpness > 0

    calibrator = CameraCalibrator()
    report = calibrator.calibrate(
        camera,
        settle_seconds=0.01,
        verification_seconds=0.01,
        save=False,
    )
    assert report.locked_metrics.frame_count >= 12
    assert report.locked.value("autofocus") == 0.0

    _, reference_frame = camera.read()
    reference = calibrator.make_reference(camera, reference_frame)
    health = calibrator.check_reference(camera, reference_frame, reference)
    assert health.healthy
    metadata = calibrator.session_metadata(
        camera,
        reference_frame,
        reference,
    )
    assert "calibration_report" in metadata
    assert "background_reference" in metadata

    print("Camera module synthetic smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
