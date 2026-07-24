from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import (
    CameraCalibrationReport,
    CameraCalibrator,
    CameraHealth,
    CameraReference,
)
from .config import DieProfile
from .feedback import FeedbackStore
from .models import ModelEnsemble
from .recognition import BurstRecognizer, RecognitionResult
from .workflow import RollWorkflow, ReviewOutcome
from . import vision


@dataclass(frozen=True)
class LiveAnalysis:
    marker_count: int
    rectified: np.ndarray | None
    mask: np.ndarray | None
    candidates: list[vision.Candidate]
    split_count: int


class ReaderController:
    """
    Facade intended for both the current OpenCV UI and a future GUI.

    It owns the model ensemble and workflow but does not create windows,
    read keys, or prompt for text.
    """

    def __init__(
        self,
        profile: DieProfile,
        count: int,
    ) -> None:
        self.profile = profile
        self.ensemble = ModelEnsemble(profile)
        self.recognizer = BurstRecognizer(
            profile,
            self.ensemble,
        )
        self.feedback_store = FeedbackStore(
            profile,
            self.ensemble,
        )
        self.workflow = RollWorkflow(
            profile,
            count,
            self.recognizer,
            self.feedback_store,
        )
        (
            self.detector,
            self.dictionary,
            self.parameters,
        ) = vision.create_aruco_detector()
        self.camera_calibrator = CameraCalibrator()
        self.camera_calibration_report: CameraCalibrationReport | None = None
        self.background_camera_reference: CameraReference | None = None

    @property
    def result(self) -> RecognitionResult | None:
        return self.workflow.result

    def analyze_frame(
        self,
        frame: np.ndarray,
    ) -> LiveAnalysis:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )
        corners, ids, _ = vision.detect_markers(
            gray,
            self.detector,
            self.dictionary,
            self.parameters,
        )
        marker_count = 0 if ids is None else len(ids)

        if marker_count != 4:
            return LiveAnalysis(
                marker_count=marker_count,
                rectified=None,
                mask=None,
                candidates=[],
                split_count=0,
            )

        source_points = vision.find_inward_marker_corners(corners)
        rectified = vision.rectify_tray(frame, source_points)

        if self.workflow.background is None:
            return LiveAnalysis(
                marker_count=4,
                rectified=rectified,
                mask=None,
                candidates=[],
                split_count=0,
            )

        mask = vision.foreground_mask(
            rectified,
            self.workflow.background,
        )
        candidates, _, split_count = vision.detect_die_candidates(
            mask,
            self.workflow.count,
        )
        return LiveAnalysis(
            marker_count=4,
            rectified=rectified,
            mask=mask,
            candidates=candidates,
            split_count=split_count,
        )

    def calibrate_camera(
        self,
        camera: cv2.VideoCapture,
        *,
        settle_seconds: float = 2.5,
        verification_seconds: float = 1.0,
        lock_controls: bool = True,
    ) -> CameraCalibrationReport:
        self.camera_calibration_report = self.camera_calibrator.calibrate(
            camera,
            settle_seconds=settle_seconds,
            verification_seconds=verification_seconds,
            lock_controls=lock_controls,
        )
        self.background_camera_reference = None
        return self.camera_calibration_report

    def set_background(
        self,
        rectified: np.ndarray,
        *,
        camera: cv2.VideoCapture | None = None,
        source_frame: np.ndarray | None = None,
    ) -> None:
        self.workflow.set_background(rectified)
        vision.save_background(rectified)
        if camera is not None and source_frame is not None:
            self.background_camera_reference = (
                self.camera_calibrator.make_reference(
                    camera,
                    source_frame,
                )
            )
        else:
            self.background_camera_reference = None

    def check_camera(
        self,
        camera: cv2.VideoCapture,
        source_frame: np.ndarray,
    ) -> CameraHealth | None:
        if self.background_camera_reference is None:
            return None
        return self.camera_calibrator.check_reference(
            camera,
            source_frame,
            self.background_camera_reference,
        )

    def camera_session_metadata(
        self,
        camera: cv2.VideoCapture,
        source_frame: np.ndarray,
    ) -> dict[str, object]:
        return self.camera_calibrator.session_metadata(
            camera,
            source_frame,
            self.background_camera_reference,
        )

    def capture_selection(
        self,
        camera,
        window_name: str,
    ) -> vision.BurstSelection:
        if self.workflow.background is None:
            raise RuntimeError("A fresh background is required.")

        selection, failure = vision.capture_ranked_burst(
            camera=camera,
            detector=self.detector,
            dictionary=self.dictionary,
            parameters=self.parameters,
            background=self.workflow.background,
            request=self.workflow.request,
            burst_window=window_name,
        )
        if selection is None:
            raise RuntimeError(failure)
        return selection

    def recognize_initial(
        self,
        selection: vision.BurstSelection,
    ) -> RecognitionResult:
        return self.workflow.recognize_initial(selection)

    def review(
        self,
        wrong_indices: set[int],
        true_labels: dict[int, str],
        *,
        camera: cv2.VideoCapture | None = None,
        source_frame: np.ndarray | None = None,
    ) -> ReviewOutcome:
        if camera is not None and source_frame is not None:
            self.feedback_store.set_context(
                self.camera_session_metadata(
                    camera,
                    source_frame,
                )
            )
        else:
            self.feedback_store.set_context(None)
        return self.workflow.review(wrong_indices, true_labels)

    def recognize_retry(
        self,
        selection: vision.BurstSelection,
    ) -> RecognitionResult:
        return self.workflow.recognize_retry(selection)
