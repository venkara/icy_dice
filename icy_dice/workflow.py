from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import numpy as np

from .config import DieProfile
from .feedback import FeedbackStore, review_result
from .recognition import (
    BurstRecognizer,
    RecognitionResult,
    merge_retry,
)
from . import vision


class WorkflowState(Enum):
    NEED_BACKGROUND = auto()
    READY_TO_CAPTURE = auto()
    REVIEW_AVAILABLE = auto()
    RETRY_READY = auto()
    COMPLETE = auto()


@dataclass(frozen=True)
class ReviewOutcome:
    result: RecognitionResult
    saved_examples: tuple[Path, ...]


class RollWorkflow:
    """
    UI-independent state machine for one declared roll.

    OpenCV, Tkinter, a web UI, or tests can all drive these same methods.
    """

    def __init__(
        self,
        profile: DieProfile,
        count: int,
        recognizer: BurstRecognizer,
        feedback_store: FeedbackStore,
    ) -> None:
        if not 1 <= count <= profile.max_count:
            raise ValueError(
                f"{profile.die_type} count must be between 1 "
                f"and {profile.max_count}."
            )

        self.profile = profile
        self.count = count
        self.recognizer = recognizer
        self.feedback_store = feedback_store
        self.background: np.ndarray | None = None
        self.result: RecognitionResult | None = None
        self.state = WorkflowState.NEED_BACKGROUND
        self.feedback_session_id: str | None = None
        self.attempt = 0

    @property
    def request(self) -> vision.RollRequest:
        return vision.RollRequest(
            count=self.count,
            die_type=self.profile.die_type,
        )

    def set_count(self, count: int) -> None:
        if not 1 <= count <= self.profile.max_count:
            raise ValueError(
                f"Count must be between 1 and {self.profile.max_count}."
            )
        self.count = count
        self.reset()

    def reset(self) -> None:
        self.background = None
        self.result = None
        self.state = WorkflowState.NEED_BACKGROUND
        self.feedback_session_id = None
        self.attempt = 0

    def set_background(self, image: np.ndarray) -> None:
        self.background = image.copy()
        self.result = None
        self.state = WorkflowState.READY_TO_CAPTURE
        self.feedback_session_id = None
        self.attempt = 0

    def recognize_initial(
        self,
        selection: vision.BurstSelection,
    ) -> RecognitionResult:
        if self.background is None:
            raise RuntimeError("A fresh background is required.")
        self.result = self.recognizer.recognize(
            selection,
            self.background,
            self.request,
        )
        self.attempt += 1
        self.feedback_session_id = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        self.state = (
            WorkflowState.COMPLETE
            if self.result.all_accepted
            else WorkflowState.REVIEW_AVAILABLE
        )
        return self.result

    def review(
        self,
        wrong_indices: set[int],
        true_labels: dict[int, str],
    ) -> ReviewOutcome:
        if self.result is None:
            raise RuntimeError("No result is available for review.")
        if self.feedback_session_id is None:
            self.feedback_session_id = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

        reviewed, saved = review_result(
            result=self.result,
            wrong_indices=wrong_indices,
            true_labels=true_labels,
            feedback_store=self.feedback_store,
            session_id=self.feedback_session_id,
            attempt=max(1, self.attempt),
        )
        self.result = reviewed
        self.state = (
            WorkflowState.COMPLETE
            if reviewed.all_accepted
            else WorkflowState.RETRY_READY
        )
        return ReviewOutcome(
            result=reviewed,
            saved_examples=tuple(saved),
        )

    def recognize_retry(
        self,
        selection: vision.BurstSelection,
    ) -> RecognitionResult:
        if self.background is None:
            raise RuntimeError("The original background is missing.")
        if self.result is None:
            raise RuntimeError("There is no result to retry.")
        if self.state is not WorkflowState.RETRY_READY:
            raise RuntimeError(
                "Review the result and flag actual errors before retrying."
            )

        current = self.recognizer.recognize(
            selection,
            self.background,
            self.request,
        )
        self.result = merge_retry(self.result, current)
        self.attempt += 1
        self.state = (
            WorkflowState.COMPLETE
            if self.result.all_accepted
            else WorkflowState.REVIEW_AVAILABLE
        )
        return self.result
