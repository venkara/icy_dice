from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2

from .config import DieProfile
from .models import ModelEnsemble
from .recognition import DiePrediction, RecognitionResult
from . import vision


class FeedbackStore:
    def __init__(
        self,
        profile: DieProfile,
        ensemble: ModelEnsemble,
        root: Path | None = None,
    ) -> None:
        self.profile = profile
        self.ensemble = ensemble
        self.root = root or profile.feedback_directory

    def _masked_crop(
        self,
        result: RecognitionResult,
        die_index: int,
    ):
        return vision.extract_candidate_images(
            result.representative_tray,
            result.representative_component_labels,
            result.representative_candidates[die_index],
        ).masked

    def save_example(
        self,
        result: RecognitionResult,
        die_index: int,
        true_label: str,
        feedback_type: str,
        session_id: str,
        attempt: int,
    ) -> Path:
        prediction = result.predictions[die_index]
        class_directory = self.root / true_label
        class_directory.mkdir(parents=True, exist_ok=True)

        serial = attempt * 100 + die_index + 1
        stem = f"{session_id}_{serial:04d}"
        image_path = class_directory / f"{stem}.png"
        metadata_path = class_directory / f"{stem}.json"

        crop = self._masked_crop(result, die_index)
        if not cv2.imwrite(str(image_path), crop):
            raise RuntimeError(
                f"Could not save feedback image: {image_path}"
            )

        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "die_type": self.profile.die_type,
            "session_id": session_id,
            "attempt": attempt,
            "die_number": die_index + 1,
            "feedback_type": feedback_type,
            "true_label": true_label,
            "true_value": self.profile.value_for_label(true_label),
            "displayed_label": prediction.label,
            "displayed_value": prediction.value,
            "accepted_before_review": prediction.accepted,
            "combined_confidence": prediction.combined_confidence,
            "models": [
                {
                    "name": model_result.model_name,
                    "label": model_result.label,
                    "value": model_result.value,
                    "confidence": model_result.confidence,
                    "vote_fraction": model_result.vote_fraction,
                    "probabilities": list(model_result.probabilities),
                }
                for model_result in prediction.model_results
            ],
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return image_path


def review_result(
    result: RecognitionResult,
    wrong_indices: set[int],
    true_labels: dict[int, str],
    feedback_store: FeedbackStore,
    session_id: str,
    attempt: int,
) -> tuple[RecognitionResult, list[Path]]:
    if result.reviewed:
        raise RuntimeError("This capture has already been reviewed.")

    invalid = sorted(
        index + 1
        for index in wrong_indices
        if not 0 <= index < len(result.predictions)
    )
    if invalid:
        raise ValueError(
            "Die numbers out of range: "
            + ", ".join(map(str, invalid))
        )

    missing_truth = sorted(
        index + 1
        for index in wrong_indices
        if index not in true_labels
    )
    if missing_truth:
        raise ValueError(
            "Missing true values for dice: "
            + ", ".join(map(str, missing_truth))
        )

    revised: list[DiePrediction] = []
    saved: list[Path] = []

    for die_index, prediction in enumerate(result.predictions):
        if die_index in wrong_indices:
            true_label = true_labels[die_index]
            if true_label not in feedback_store.profile.class_names:
                raise ValueError(
                    f"Invalid true label {true_label!r} for "
                    f"{feedback_store.profile.die_type}."
                )
            if true_label == prediction.label:
                raise ValueError(
                    f"Die {die_index + 1}: true label matches the "
                    "displayed label."
                )

            saved.append(
                feedback_store.save_example(
                    result=result,
                    die_index=die_index,
                    true_label=true_label,
                    feedback_type="user_flagged_misclassification",
                    session_id=session_id,
                    attempt=attempt,
                )
            )
            revised.append(
                replace(
                    prediction,
                    accepted=False,
                    user_confirmed=False,
                    flagged_wrong=True,
                    true_label=true_label,
                    true_value=feedback_store.profile.value_for_label(
                        true_label
                    ),
                )
            )
            continue

        if not prediction.accepted:
            saved.append(
                feedback_store.save_example(
                    result=result,
                    die_index=die_index,
                    true_label=prediction.label,
                    feedback_type=(
                        "user_confirmed_low_confidence_correct"
                    ),
                    session_id=session_id,
                    attempt=attempt,
                )
            )
            revised.append(
                replace(
                    prediction,
                    accepted=True,
                    user_confirmed=True,
                    flagged_wrong=False,
                    true_label=prediction.label,
                    true_value=prediction.value,
                )
            )
            continue

        revised.append(prediction)

    return (
        replace(
            result,
            predictions=revised,
            reviewed=True,
        ),
        saved,
    )
