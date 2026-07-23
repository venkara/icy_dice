from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .config import DieProfile
from .models import ModelEnsemble, center_crop_fraction
from . import vision


MAX_CANDIDATE_MATCH_DISTANCE = 70.0
MAX_LOCKED_DIE_MOVEMENT = 45.0
MIN_AGGREGATION_FRAMES = 8


@dataclass(frozen=True)
class ModelAggregate:
    model_name: str
    label: str
    value: int
    confidence: float
    vote_fraction: float
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class DiePrediction:
    label: str
    value: int
    accepted: bool
    combined_confidence: float
    model_results: tuple[ModelAggregate, ...]
    display_crop: np.ndarray
    user_confirmed: bool = False
    flagged_wrong: bool = False
    true_label: str | None = None
    true_value: int | None = None


@dataclass(frozen=True)
class RecognitionResult:
    predictions: list[DiePrediction]
    representative_tray: np.ndarray
    representative_mask: np.ndarray
    representative_candidates: list[vision.Candidate]
    representative_component_labels: np.ndarray
    representative_split_count: int
    frames_used: int
    selected_frames: int
    reviewed: bool = False

    @property
    def all_accepted(self) -> bool:
        return all(prediction.accepted for prediction in self.predictions)

    @property
    def total(self) -> int:
        return sum(prediction.value for prediction in self.predictions)


def candidate_match_order(
    reference: list[vision.Candidate],
    current: list[vision.Candidate],
    maximum_distance: float = MAX_CANDIDATE_MATCH_DISTANCE,
) -> list[int] | None:
    if len(reference) != len(current):
        return None
    if not reference:
        return []

    pairs: list[tuple[float, int, int]] = []
    for reference_index, reference_candidate in enumerate(reference):
        reference_point = np.asarray(
            reference_candidate.centroid,
            dtype=np.float32,
        )
        for current_index, current_candidate in enumerate(current):
            current_point = np.asarray(
                current_candidate.centroid,
                dtype=np.float32,
            )
            pairs.append(
                (
                    float(np.linalg.norm(reference_point - current_point)),
                    reference_index,
                    current_index,
                )
            )

    pairs.sort(key=lambda item: item[0])
    assigned_reference: set[int] = set()
    assigned_current: set[int] = set()
    mapping = [-1] * len(reference)
    largest_distance = 0.0

    for distance, reference_index, current_index in pairs:
        if reference_index in assigned_reference:
            continue
        if current_index in assigned_current:
            continue
        mapping[reference_index] = current_index
        assigned_reference.add(reference_index)
        assigned_current.add(current_index)
        largest_distance = max(largest_distance, distance)
        if len(assigned_reference) == len(reference):
            break

    if any(index < 0 for index in mapping):
        return None
    if largest_distance > maximum_distance:
        return None
    return mapping


def _weighted_average(
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    normalized = weights / weights.sum()
    return np.tensordot(normalized, probabilities, axes=(0, 0))


def _aggregate_model(
    model_name: str,
    probabilities: np.ndarray,
    weights: np.ndarray,
    profile: DieProfile,
) -> list[ModelAggregate]:
    averaged = _weighted_average(probabilities, weights)
    frame_votes = np.argmax(probabilities, axis=2)
    aggregates: list[ModelAggregate] = []

    for die_index in range(averaged.shape[0]):
        class_index = int(np.argmax(averaged[die_index]))
        label = profile.class_names[class_index]
        aggregates.append(
            ModelAggregate(
                model_name=model_name,
                label=label,
                value=profile.value_for_label(label),
                confidence=float(averaged[die_index, class_index]),
                vote_fraction=float(
                    np.mean(frame_votes[:, die_index] == class_index)
                ),
                probabilities=tuple(
                    float(value) for value in averaged[die_index]
                ),
            )
        )
    return aggregates


class BurstRecognizer:
    def __init__(
        self,
        profile: DieProfile,
        ensemble: ModelEnsemble,
    ) -> None:
        self.profile = profile
        self.ensemble = ensemble

    def recognize(
        self,
        selection: vision.BurstSelection,
        background: np.ndarray,
        request: vision.RollRequest,
    ) -> RecognitionResult:
        representative = selection.representative
        representative_mask = vision.foreground_mask(
            representative.rectified,
            background,
        )
        (
            reference_candidates,
            reference_labels,
            representative_split_count,
        ) = vision.detect_die_candidates(
            representative_mask,
            request.count,
        )
        if len(reference_candidates) != request.count:
            raise RuntimeError(
                "The representative frame no longer contains "
                f"{request.count} detectable dice."
            )

        frame_crops: list[list[np.ndarray]] = []
        frame_weights: list[float] = []

        for frame in selection.selected_frames:
            mask = vision.foreground_mask(frame.rectified, background)
            candidates, component_labels, _ = vision.detect_die_candidates(
                mask,
                request.count,
            )
            mapping = candidate_match_order(
                reference_candidates,
                candidates,
            )
            if mapping is None:
                continue

            ordered_crops: list[np.ndarray] = []
            for current_index in mapping:
                images = vision.extract_candidate_images(
                    frame.rectified,
                    component_labels,
                    candidates[current_index],
                )
                ordered_crops.append(images.masked)
            frame_crops.append(ordered_crops)
            frame_weights.append(max(0.05, float(frame.quality_score)))

        if len(frame_crops) < MIN_AGGREGATION_FRAMES:
            raise RuntimeError(
                "Too few burst frames could be matched consistently: "
                f"{len(frame_crops)} available, "
                f"{MIN_AGGREGATION_FRAMES} required."
            )

        frame_count = len(frame_crops)
        die_count = request.count
        flat_crops = [
            crop
            for crops in frame_crops
            for crop in crops
        ]
        probability_sets = self.ensemble.classify(flat_crops)
        weights = np.asarray(frame_weights, dtype=np.float64)

        aggregate_sets: list[list[ModelAggregate]] = []
        for model_name, probabilities in probability_sets.items():
            reshaped = probabilities.reshape(
                frame_count,
                die_count,
                len(self.profile.class_names),
            )
            aggregate_sets.append(
                _aggregate_model(
                    model_name,
                    reshaped,
                    weights,
                    self.profile,
                )
            )

        representative_images = [
            vision.extract_candidate_images(
                representative.rectified,
                reference_labels,
                candidate,
            )
            for candidate in reference_candidates
        ]
        predictions: list[DiePrediction] = []

        for die_index in range(die_count):
            model_results = tuple(
                aggregate_set[die_index]
                for aggregate_set in aggregate_sets
            )
            labels = {result.label for result in model_results}
            agreement = (
                len(labels) == 1
                if self.profile.require_model_agreement
                else True
            )
            thresholds_ok = all(
                result.confidence
                >= self.profile.model_confidence_threshold
                and result.vote_fraction
                >= self.profile.frame_vote_threshold
                for result in model_results
            )
            accepted = agreement and thresholds_ok

            combined = np.mean(
                np.asarray(
                    [result.probabilities for result in model_results],
                    dtype=np.float32,
                ),
                axis=0,
            )
            combined_index = int(np.argmax(combined))
            combined_label = self.profile.class_names[combined_index]

            if len(labels) == 1:
                label = model_results[0].label
            else:
                label = combined_label

            predictions.append(
                DiePrediction(
                    label=label,
                    value=self.profile.value_for_label(label),
                    accepted=accepted,
                    combined_confidence=float(combined[combined_index]),
                    model_results=model_results,
                    display_crop=center_crop_fraction(
                        representative_images[die_index].masked,
                        self.ensemble.bundles[0].crop_fraction,
                    ),
                )
            )

        return RecognitionResult(
            predictions=predictions,
            representative_tray=representative.rectified.copy(),
            representative_mask=representative_mask,
            representative_candidates=reference_candidates,
            representative_component_labels=reference_labels,
            representative_split_count=representative_split_count,
            frames_used=frame_count,
            selected_frames=len(selection.selected_frames),
        )


def _locked_mapping(
    previous: RecognitionResult,
    current: RecognitionResult,
) -> dict[int, int]:
    locked_slots = [
        index
        for index, prediction in enumerate(previous.predictions)
        if prediction.accepted
    ]
    if not locked_slots:
        return {}

    pairs: list[tuple[float, int, int]] = []
    for slot_index in locked_slots:
        old_point = np.asarray(
            previous.representative_candidates[slot_index].centroid,
            dtype=np.float32,
        )
        for current_index, candidate in enumerate(
            current.representative_candidates
        ):
            new_point = np.asarray(candidate.centroid, dtype=np.float32)
            pairs.append(
                (
                    float(np.linalg.norm(old_point - new_point)),
                    slot_index,
                    current_index,
                )
            )

    pairs.sort(key=lambda item: item[0])
    assigned_slots: set[int] = set()
    assigned_current: set[int] = set()
    mapping: dict[int, int] = {}

    for distance, slot_index, current_index in pairs:
        if distance > MAX_LOCKED_DIE_MOVEMENT:
            continue
        if slot_index in assigned_slots or current_index in assigned_current:
            continue
        mapping[slot_index] = current_index
        assigned_slots.add(slot_index)
        assigned_current.add(current_index)
        if len(mapping) == len(locked_slots):
            break

    missing = [
        slot_index + 1
        for slot_index in locked_slots
        if slot_index not in mapping
    ]
    if missing:
        raise RuntimeError(
            "Could not find previously accepted dice at their old "
            f"positions: {', '.join(map(str, missing))}. "
            "Keep green dice still and move only red dice."
        )
    return mapping


def merge_retry(
    previous: RecognitionResult,
    current: RecognitionResult,
) -> RecognitionResult:
    if len(previous.predictions) != len(current.predictions):
        raise RuntimeError(
            "The retry contains a different number of dice."
        )

    locked_mapping = _locked_mapping(previous, current)
    uncertain_slots = [
        index
        for index, prediction in enumerate(previous.predictions)
        if not prediction.accepted
    ]
    used_current = set(locked_mapping.values())
    retry_current_indices = [
        index
        for index in range(len(current.predictions))
        if index not in used_current
    ]
    if len(retry_current_indices) != len(uncertain_slots):
        raise RuntimeError(
            "Could not separate repositioned dice from locked dice."
        )

    merged_predictions: list[DiePrediction | None] = [
        None
    ] * len(previous.predictions)
    merged_candidates: list[vision.Candidate | None] = [
        None
    ] * len(previous.predictions)

    for slot_index, current_index in locked_mapping.items():
        merged_predictions[slot_index] = previous.predictions[slot_index]
        merged_candidates[slot_index] = (
            current.representative_candidates[current_index]
        )

    for slot_index, current_index in zip(
        uncertain_slots,
        retry_current_indices,
        strict=True,
    ):
        merged_predictions[slot_index] = current.predictions[current_index]
        merged_candidates[slot_index] = (
            current.representative_candidates[current_index]
        )

    return RecognitionResult(
        predictions=[
            prediction
            for prediction in merged_predictions
            if prediction is not None
        ],
        representative_tray=current.representative_tray,
        representative_mask=current.representative_mask,
        representative_candidates=[
            candidate
            for candidate in merged_candidates
            if candidate is not None
        ],
        representative_component_labels=(
            current.representative_component_labels
        ),
        representative_split_count=current.representative_split_count,
        frames_used=current.frames_used,
        selected_frames=current.selected_frames,
        reviewed=False,
    )
