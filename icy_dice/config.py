from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    crop_fraction: float


@dataclass(frozen=True)
class DieProfile:
    die_type: str
    class_names: tuple[str, ...]
    semantic_values: dict[str, int]
    models: tuple[ModelSpec, ...]
    model_confidence_threshold: float
    frame_vote_threshold: float
    require_model_agreement: bool = True
    max_count: int = 30

    @property
    def display_name(self) -> str:
        return self.die_type

    @property
    def feedback_directory(self) -> Path:
        return Path("dataset_feedback") / self.die_type

    @property
    def debug_directory(self) -> Path:
        return Path("captures") / f"{self.die_type}_reader"

    def value_for_label(self, label: str) -> int:
        return self.semantic_values[label]

    def label_for_value(self, value: int) -> str:
        matches = [
            label
            for label, semantic_value in self.semantic_values.items()
            if semantic_value == value
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{value} is not a unique semantic value for {self.die_type}."
            )
        return matches[0]

    @property
    def allowed_values(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.semantic_values.values())))


PROFILES: dict[str, DieProfile] = {
    "d6": DieProfile(
        die_type="d6",
        class_names=("1", "2", "3", "4", "5", "6"),
        semantic_values={str(value): value for value in range(1, 7)},
        models=(
            ModelSpec(
                name="d6 center 68%",
                path=Path("models/d6_center/d6_mobilenet_v3_small.pt"),
                # The older d6 checkpoint predates crop metadata.
                crop_fraction=0.68,
            ),
        ),
        model_confidence_threshold=0.75,
        frame_vote_threshold=0.75,
        require_model_agreement=False,
    ),
    "d8": DieProfile(
        die_type="d8",
        class_names=tuple(str(value) for value in range(1, 9)),
        semantic_values={str(value): value for value in range(1, 9)},
        models=(
            ModelSpec(
                name="d8 center 55%",
                path=Path("models/d8_center55/d8_mobilenet_v3_small.pt"),
                crop_fraction=0.55,
            ),
            ModelSpec(
                name="d8 center 60%",
                path=Path("models/d8_center60/d8_mobilenet_v3_small.pt"),
                crop_fraction=0.60,
            ),
        ),
        model_confidence_threshold=0.70,
        frame_vote_threshold=0.75,
        require_model_agreement=True,
    ),
    "d10": DieProfile(
        die_type="d10",
        class_names=tuple(str(value) for value in range(10)),
        # Printed 0 is the semantic value 10 in an ordinary d10 roll.
        semantic_values={
            **{str(value): value for value in range(1, 10)},
            "0": 10,
        },
        # Fill these after the crop sweep identifies the best d10 model(s).
        models=(),
        model_confidence_threshold=0.70,
        frame_vote_threshold=0.75,
        require_model_agreement=True,
    ),
}


def normalize_die_type(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "")
    if not normalized.startswith("d"):
        normalized = f"d{normalized}"
    return normalized


def get_profile(die_type: str) -> DieProfile:
    normalized = normalize_die_type(die_type)
    try:
        return PROFILES[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(PROFILES))
        raise ValueError(
            f"Unsupported die type {die_type!r}. Configured types: {supported}."
        ) from error
