"""Camera capability probing, calibration, locking, and drift detection.

The camera driver is allowed to decline any property.  This module records
what it can actually read and set instead of assuming that every OpenCV
``CAP_PROP_*`` value is implemented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Iterable

import cv2
import numpy as np


CAMERA_REPORT_PATH = Path("calibration/camera_calibration.json")
CAMERA_PREVIEW_PATH = Path("calibration/camera_calibration_preview.png")

# Diagnostic thresholds.  These are deliberately broad: the goal is to catch
# a changed lighting/camera state, not to declare one artistic exposure ideal.
DARK_MEAN_LUMA = 45.0
BRIGHT_MEAN_LUMA = 210.0
MAX_SHADOW_FRACTION = 0.25
MAX_HIGHLIGHT_FRACTION = 0.03
MIN_ABSOLUTE_SHARPNESS = 60.0
MAX_SETTLED_LUMA_STD = 2.5
MAX_SETTLED_LUMA_RANGE = 8.0
MAX_SETTLED_CHANNEL_RANGE = 5.0

# Drift relative to the empty-tray background reference.
MAX_REFERENCE_LUMA_DELTA = 8.0
MAX_REFERENCE_LUMA_FRACTION = 0.10
MAX_REFERENCE_CHANNEL_DELTA = 8.0
MIN_REFERENCE_SHARPNESS_RATIO = 0.60
MAX_PROPERTY_DELTAS = {
    "exposure": 0.10,
    "gain": 0.50,
    "white_balance_temperature": 75.0,
    "focus": 1.0,
    "auto_exposure": 0.10,
    "auto_white_balance": 0.10,
    "autofocus": 0.10,
}


@dataclass(frozen=True)
class PropertySpec:
    name: str
    property_id: int
    writable_probe: bool = False


PROPERTY_SPECS: tuple[PropertySpec, ...] = (
    PropertySpec("frame_width", cv2.CAP_PROP_FRAME_WIDTH),
    PropertySpec("frame_height", cv2.CAP_PROP_FRAME_HEIGHT),
    PropertySpec("fps", cv2.CAP_PROP_FPS),
    PropertySpec("fourcc", cv2.CAP_PROP_FOURCC),
    PropertySpec("format", cv2.CAP_PROP_FORMAT),
    PropertySpec("mode", cv2.CAP_PROP_MODE),
    PropertySpec("brightness", cv2.CAP_PROP_BRIGHTNESS, True),
    PropertySpec("contrast", cv2.CAP_PROP_CONTRAST, True),
    PropertySpec("saturation", cv2.CAP_PROP_SATURATION, True),
    PropertySpec("hue", cv2.CAP_PROP_HUE, True),
    PropertySpec("gain", cv2.CAP_PROP_GAIN, True),
    PropertySpec("exposure", cv2.CAP_PROP_EXPOSURE, True),
    PropertySpec("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, True),
    PropertySpec(
        "white_balance_temperature",
        cv2.CAP_PROP_WB_TEMPERATURE,
        True,
    ),
    PropertySpec("auto_white_balance", cv2.CAP_PROP_AUTO_WB, True),
    PropertySpec("focus", cv2.CAP_PROP_FOCUS, True),
    PropertySpec("autofocus", cv2.CAP_PROP_AUTOFOCUS, True),
    PropertySpec("sharpness", cv2.CAP_PROP_SHARPNESS, True),
    PropertySpec("gamma", cv2.CAP_PROP_GAMMA, True),
    PropertySpec("backlight", cv2.CAP_PROP_BACKLIGHT, True),
    PropertySpec("zoom", cv2.CAP_PROP_ZOOM, True),
    PropertySpec("pan", cv2.CAP_PROP_PAN, True),
    PropertySpec("tilt", cv2.CAP_PROP_TILT, True),
    PropertySpec("iris", cv2.CAP_PROP_IRIS, True),
)


@dataclass(frozen=True)
class CameraProperty:
    name: str
    property_id: int
    value: float | None
    readable: bool
    settable: bool | None


@dataclass(frozen=True)
class CameraSnapshot:
    captured_at: str
    backend: str
    properties: dict[str, CameraProperty]

    def value(self, name: str) -> float | None:
        item = self.properties.get(name)
        return None if item is None else item.value

    def to_dict(self) -> dict[str, object]:
        return {
            "captured_at": self.captured_at,
            "backend": self.backend,
            "properties": {
                name: asdict(value)
                for name, value in self.properties.items()
            },
        }


@dataclass(frozen=True)
class FrameMetrics:
    mean_luminance: float
    luminance_std: float
    shadow_fraction: float
    highlight_fraction: float
    sharpness: float
    mean_blue: float
    mean_green: float
    mean_red: float

    @property
    def channel_means(self) -> tuple[float, float, float]:
        return (self.mean_blue, self.mean_green, self.mean_red)


@dataclass(frozen=True)
class StabilityMetrics:
    frame_count: int
    mean_luminance: float
    luminance_std_across_frames: float
    luminance_range: float
    mean_shadow_fraction: float
    mean_highlight_fraction: float
    median_sharpness: float
    minimum_sharpness: float
    channel_ranges: tuple[float, float, float]


@dataclass(frozen=True)
class ControlAction:
    control: str
    attempted: bool
    succeeded: bool
    before: float | None
    requested: float | None
    after: float | None
    note: str = ""


@dataclass(frozen=True)
class CameraCalibrationReport:
    created_at: str
    before_settle: CameraSnapshot
    settled: CameraSnapshot
    locked: CameraSnapshot
    settle_metrics: StabilityMetrics
    locked_metrics: StabilityMetrics
    actions: tuple[ControlAction, ...]
    warnings: tuple[str, ...]
    preview_metrics: FrameMetrics | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "before_settle": self.before_settle.to_dict(),
            "settled": self.settled.to_dict(),
            "locked": self.locked.to_dict(),
            "settle_metrics": asdict(self.settle_metrics),
            "locked_metrics": asdict(self.locked_metrics),
            "actions": [asdict(action) for action in self.actions],
            "warnings": list(self.warnings),
            "preview_metrics": (
                None if self.preview_metrics is None else asdict(self.preview_metrics)
            ),
        }


@dataclass(frozen=True)
class CameraReference:
    created_at: str
    snapshot: CameraSnapshot
    frame_metrics: FrameMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "snapshot": self.snapshot.to_dict(),
            "frame_metrics": asdict(self.frame_metrics),
        }


@dataclass(frozen=True)
class CameraHealth:
    checked_at: str
    current_snapshot: CameraSnapshot
    current_metrics: FrameMetrics
    warnings: tuple[str, ...]

    @property
    def healthy(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict[str, object]:
        return {
            "checked_at": self.checked_at,
            "healthy": self.healthy,
            "current_snapshot": self.current_snapshot.to_dict(),
            "current_metrics": asdict(self.current_metrics),
            "warnings": list(self.warnings),
        }


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _read_property(camera: cv2.VideoCapture, property_id: int) -> float | None:
    try:
        value = float(camera.get(property_id))
    except (TypeError, ValueError, cv2.error):
        return None
    return value if _finite(value) else None


def _same_value_probe(
    camera: cv2.VideoCapture,
    property_id: int,
    value: float | None,
) -> bool | None:
    if value is None:
        return False
    try:
        return bool(camera.set(property_id, float(value)))
    except cv2.error:
        return False


def capture_snapshot(
    camera: cv2.VideoCapture,
    *,
    probe_writable: bool = False,
) -> CameraSnapshot:
    try:
        backend = camera.getBackendName()
    except (AttributeError, cv2.error):
        backend = "unknown"

    properties: dict[str, CameraProperty] = {}
    for spec in PROPERTY_SPECS:
        value = _read_property(camera, spec.property_id)
        settable = (
            _same_value_probe(camera, spec.property_id, value)
            if probe_writable and spec.writable_probe
            else None
        )
        properties[spec.name] = CameraProperty(
            name=spec.name,
            property_id=spec.property_id,
            value=value,
            readable=value is not None,
            settable=settable,
        )

    return CameraSnapshot(
        captured_at=datetime.now().isoformat(timespec="seconds"),
        backend=backend,
        properties=properties,
    )


def measure_frame(frame: np.ndarray) -> FrameMetrics:
    if frame is None or frame.size == 0:
        raise ValueError("Cannot measure an empty frame.")

    if frame.ndim == 2:
        gray = frame
        blue = green = red = float(np.mean(frame))
    elif frame.ndim == 3 and frame.shape[2] >= 3:
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        blue, green, red = (
            float(value)
            for value in np.mean(frame[:, :, :3], axis=(0, 1))
        )
    else:
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    gray_float = gray.astype(np.float32)
    return FrameMetrics(
        mean_luminance=float(np.mean(gray_float)),
        luminance_std=float(np.std(gray_float)),
        shadow_fraction=float(np.mean(gray <= 8)),
        highlight_fraction=float(np.mean(gray >= 247)),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        mean_blue=blue,
        mean_green=green,
        mean_red=red,
    )


def summarize_metrics(metrics: Iterable[FrameMetrics]) -> StabilityMetrics:
    samples = tuple(metrics)
    if not samples:
        raise ValueError("At least one frame metric is required.")

    luminance = np.asarray(
        [sample.mean_luminance for sample in samples],
        dtype=np.float64,
    )
    shadows = np.asarray(
        [sample.shadow_fraction for sample in samples],
        dtype=np.float64,
    )
    highlights = np.asarray(
        [sample.highlight_fraction for sample in samples],
        dtype=np.float64,
    )
    sharpness = np.asarray(
        [sample.sharpness for sample in samples],
        dtype=np.float64,
    )
    channels = np.asarray(
        [sample.channel_means for sample in samples],
        dtype=np.float64,
    )

    return StabilityMetrics(
        frame_count=len(samples),
        mean_luminance=float(np.mean(luminance)),
        luminance_std_across_frames=float(np.std(luminance)),
        luminance_range=float(np.ptp(luminance)),
        mean_shadow_fraction=float(np.mean(shadows)),
        mean_highlight_fraction=float(np.mean(highlights)),
        median_sharpness=float(np.median(sharpness)),
        minimum_sharpness=float(np.min(sharpness)),
        channel_ranges=tuple(float(value) for value in np.ptp(channels, axis=0)),
    )


def collect_stability(
    camera: cv2.VideoCapture,
    duration_seconds: float,
    *,
    minimum_frames: int = 12,
) -> tuple[StabilityMetrics, np.ndarray]:
    start = time.perf_counter()
    metrics: list[FrameMetrics] = []
    last_frame: np.ndarray | None = None

    while (
        time.perf_counter() - start < duration_seconds
        or len(metrics) < minimum_frames
    ):
        ok, frame = camera.read()
        if not ok or frame is None:
            continue
        last_frame = frame.copy()
        metrics.append(measure_frame(frame))

    if last_frame is None:
        raise RuntimeError("No camera frames were available during calibration.")
    return summarize_metrics(metrics), last_frame


def _set_and_verify(
    camera: cv2.VideoCapture,
    name: str,
    property_id: int,
    requested: float,
    *,
    tolerance: float = 0.10,
) -> ControlAction:
    before = _read_property(camera, property_id)
    try:
        attempted = bool(camera.set(property_id, float(requested)))
    except cv2.error:
        attempted = False
    after = _read_property(camera, property_id)
    succeeded = bool(
        attempted
        and after is not None
        and abs(after - requested) <= tolerance
    )
    note = ""
    if not attempted:
        note = "driver declined property write"
    elif after is None:
        note = "property could not be read back"
    elif not succeeded:
        note = "driver accepted write but reported a different value"
    return ControlAction(
        control=name,
        attempted=attempted,
        succeeded=succeeded,
        before=before,
        requested=float(requested),
        after=after,
        note=note,
    )


def _manual_exposure_candidates(auto_value: float | None) -> tuple[float, ...]:
    # OpenCV backends use incompatible conventions:
    # DirectShow: 0.25 manual / 0.75 auto
    # V4L2:       1 manual / 3 auto
    # Some MSMF drivers use 0 / 1.
    if auto_value is not None and auto_value > 1.5:
        return (1.0, 0.25, 0.0)
    if auto_value is not None and 0.5 < auto_value < 1.0:
        return (0.25, 1.0, 0.0)
    return (0.0, 0.25, 1.0)


def lock_camera_controls(
    camera: cv2.VideoCapture,
) -> tuple[ControlAction, ...]:
    actions: list[ControlAction] = []

    # Preserve the settled manual values before disabling their automatic
    # controls.  Some drivers reset the manual value during the transition.
    settled_exposure = _read_property(camera, cv2.CAP_PROP_EXPOSURE)
    settled_wb = _read_property(camera, cv2.CAP_PROP_WB_TEMPERATURE)
    settled_focus = _read_property(camera, cv2.CAP_PROP_FOCUS)
    auto_exposure = _read_property(camera, cv2.CAP_PROP_AUTO_EXPOSURE)

    exposure_action: ControlAction | None = None
    for candidate in _manual_exposure_candidates(auto_exposure):
        action = _set_and_verify(
            camera,
            "auto_exposure",
            cv2.CAP_PROP_AUTO_EXPOSURE,
            candidate,
            tolerance=0.12,
        )
        exposure_action = action
        if action.succeeded:
            break
    if exposure_action is not None:
        actions.append(exposure_action)
    if settled_exposure is not None and exposure_action and exposure_action.succeeded:
        actions.append(
            _set_and_verify(
                camera,
                "exposure",
                cv2.CAP_PROP_EXPOSURE,
                settled_exposure,
                tolerance=0.25,
            )
        )

    actions.append(
        _set_and_verify(
            camera,
            "auto_white_balance",
            cv2.CAP_PROP_AUTO_WB,
            0.0,
            tolerance=0.10,
        )
    )
    if settled_wb is not None and actions[-1].succeeded:
        actions.append(
            _set_and_verify(
                camera,
                "white_balance_temperature",
                cv2.CAP_PROP_WB_TEMPERATURE,
                settled_wb,
                tolerance=75.0,
            )
        )

    actions.append(
        _set_and_verify(
            camera,
            "autofocus",
            cv2.CAP_PROP_AUTOFOCUS,
            0.0,
            tolerance=0.10,
        )
    )
    if settled_focus is not None and actions[-1].succeeded:
        actions.append(
            _set_and_verify(
                camera,
                "focus",
                cv2.CAP_PROP_FOCUS,
                settled_focus,
                tolerance=1.0,
            )
        )

    return tuple(actions)


def calibration_warnings(metrics: StabilityMetrics) -> tuple[str, ...]:
    warnings: list[str] = []
    if metrics.mean_luminance < DARK_MEAN_LUMA:
        warnings.append("image is quite dark")
    if metrics.mean_luminance > BRIGHT_MEAN_LUMA:
        warnings.append("image is very bright")
    if metrics.mean_shadow_fraction > MAX_SHADOW_FRACTION:
        warnings.append("a large fraction of pixels are clipped in shadow")
    if metrics.mean_highlight_fraction > MAX_HIGHLIGHT_FRACTION:
        warnings.append("a noticeable fraction of pixels are clipped in highlights")
    if metrics.median_sharpness < MIN_ABSOLUTE_SHARPNESS:
        warnings.append("image sharpness is low; inspect focus and camera motion")
    if metrics.luminance_std_across_frames > MAX_SETTLED_LUMA_STD:
        warnings.append("brightness is still fluctuating after control locking")
    if metrics.luminance_range > MAX_SETTLED_LUMA_RANGE:
        warnings.append("brightness range across verification frames is high")
    if max(metrics.channel_ranges) > MAX_SETTLED_CHANNEL_RANGE:
        warnings.append("color balance is drifting across verification frames")
    return tuple(warnings)


class CameraCalibrator:
    def __init__(
        self,
        report_path: Path = CAMERA_REPORT_PATH,
        preview_path: Path = CAMERA_PREVIEW_PATH,
    ) -> None:
        self.report_path = report_path
        self.preview_path = preview_path
        self.last_report: CameraCalibrationReport | None = None

    def calibrate(
        self,
        camera: cv2.VideoCapture,
        *,
        settle_seconds: float = 2.5,
        verification_seconds: float = 1.0,
        lock_controls: bool = True,
        save: bool = True,
    ) -> CameraCalibrationReport:
        before = capture_snapshot(camera, probe_writable=True)
        settle_metrics, settled_frame = collect_stability(
            camera,
            settle_seconds,
        )
        settled = capture_snapshot(camera)
        actions = (
            lock_camera_controls(camera)
            if lock_controls
            else tuple()
        )
        locked_metrics, locked_frame = collect_stability(
            camera,
            verification_seconds,
        )
        locked = capture_snapshot(camera)
        warnings = list(calibration_warnings(locked_metrics))

        for control in ("auto_exposure", "auto_white_balance", "autofocus"):
            matching = [action for action in actions if action.control == control]
            if lock_controls and matching and not matching[-1].succeeded:
                warnings.append(
                    f"{control.replace('_', ' ')} could not be locked by this driver"
                )

        report = CameraCalibrationReport(
            created_at=datetime.now().isoformat(timespec="seconds"),
            before_settle=before,
            settled=settled,
            locked=locked,
            settle_metrics=settle_metrics,
            locked_metrics=locked_metrics,
            actions=actions,
            warnings=tuple(dict.fromkeys(warnings)),
            preview_metrics=measure_frame(locked_frame),
        )
        self.last_report = report

        if save:
            self.save_report(report, locked_frame)
        return report

    def save_report(
        self,
        report: CameraCalibrationReport,
        preview_frame: np.ndarray | None = None,
    ) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )
        if preview_frame is not None:
            cv2.imwrite(str(self.preview_path), preview_frame)

    def make_reference(
        self,
        camera: cv2.VideoCapture,
        frame: np.ndarray,
    ) -> CameraReference:
        return CameraReference(
            created_at=datetime.now().isoformat(timespec="seconds"),
            snapshot=capture_snapshot(camera),
            frame_metrics=measure_frame(frame),
        )

    def check_reference(
        self,
        camera: cv2.VideoCapture,
        frame: np.ndarray,
        reference: CameraReference,
    ) -> CameraHealth:
        current_snapshot = capture_snapshot(camera)
        current_metrics = measure_frame(frame)
        warnings: list[str] = []

        for name, tolerance in MAX_PROPERTY_DELTAS.items():
            baseline = reference.snapshot.value(name)
            current = current_snapshot.value(name)
            if baseline is None or current is None:
                continue
            if abs(current - baseline) > tolerance:
                warnings.append(
                    f"camera {name.replace('_', ' ')} changed "
                    f"from {baseline:g} to {current:g}"
                )

        baseline_metrics = reference.frame_metrics
        luma_delta = abs(
            current_metrics.mean_luminance - baseline_metrics.mean_luminance
        )
        allowed_luma_delta = max(
            MAX_REFERENCE_LUMA_DELTA,
            abs(baseline_metrics.mean_luminance) * MAX_REFERENCE_LUMA_FRACTION,
        )
        if luma_delta > allowed_luma_delta:
            warnings.append(
                "scene brightness changed substantially since the empty-tray background"
            )

        channel_delta = max(
            abs(current - baseline)
            for current, baseline in zip(
                current_metrics.channel_means,
                baseline_metrics.channel_means,
                strict=True,
            )
        )
        if channel_delta > MAX_REFERENCE_CHANNEL_DELTA:
            warnings.append(
                "scene color balance changed substantially since the empty-tray background"
            )

        if (
            baseline_metrics.sharpness > 0
            and current_metrics.sharpness
            < baseline_metrics.sharpness * MIN_REFERENCE_SHARPNESS_RATIO
        ):
            warnings.append(
                "image sharpness fell substantially since the empty-tray background"
            )

        return CameraHealth(
            checked_at=datetime.now().isoformat(timespec="seconds"),
            current_snapshot=current_snapshot,
            current_metrics=current_metrics,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def session_metadata(
        self,
        camera: cv2.VideoCapture,
        frame: np.ndarray,
        reference: CameraReference | None = None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "current_snapshot": capture_snapshot(camera).to_dict(),
            "current_frame_metrics": asdict(measure_frame(frame)),
        }
        if self.last_report is not None:
            metadata["calibration_report"] = self.last_report.to_dict()
        if reference is not None:
            metadata["background_reference"] = reference.to_dict()
            metadata["health"] = self.check_reference(
                camera,
                frame,
                reference,
            ).to_dict()
        return metadata


def concise_report(report: CameraCalibrationReport) -> str:
    metrics = report.locked_metrics
    lines = [
        "Camera calibration complete:",
        f"  backend: {report.locked.backend}",
        f"  verification frames: {metrics.frame_count}",
        f"  mean luminance: {metrics.mean_luminance:.1f}",
        f"  brightness frame-to-frame std: "
        f"{metrics.luminance_std_across_frames:.2f}",
        f"  median sharpness: {metrics.median_sharpness:.1f}",
    ]
    for action in report.actions:
        state = "locked" if action.succeeded else "not locked"
        lines.append(
            f"  {action.control.replace('_', ' ')}: {state}"
        )
    if report.warnings:
        lines.append("  warnings:")
        lines.extend(f"    - {warning}" for warning in report.warnings)
    else:
        lines.append("  warnings: none")
    return "\n".join(lines)
