from __future__ import annotations

import argparse
import sys

from icy_dice.camera import CameraCalibrator, concise_report
from icy_dice import vision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Icy Dice camera stability and attempt to lock "
            "exposure, white balance, and focus."
        )
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.5,
        help="Seconds to let automatic camera controls settle.",
    )
    parser.add_argument(
        "--verification-seconds",
        type=float,
        default=1.0,
        help="Seconds to measure stability after locking.",
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Measure the camera without disabling automatic controls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    camera = vision.open_camera()
    calibrator = CameraCalibrator()

    try:
        print("Remove all dice and leave the tray lighting in its normal state.")
        print("Letting camera controls settle...")
        report = calibrator.calibrate(
            camera,
            settle_seconds=args.settle_seconds,
            verification_seconds=args.verification_seconds,
            lock_controls=not args.no_lock,
        )
        print(concise_report(report))
        print("\nSaved:")
        print(f"  {calibrator.report_path.resolve()}")
        print(f"  {calibrator.preview_path.resolve()}")
        return 0
    except RuntimeError as error:
        print(f"Camera calibration failed: {error}")
        return 1
    finally:
        camera.release()


if __name__ == "__main__":
    sys.exit(main())
