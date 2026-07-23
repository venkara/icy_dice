from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

try:
    import msvcrt
except ImportError:
    msvcrt = None

import cv2
import numpy as np

from ..config import get_profile
from ..controller import ReaderController
from ..presentation import (
    annotate_result,
    raw_marker_view,
    result_sheet,
    waiting_for_background_view,
)
from ..workflow import WorkflowState
from .. import vision


RESULT_WINDOW_X = 425
RESULT_WINDOW_Y = 1150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Icy Dice generic live reader."
    )
    parser.add_argument(
        "--die-type",
        default="d8",
        help="Configured die type, currently d6 or d8.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Initial number of dice.",
    )
    parser.add_argument(
        "--model-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--vote-threshold",
        type=float,
        default=None,
    )
    return parser.parse_args()


def prompt_count(
    die_type: str,
    maximum: int,
    current: int | None = None,
) -> int:
    while True:
        prompt = (
            f"How many {die_type} dice? "
            if current is None
            else f"New {die_type} count [{current}]: "
        )
        response = input(prompt).strip()
        if not response and current is not None:
            return current
        normalized = response.lower().replace(" ", "")
        if normalized.endswith(die_type):
            normalized = normalized[: -len(die_type)]
        try:
            count = int(normalized)
        except ValueError:
            print(f"Enter a count such as 4 or 4{die_type}.")
            continue
        if 1 <= count <= maximum:
            return count
        print(f"Count must be between 1 and {maximum}.")


def poll_command() -> str | None:
    key = cv2.waitKeyEx(1)
    if key == 27:
        return "q"
    if key != -1:
        code = key & 0xFF
        if code:
            character = chr(code).lower()
            if character in {"b", "c", "f", "n", "r", "s", "q"}:
                return character

    if msvcrt is not None and msvcrt.kbhit():
        character = msvcrt.getwch()
        if character in {"\x00", "\xe0"}:
            if msvcrt.kbhit():
                msvcrt.getwch()
            return None
        if character == "\x1b":
            return "q"
        character = character.lower()
        if character in {"b", "c", "f", "n", "r", "s", "q"}:
            return character
    return None


def parse_die_indices(
    text: str,
    die_count: int,
) -> set[int]:
    if not text.strip():
        return set()

    values: set[int] = set()
    for token in text.replace(",", " ").split():
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(token))

    invalid = sorted(
        value
        for value in values
        if not 1 <= value <= die_count
    )
    if invalid:
        raise ValueError(
            "Die numbers out of range: "
            + ", ".join(map(str, invalid))
        )
    return {value - 1 for value in values}


def prompt_review(controller: ReaderController):
    result = controller.result
    if result is None:
        raise RuntimeError("No result is available.")

    print("\nReview the numbered dice in the result window.")
    print("Enter every die whose displayed value is wrong.")
    print("Press Enter if all displayed values are correct.")

    while True:
        try:
            wrong = parse_die_indices(
                input("Wrong die numbers: "),
                len(result.predictions),
            )
            break
        except (TypeError, ValueError) as error:
            print(f"{error} Examples: 2 5, 2,5, or 2-4.")

    true_labels: dict[int, str] = {}
    for die_index in sorted(wrong):
        prediction = result.predictions[die_index]
        while True:
            response = input(
                f"True value for die {die_index + 1} "
                f"(displayed {prediction.value}): "
            ).strip()
            try:
                semantic_value = int(response)
                true_label = controller.profile.label_for_value(
                    semantic_value
                )
            except (TypeError, ValueError):
                allowed = ", ".join(
                    map(str, controller.profile.allowed_values)
                )
                print(f"Enter one of: {allowed}.")
                continue
            if true_label == prediction.label:
                print(
                    "That matches the displayed value; remove this die "
                    "from the wrong-dice list or enter its actual value."
                )
                continue
            true_labels[die_index] = true_label
            break

    return controller.review(wrong, true_labels)


def show_result_window(
    name: str,
    image: np.ndarray,
) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    maximum_width = 1200
    maximum_height = 760
    scale = min(
        1.0,
        maximum_width / max(image.shape[1], 1),
        maximum_height / max(image.shape[0], 1),
    )
    cv2.resizeWindow(
        name,
        max(420, int(round(image.shape[1] * scale))),
        max(260, int(round(image.shape[0] * scale))),
    )
    cv2.moveWindow(name, RESULT_WINDOW_X, RESULT_WINDOW_Y)
    cv2.imshow(name, image)
    cv2.waitKey(1)


def save_debug(
    controller: ReaderController,
    raw_frame,
    rectified,
    mask,
    annotated,
    sheet,
) -> None:
    directory = controller.profile.debug_directory
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = directory / timestamp

    images = {
        "camera": raw_frame,
        "tray": rectified,
        "mask": mask,
        "annotated": annotated,
        "predictions": sheet,
    }
    for suffix, image in images.items():
        if image is not None:
            cv2.imwrite(str(Path(f"{stem}_{suffix}.png")), image)

    result = controller.result
    metadata = {
        "die_type": controller.profile.die_type,
        "count": controller.workflow.count,
        "state": controller.workflow.state.name,
        "predictions": [],
    }
    if result is not None:
        metadata["predictions"] = [
            {
                "die": index,
                "label": prediction.label,
                "value": prediction.value,
                "accepted": prediction.accepted,
                "user_confirmed": prediction.user_confirmed,
                "flagged_wrong": prediction.flagged_wrong,
                "true_label": prediction.true_label,
                "true_value": prediction.true_value,
                "combined_confidence": prediction.combined_confidence,
                "models": [
                    {
                        "name": model_result.model_name,
                        "label": model_result.label,
                        "value": model_result.value,
                        "confidence": model_result.confidence,
                        "vote_fraction": model_result.vote_fraction,
                    }
                    for model_result in prediction.model_results
                ],
            }
            for index, prediction in enumerate(
                result.predictions,
                start=1,
            )
        ]
    Path(f"{stem}_result.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("Debug files saved in:", directory.resolve())


def main() -> int:
    args = parse_args()
    profile = get_profile(args.die_type)

    if args.model_threshold is not None:
        profile = replace(
            profile,
            model_confidence_threshold=args.model_threshold,
        )
    if args.vote_threshold is not None:
        profile = replace(
            profile,
            frame_vote_threshold=args.vote_threshold,
        )

    if not profile.models:
        print(
            f"No models are configured for {profile.die_type}. "
            "Train them and add ModelSpec entries in icy_dice/config.py."
        )
        return 2

    count = (
        args.count
        if args.count is not None
        else prompt_count(profile.die_type, profile.max_count)
    )
    controller = ReaderController(profile, count)
    camera = vision.open_camera()

    main_window = f"Icy Dice - {profile.die_type} Live Reader"
    result_window = "Icy Dice - Result Details"
    cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(main_window, 1000, 735)

    print(f"Icy Dice generic reader: {profile.die_type}")
    print("Controls: B background, C capture, F review, R retry,")
    print("          N new count, S save debug, Q quit")
    print(
        f"Models loaded on {controller.ensemble.device}: "
        + ", ".join(bundle.name for bundle in controller.ensemble.bundles)
    )
    print("\nRemove all dice and press B.")

    last_frame = None
    last_rectified = None
    last_mask = None
    last_main = None
    last_sheet = None
    last_corners = None
    last_ids = None
    marker_count = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera frame read failed.")
                return 1
            last_frame = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = vision.detect_markers(
                gray,
                controller.detector,
                controller.dictionary,
                controller.parameters,
            )
            last_corners = corners
            last_ids = ids
            marker_count = 0 if ids is None else len(ids)

            if marker_count == 4:
                source_points = vision.find_inward_marker_corners(corners)
                last_rectified = vision.rectify_tray(frame, source_points)

            if (
                controller.result is not None
                and controller.workflow.state
                in {
                    WorkflowState.REVIEW_AVAILABLE,
                    WorkflowState.RETRY_READY,
                    WorkflowState.COMPLETE,
                }
                and last_main is not None
            ):
                cv2.imshow(main_window, last_main)
            elif marker_count != 4:
                last_main = raw_marker_view(
                    frame,
                    corners,
                    ids,
                    marker_count,
                )
                cv2.imshow(main_window, last_main)
            elif controller.workflow.background is None:
                last_mask = None
                last_main = waiting_for_background_view(
                    last_rectified,
                    profile,
                    controller.workflow.count,
                )
                cv2.imshow(main_window, last_main)
            else:
                last_mask = vision.foreground_mask(
                    last_rectified,
                    controller.workflow.background,
                )
                candidates, _, split_count = vision.detect_die_candidates(
                    last_mask,
                    controller.workflow.count,
                )
                last_main = vision.annotate_candidates(
                    last_rectified,
                    candidates,
                    controller.workflow.request,
                    split_count=split_count,
                )
                cv2.imshow(main_window, last_main)

            command = poll_command()
            if command == "q":
                return 0

            if command == "b":
                if marker_count != 4 or last_rectified is None:
                    print("All four markers are required.")
                    continue

                existing_background = controller.workflow.background
                if existing_background is not None:
                    removal_mask = vision.foreground_mask(
                        last_rectified,
                        existing_background,
                    )
                    if vision.connected_regions(removal_mask):
                        print(
                            "Background capture blocked: foreground "
                            "objects are still visible. Remove all dice."
                        )
                        continue

                controller.set_background(last_rectified)
                last_main = None
                last_sheet = None
                vision.close_window_if_open(result_window)
                print(
                    f"Background captured. Roll "
                    f"{controller.workflow.request.expression}; "
                    "after the dice stop, press C."
                )

            if command == "n":
                new_count = prompt_count(
                    profile.die_type,
                    profile.max_count,
                    controller.workflow.count,
                )
                controller.workflow.set_count(new_count)
                last_main = None
                last_sheet = None
                vision.close_window_if_open(result_window)
                print("Remove all dice and press B.")

            if command == "c":
                if controller.workflow.background is None:
                    print("Capture a fresh background with B first.")
                    continue
                if controller.result is not None:
                    print(
                        "A result is displayed. Use F to review it, "
                        "R after review, or B for a new roll."
                    )
                    continue
                try:
                    selection = controller.capture_selection(
                        camera,
                        main_window,
                    )
                    result = controller.recognize_initial(selection)
                except RuntimeError as error:
                    print(f"Capture failed: {error}")
                    continue
                last_main = annotate_result(result)
                last_sheet = result_sheet(result)
                cv2.imshow(main_window, last_main)
                show_result_window(result_window, last_sheet)
                print(
                    f"Values: {[p.value for p in result.predictions]} "
                    f"Total: {result.total}"
                )
                if result.all_accepted:
                    print(
                        "Accepted. Press F if any displayed value is wrong."
                    )
                else:
                    print(
                        "Do not move dice. Press F to review the result."
                    )

            if command == "f":
                if controller.result is None:
                    print("No result is available for review.")
                    continue
                if controller.result.reviewed:
                    print("This capture has already been reviewed.")
                    continue
                try:
                    outcome = prompt_review(controller)
                except (RuntimeError, ValueError) as error:
                    print(f"Review failed: {error}")
                    continue
                last_main = annotate_result(outcome.result)
                last_sheet = result_sheet(outcome.result)
                cv2.imshow(main_window, last_main)
                show_result_window(result_window, last_sheet)
                for path in outcome.saved_examples:
                    print("Feedback saved:", path.resolve())
                if outcome.result.all_accepted:
                    print("All displayed values are accepted.")
                else:
                    print(
                        "Move only red dice, keep green dice still, "
                        "then press R."
                    )

            if command == "r":
                if controller.workflow.state is not WorkflowState.RETRY_READY:
                    print(
                        "Review with F and flag actual errors before retrying."
                    )
                    continue
                try:
                    selection = controller.capture_selection(
                        camera,
                        main_window,
                    )
                    result = controller.recognize_retry(selection)
                except RuntimeError as error:
                    print(f"Retry failed: {error}")
                    continue
                last_main = annotate_result(result)
                last_sheet = result_sheet(result)
                cv2.imshow(main_window, last_main)
                show_result_window(result_window, last_sheet)
                print(
                    f"Values: {[p.value for p in result.predictions]} "
                    f"Total: {result.total}"
                )
                if result.all_accepted:
                    print("All dice accepted.")
                else:
                    print(
                        "Do not move dice. Press F to review this retry."
                    )

            if command == "s":
                save_debug(
                    controller,
                    last_frame,
                    last_rectified,
                    last_mask,
                    last_main,
                    last_sheet,
                )
    finally:
        camera.release()
        vision.close_window_if_open(result_window)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
