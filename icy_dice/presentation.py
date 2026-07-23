from __future__ import annotations

import math

import cv2
import numpy as np

from .config import DieProfile
from .recognition import RecognitionResult
from . import vision


def add_two_line_banner(
    image: np.ndarray,
    first_line: str,
    second_line: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    height = 92
    output = cv2.copyMakeBorder(
        image,
        height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    cv2.putText(
        output,
        first_line,
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        second_line,
        (18, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def annotate_result(
    result: RecognitionResult,
) -> np.ndarray:
    output = result.representative_tray.copy()

    for index, (candidate, prediction) in enumerate(
        zip(
            result.representative_candidates,
            result.predictions,
            strict=True,
        ),
        start=1,
    ):
        x, y, width, height = candidate.bbox
        color = (
            (0, 255, 0)
            if prediction.accepted
            else (0, 0, 255)
        )
        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            color,
            3,
        )

        if prediction.flagged_wrong:
            state = f"WRONG truth={prediction.true_value}"
        elif prediction.user_confirmed:
            state = "CONFIRMED"
        elif prediction.accepted and not result.all_accepted:
            state = "LOCKED"
        else:
            state = ""

        label = (
            f"#{index}: {prediction.value} "
            f"{prediction.combined_confidence:.0%}"
            f"{' ' + state if state else ''}"
        )
        cv2.putText(
            output,
            label,
            (x, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    values = " + ".join(
        str(prediction.value)
        for prediction in result.predictions
    )
    split_text = (
        f" | auto-split {result.representative_split_count}"
        if result.representative_split_count
        else ""
    )

    if result.all_accepted:
        first = (
            f"ACCEPTED | {values} = {result.total}"
            f" | {result.frames_used} burst frames{split_text}"
        )
        second = (
            "Press F if any value is wrong; "
            "otherwise remove dice and press B."
        )
        color = (0, 255, 0)
    else:
        uncertain = sum(
            not prediction.accepted
            for prediction in result.predictions
        )
        first = (
            f"UNCERTAIN ({uncertain}) | "
            f"provisional {values} = {result.total}"
            f" | {result.frames_used} burst frames{split_text}"
        )
        if result.reviewed:
            second = (
                "Move RED dice only; keep GREEN dice still; "
                "then press R."
            )
        else:
            second = (
                "Do not move dice. Press F to identify "
                "which displayed values are wrong."
            )
        color = (0, 0, 255)

    return add_two_line_banner(output, first, second, color)


def result_sheet(
    result: RecognitionResult,
) -> np.ndarray:
    tile_width = 250
    tile_height = 258
    columns = min(5, max(1, len(result.predictions)))
    rows = math.ceil(len(result.predictions) / columns)
    sheet = np.full(
        (rows * tile_height, columns * tile_width, 3),
        45,
        dtype=np.uint8,
    )

    for index, prediction in enumerate(result.predictions, start=1):
        row, column = divmod(index - 1, columns)
        x = column * tile_width
        y = row * tile_height
        image_x = x + 61
        image_y = y + 34
        sheet[
            image_y : image_y + vision.CROP_SIZE,
            image_x : image_x + vision.CROP_SIZE,
        ] = prediction.display_crop

        color = (
            (0, 255, 0)
            if prediction.accepted
            else (0, 0, 255)
        )
        cv2.putText(
            sheet,
            f"Die {index}: {prediction.value}",
            (x + 12, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

        line_y = y + 184
        for model_result in prediction.model_results:
            cv2.putText(
                sheet,
                (
                    f"{model_result.model_name}: "
                    f"{model_result.value} "
                    f"{model_result.confidence:.0%} "
                    f"votes {model_result.vote_fraction:.0%}"
                ),
                (x + 10, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )
            line_y += 19

        if prediction.flagged_wrong:
            state = f"WRONG -> {prediction.true_value}; MOVE + R"
        elif prediction.user_confirmed:
            state = "USER CONFIRMED; KEEP STILL"
        elif prediction.accepted:
            state = (
                "KEEP STILL"
                if not result.all_accepted
                else "MODEL ACCEPTED"
            )
        elif result.reviewed:
            state = "REPOSITION + R"
        else:
            state = "REVIEW WITH F"

        cv2.putText(
            sheet,
            state,
            (x + 10, y + tile_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return sheet


def raw_marker_view(
    frame: np.ndarray,
    corners,
    ids,
    marker_count: int,
) -> np.ndarray:
    output = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(output, corners, ids)
    return vision.add_status_banner(
        output,
        (
            f"ARUCO MARKERS {marker_count}/4 | "
            "adjust camera, lighting, or marker visibility"
        ),
        (0, 0, 255),
        height=58,
    )


def waiting_for_background_view(
    tray: np.ndarray,
    profile: DieProfile,
    count: int,
) -> np.ndarray:
    return vision.add_status_banner(
        tray,
        (
            f"REMOVE ALL DICE | press B for fresh background | "
            f"next {count}{profile.die_type}"
        ),
        (0, 255, 255),
        height=58,
    )
