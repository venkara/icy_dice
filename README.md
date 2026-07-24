# Icy Dice refactor

This directory separates the recognition system from its current OpenCV and
PowerShell user interface.

## Structure

```text
icy_dice/
    config.py       Die-type/model profiles
    vision.py       Camera, ArUco, segmentation, crops, burst ranking
    models.py       Checkpoint loading and batch inference
    recognition.py Burst aggregation, confidence rules, retry matching
    feedback.py     Hard-example storage and user-confirmed truth
    workflow.py     UI-independent roll state machine
    controller.py   Facade for CLI or future GUI
    presentation.py Image annotation and result sheets
    interfaces/
        opencv_cli.py
run_reader.py
collect_dataset.py
train_die_classifier.py
smoke_test_models.py
```

`ReaderController` does not create windows, read keys, or call `input()`.
A later Tkinter, Qt, or web front end can drive the same controller and
workflow methods.

## Required project folders

Keep the refactor directory at the project root, with models stored as:

```text
models/
    d6_center/
        d6_mobilenet_v3_small.pt
    d8_center55/
        d8_mobilenet_v3_small.pt
    d8_center60/
        d8_mobilenet_v3_small.pt
```

## Generic reader

```powershell
py .\run_reader.py --die-type d8 --count 8
py .\run_reader.py --die-type d6 --count 6
```

The d6 profile uses one 68% center-crop model. The d8 profile uses the
55%/60% ensemble. Both use the same burst capture, review, retry, and feedback
workflow.

## Model smoke test

This checks checkpoint loading and one inference pass without opening the
camera:

```powershell
py .\smoke_test_models.py d6 d8
```

It does not replace a live camera test.

## d10 collection and training

The existing collector already accepts d10 and stores printed labels `0`–`9`:

```powershell
py .\collect_dataset.py
```

Train a crop candidate with:

```powershell
py .\train_die_classifier.py `
    --die-type d10 `
    --crop-fraction 0.60
```

The default output folder for that command is `models/d10_center60/`.
Run a crop sweep before adding d10 models to `icy_dice/config.py`.

For ordinary d10 semantics, the profile maps printed `0` to a roll value of
10. Training and feedback directories continue to use the printed label `0`.

## GUI transition

A GUI should call `ReaderController` and observe `RollWorkflow.state`.
The main operations are:

```python
controller.analyze_frame(frame)
controller.set_background(rectified)
selection = controller.capture_selection(camera, window_name)
controller.recognize_initial(selection)
controller.review(wrong_indices, true_labels)
controller.recognize_retry(selection)
```

The current OpenCV interface is now only one client of those operations.

## Camera calibration and control locking

The reader now lets automatic exposure, white balance, and autofocus settle at
startup, then asks the camera driver to lock them. Because webcam drivers use
incompatible property conventions, each write is verified and unsupported
controls are reported rather than assumed.

Run the camera module by itself with an empty tray:

```powershell
py .\calibrate_camera.py
```

It writes:

```text
calibration/camera_calibration.json
calibration/camera_calibration_preview.png
```

The JSON records all readable OpenCV camera properties, attempted control
locks, brightness/clipping/sharpness measurements, temporal brightness and
color stability, and any warnings.

The generic reader runs this calibration automatically. Options are:

```powershell
py .\run_reader.py --die-type d8 --count 8
py .\run_reader.py --die-type d8 --count 8 --skip-camera-calibration
py .\run_reader.py --die-type d8 --count 8 --no-lock-camera-controls
py .\run_reader.py --die-type d8 --count 8 --settle-seconds 4
```

Press **K** in the reader or collector to recalibrate. Recalibration invalidates
the current empty-tray background, because the camera image may have changed.
Press **B** afterward.

When **B** is pressed, the controller stores a camera-property and image-metric
reference. It periodically warns when exposure, gain, white balance, focus,
brightness, color balance, or sharpness changes materially from that reference.
Warnings do not block capture; they indicate that a fresh background may be
needed.

Debug records, feedback records, and newly collected session metadata include
the available camera settings and calibration measurements.

A hardware-free module test is included:

```powershell
py .\smoke_test_camera.py
```
