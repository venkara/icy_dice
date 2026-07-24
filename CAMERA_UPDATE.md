# Camera calibration update

## Files added

- `icy_dice/camera.py`
- `calibrate_camera.py`
- `smoke_test_camera.py`

## Files updated

- `icy_dice/controller.py`
- `icy_dice/feedback.py`
- `icy_dice/interfaces/opencv_cli.py`
- `icy_dice/__init__.py`
- `collect_dataset.py`
- `README.md`

## First test

From the repository root:

```powershell
py .\smoke_test_camera.py
py .\calibrate_camera.py
```

The second command requires the physical camera. Remove the dice, use the
normal lighting, and leave the tray in view. Inspect:

```text
calibration/camera_calibration.json
calibration/camera_calibration_preview.png
```

Then test the normal reader:

```powershell
py .\run_reader.py --die-type d8 --count 4
```

The reader calibrates at startup. Press `K` to repeat calibration. Any camera
recalibration invalidates the current background and requires a new `B`.

## What is measured

- All readable OpenCV camera properties
- Whether same-value writes appear to be accepted
- Exposure, gain, white-balance temperature, focus, and automatic-control state
- Mean luminance and contrast
- Shadow and highlight clipping
- Laplacian-variance sharpness
- Frame-to-frame brightness range and standard deviation
- Frame-to-frame channel/color drift

## What is locked

After a settling period, the module attempts to disable:

- Automatic exposure
- Automatic white balance
- Autofocus

It restores the settled manual exposure, white-balance temperature, and focus
where the driver permits. OpenCV backends use incompatible conventions, so
every requested change is read back and reported. A failed lock is a warning,
not a crash.

## Background drift checking

Pressing `B` records a property and image-metric reference. The reader warns if
camera properties or the image appearance later differ materially. Warnings do
not block capture; they indicate that the background subtraction may benefit
from a fresh `B` image.

## Metadata

Camera calibration, current settings, background reference, and health results
are added to:

- Reader debug JSON
- User-feedback JSON
- New collector session `metadata.json`
