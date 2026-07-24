# Verification performed

The generated Python files were parsed successfully.

A model smoke test was run against the available trained checkpoints:

```text
d6: loaded 1 model on CPU; output shape (1, 6)
d8: loaded 2 models on CPU; output shapes (1, 8) and (1, 8)
```

This verifies that the shared `ModelEnsemble` loads the existing d6 and d8
checkpoint formats and performs inference through the same code path.

The camera, ArUco, physical segmentation, and selective-retry workflow cannot
be fully tested without the user's installed camera and tray.

## Camera module

The camera module passed a synthetic-driver smoke test covering:

- Property snapshot creation
- DirectShow-style automatic exposure locking
- Automatic white-balance locking
- Autofocus locking
- Frame luminance/color/sharpness metrics
- Background reference and health comparison

All generated Python files were parsed successfully. The reader and calibration
command-line help paths were imported and executed successfully. Physical
camera behavior remains driver-dependent and must be checked on the Elgato
camera.
