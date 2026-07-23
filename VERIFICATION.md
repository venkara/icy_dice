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
