from __future__ import annotations

import argparse
import sys

import numpy as np

from icy_dice.config import get_profile
from icy_dice.models import ModelEnsemble


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load configured models and run one synthetic inference."
    )
    parser.add_argument(
        "die_types",
        nargs="*",
        default=["d6", "d8"],
    )
    args = parser.parse_args()

    crop = np.full((128, 128, 3), 128, dtype=np.uint8)
    failures = 0

    for die_type in args.die_types:
        profile = get_profile(die_type)
        try:
            ensemble = ModelEnsemble(profile)
            outputs = ensemble.classify([crop])
        except Exception as error:
            failures += 1
            print(f"{profile.die_type}: FAILED: {error}")
            continue

        shapes = {
            name: tuple(probabilities.shape)
            for name, probabilities in outputs.items()
        }
        print(
            f"{profile.die_type}: loaded "
            f"{len(ensemble.bundles)} model(s) on "
            f"{ensemble.device}; output shapes {shapes}"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
