from __future__ import annotations

import os
import pandas as pd

from research.engine.backtest_core import load_flight_recorder
from research.strategies.regime_classifier_v1 import (
    build_regime_classifier_v1,
    RegimeClassifierV1Params,
)

OUTPUT_DIR = os.path.join("output")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(
        "research",
        "backtests",
        "data",
        "flight_recorder.csv",
    )

    print("Loading data...")
    df = load_flight_recorder(csv_path)

    params = RegimeClassifierV1Params()

    print("Building regimes...")
    regimes = build_regime_classifier_v1(df, params)

    out = pd.concat([df, regimes], axis=1)

    out_path = os.path.join(
        OUTPUT_DIR,
        "regime_classifier_v1_labeled.csv",
    )

    out.to_csv(out_path, index=False)

    print("Saved:")
    print(f" - {out_path}")

    print("\nRegime distribution:")
    print(out["regime_label"].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    main()
