"""Train, select and freeze the expected-points model.

    python -m ml.sources          # download the historical archive (once)
    python -m ml.train            # assemble, compare candidates, persist winner
    python -m ml.backtest         # walk-forward evaluation + methodology results

Protocol, in order, with the discipline that makes the numbers mean something:

1. Assemble the panel and build features. No random shuffling anywhere.
2. Split by season: train / validation / test (:mod:`ml.config`).
3. Fit every candidate on **train** and score on **validation**.
4. Pick the winner on validation by ``spearman_gw`` — the ranking metric, not
   MAE, because the optimiser consumes an ordering.
5. Refit the winner on **train + validation** and persist it.
6. Score once on **test**, alongside every baseline, and write the table.

The test split is touched exactly once, in step 6, after every choice has been
made. Rerunning step 3 with a different feature idea and re-reading the test
number would turn it into a validation set with extra steps, and the reported
figure would be an optimistic one dressed as a held-out one.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import baselines, config, features as feature_mod, metrics, models, panel, splits

SELECTION_METRIC = "spearman_gw"


def load_frame(refresh: bool = False):
    """Panel -> features, with a leakage assertion on real data before returning."""
    raw = panel.build_cached(refresh=refresh)
    frame = feature_mod.build_frame(raw)

    sample = frame.iloc[len(frame) // 2]
    feature_mod.assert_no_lookahead(
        raw,
        frame,
        season=sample["season"],
        element=int(sample["element"]),
        gameweek=int(sample["GW"]),
    )
    print(
        f"[features] {len(frame):,} rows, {len(feature_mod.FEATURES)} features, "
        f"seasons {sorted(frame['season'].unique())}"
    )
    return frame


def run(refresh: bool = False, candidates=None, output: str | None = None) -> dict:
    frame = load_frame(refresh=refresh)
    train, valid, test = splits.season_split(frame)

    for name, part in (("train", train), ("valid", valid), ("test", test)):
        if part.empty:
            raise SystemExit(
                f"The {name} split is empty. Check TRAIN/VALID/TEST_SEASONS in "
                "ml/config.py against the seasons actually cached."
            )
        print(f"[split] {name}: {len(part):,} rows  seasons={sorted(part['season'].unique())}")

    calibration = baselines.add_all(train, valid, test)
    print(f"[baselines] FCPS calibrated by {calibration['calibrator']} on train only")

    candidates = list(candidates or models.available())

    # ── Step 3-4: select on validation ──────────────────────────────────────
    validation_scores: dict[str, dict] = {}
    fitted = {}
    for name in candidates:
        print(f"[fit] {name} on train ...", flush=True)
        try:
            model = models.fit(name, train)
        except Exception as error:
            print(f"[fit] {name} failed: {type(error).__name__}: {error}")
            continue
        fitted[name] = model
        valid[f"pred_{name}"] = models.predict(model, valid)
        validation_scores[name] = metrics.evaluate(valid, f"pred_{name}")
        print(
            f"      valid  MAE {validation_scores[name]['mae']:.3f}  "
            f"Spearman {validation_scores[name]['spearman_gw']:.3f}  "
            f"P@20 {validation_scores[name]['precision_at_k']:.3f}"
        )

    if not validation_scores:
        raise SystemExit("Every candidate failed to fit. See the errors above.")

    winner = max(
        validation_scores,
        key=lambda name: (
            _finite(validation_scores[name][SELECTION_METRIC]),
            -_finite(validation_scores[name]["mae"], default=1e9),
        ),
    )
    print(f"[select] {winner} (best {SELECTION_METRIC} on validation)")

    # ── Step 5: refit on train + validation ─────────────────────────────────
    import pandas as pd

    full = pd.concat([train, valid], ignore_index=True)
    final = models.fit(winner, full)

    # ── Step 6: score once on test ──────────────────────────────────────────
    test[f"pred_{winner}"] = models.predict(final, test)
    columns = [f"pred_{winner}"] + list(baselines.BASELINE_COLUMNS)
    test_scores = metrics.compare(test, columns)

    # FCPS ranking metrics use the raw score; the calibrated column is for MAE
    # only, and a monotone calibration cannot change a rank metric anyway.
    fcps_rank = metrics.evaluate(test, "bl_fcps")
    for key in ("spearman_gw", "precision_at_k", "mean_actual_top_k", "lift_over_mean"):
        test_scores["bl_fcps_points"][key] = fcps_rank[key]

    metadata = {
        "model": winner,
        "features": list(feature_mod.FEATURES),
        "target": feature_mod.TARGET,
        "train_seasons": list(config.TRAIN_SEASONS),
        "valid_seasons": list(config.VALID_SEASONS),
        "test_seasons": list(config.TEST_SEASONS),
        "refit_on": "train + validation",
        "min_gameweek": config.MIN_GAMEWEEK,
        "selection_metric": SELECTION_METRIC,
        "n_train_rows": int(len(full)),
        "n_test_rows": int(len(test)),
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "fcps_calibration": calibration,
        "importances": models.importances(final),
        "candidates": candidates,
    }
    models.save(final, metadata)
    print(f"[save] {config.artifact_path()}")

    print("\nValidation (model selection):")
    print(metrics.format_table(validation_scores))
    print("\nTest (scored once, after selection):")
    print(metrics.format_table(test_scores))

    if output:
        config.ensure_dirs()
        (config.REPORT_DIR / output).write_text(json.dumps(metadata, indent=2, default=str))
        print(f"[report] {config.REPORT_DIR / output}")
    return metadata


def _finite(value, default: float = -1e9) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if value != value else value  # NaN check without importing numpy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="rebuild the panel cache")
    parser.add_argument(
        "--models", nargs="*", default=None, help="candidate subset to compare"
    )
    parser.add_argument("--report", default="train_report.json")
    args = parser.parse_args(argv)

    run(refresh=args.refresh, candidates=args.models, output=args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
