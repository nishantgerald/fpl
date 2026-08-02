"""Walk-forward backtest, and the results section of the methodology document.

    python -m ml.backtest --season 2024-25 --step 2

``ml.train`` freezes a model before the test season and scores it once. That is
the honest *lower* bound, but it isn't how the thing would be run: in production
the model is retrained as gameweeks land. This module measures that instead —
expanding window, retrain every ``step`` gameweeks, predict the next one — and
reports the gap between the two, which is what retraining is worth.

Every baseline is refit under the same protocol on the same rows, so the
comparison is like-for-like. The FCPS calibration in particular is refit at every
origin from that origin's history only; calibrating it once over the whole season
would hand it information the model doesn't get.

Output is a markdown block written to ``ml/reports/backtest.md`` and echoed to
stdout, ready to paste under "Results" in ``PRDs/ml-methodology.md``.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import baselines, config, features as feature_mod, metrics, models, splits, train

DEFAULT_STEP = 2
DEFAULT_START_GW = 8


def run(
    season: str | None = None,
    step: int = DEFAULT_STEP,
    start_gameweek: int = DEFAULT_START_GW,
    model_name: str | None = None,
    refresh: bool = False,
) -> dict:
    import pandas as pd

    season = season or config.TEST_SEASONS[-1]
    frame = train.load_frame(refresh=refresh)

    if model_name is None:
        _model, metadata = models.load()
        model_name = (metadata or {}).get("model", "hgb_poisson")
    print(f"[backtest] {season}, model={model_name}, retrain every {step} GW(s)")

    collected = []
    for train_part, test_part, gameweek in splits.walk_forward(
        frame, season, start_gameweek=start_gameweek, step=step
    ):
        train_part = train_part.copy()
        test_part = test_part.copy()

        # Baselines are refit at this origin, from this origin's history only.
        baselines.add_all(train_part, test_part)

        try:
            model = models.fit(model_name, train_part)
        except Exception as error:
            print(f"  GW{gameweek}: fit failed ({type(error).__name__}), skipped")
            continue

        test_part["pred_model"] = models.predict(model, test_part)
        collected.append(test_part)
        print(
            f"  GW{gameweek:>2}: train {len(train_part):>6,} rows -> "
            f"test {len(test_part):>4,} rows",
            flush=True,
        )

    if not collected:
        raise SystemExit(
            f"No walk-forward folds produced for {season}. Is it in the cached data?"
        )

    pooled = pd.concat(collected, ignore_index=True)
    columns = ["pred_model"] + list(baselines.BASELINE_COLUMNS)
    results = metrics.compare(pooled, columns)

    fcps_rank = metrics.evaluate(pooled, "bl_fcps")
    for key in ("spearman_gw", "precision_at_k", "mean_actual_top_k", "lift_over_mean"):
        results["bl_fcps_points"][key] = fcps_rank[key]

    by_gameweek = _per_gameweek(pooled, columns)
    report = {
        "season": season,
        "model": model_name,
        "protocol": "expanding-window walk-forward",
        "retrain_step": step,
        "start_gameweek": start_gameweek,
        "folds": len(collected),
        "rows": int(len(pooled)),
        "results": results,
        "per_gameweek": by_gameweek,
    }

    config.ensure_dirs()
    (config.REPORT_DIR / "backtest.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    markdown = render(report)
    (config.REPORT_DIR / "backtest.md").write_text(markdown)
    print("\n" + markdown)
    return report


def _per_gameweek(pooled, columns) -> list[dict]:
    """Per-gameweek MAE for every column, so a bad week is visible, not averaged."""
    rows = []
    for gameweek, part in pooled.groupby("GW", sort=True):
        entry = {"gameweek": int(gameweek), "n": int(len(part))}
        for column in columns:
            entry[column] = metrics.mae(
                part[feature_mod.TARGET].to_numpy(dtype="float64"),
                part[column].to_numpy(dtype="float64"),
            )
        rows.append(entry)
    return rows


LABELS = {
    "pred_model": "**ML model**",
    "bl_mean": "Baseline: train mean",
    "bl_ppg": "Baseline: points per game",
    "bl_form": "Baseline: form (last 4 GW)",
    "bl_fcps_points": "Incumbent: FCPS",
    "bl_fpl_xp": "FPL's own expected points",
}


def render(report: dict) -> str:
    results = {LABELS.get(k, k): v for k, v in report["results"].items()}
    lines = [
        f"### Walk-forward backtest — {report['season']}",
        "",
        f"Model: `{report['model']}`. Protocol: {report['protocol']}, retrained every "
        f"{report['retrain_step']} gameweek(s) from GW{report['start_gameweek']}. "
        f"{report['folds']} folds, {report['rows']:,} player-gameweeks scored.",
        "",
        metrics.format_table(results),
        "",
        "Every baseline is refit at each origin from that origin's history only. "
        "FCPS rank metrics use the raw 0-1000 score; its MAE/RMSE use an isotonic "
        "calibration to points fitted on the same history.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=None)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--start-gw", type=int, default=DEFAULT_START_GW)
    parser.add_argument("--model", default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)

    run(
        season=args.season,
        step=args.step,
        start_gameweek=args.start_gw,
        model_name=args.model,
        refresh=args.refresh,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
