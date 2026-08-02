"""Comparators. A model that isn't beating these isn't worth deploying.

Five, in increasing order of how hard they are to beat:

``mean``
    The training-set mean, for every player, every week. The floor. Any model
    below this is broken, not weak.
``ppg``
    The player's season-to-date points per game. What "he's been good this
    season" amounts to numerically.
``form``
    The player's mean points over the last four gameweeks — FPL's own ``form``
    field. This is what most managers actually use, and it is a genuinely strong
    baseline over short horizons.
``fcps``
    This app's incumbent. Reconstructed from as-of season-to-date totals with
    the exact weights in :mod:`engine.fcps`, then calibrated to points (see
    below). Beating it is the specific claim this package needs to support.
``fpl_xp``
    FPL's own published expected points for the gameweek. Produced by the people
    with the proprietary data, and the hardest of the five.

**Calibrating FCPS.** FCPS is a unitless 0-1000 score, so it cannot be compared
on MAE without a mapping into points. The mapping is fit by isotonic regression
**on the training split only** and then applied frozen to validation and test.
Isotonic rather than linear because only FCPS's *ordering* claims to be
meaningful, and isotonic is the most flexible transform that preserves it —
which means FCPS is given the most favourable points-scale reading available,
rather than being handicapped by an arbitrary linear rescale. Its rank metrics
(Spearman, P@k) are computed on the raw score and are unaffected by any of this.
"""

from __future__ import annotations

from . import features as feature_mod

# Kept in sync with engine.fcps.LEGACY_WEIGHTS. Duplicated rather than imported
# so that this package has no import-time dependency on the web engine; the test
# suite asserts the two are identical.
FCPS_WEIGHTS = {
    "total_points_weight": 0.20,
    "form_weight": 0.40,
    "fdr_weight": 0.25,
    "ict_index_weight": 0.15,
}
FCPS_MAX_NEXT_3_FDR = 15.0

BASELINE_COLUMNS = ("bl_mean", "bl_ppg", "bl_form", "bl_fcps_points", "bl_fpl_xp")


def add_simple(train, *frames) -> None:
    """Attach the mean, ppg and form baselines to every frame, in place."""
    train_mean = float(train[feature_mod.TARGET].mean())
    for frame in (train,) + frames:
        frame["bl_mean"] = train_mean
        frame["bl_ppg"] = frame["points_per_game"]
        frame["bl_form"] = frame["form"]
        frame["bl_fpl_xp"] = frame["fpl_xp"]


def fcps_raw(frame):
    """FCPS, on its native 0-1000 scale, from as-of season-to-date quantities.

    Normalisation divisors are the league maxima **within each (season,
    gameweek)** — which is what the live app does, since it normalises against
    the current bootstrap. Computing them over the whole panel instead would leak
    the end of the season into gameweek 6.
    """
    import numpy as np
    import pandas as pd

    df = frame
    parts = []
    for _key, rows in df.groupby(["season", "GW"], sort=False):
        total_points = rows["bl_total_points"].fillna(0.0)
        form = rows["form"].fillna(0.0)
        ict = rows["bl_ict_index"].fillna(0.0)
        fdr = rows["bl_next3_fdr"]

        def divisor(series):
            top = float(series.max())
            return top if top > 0 else 1.0

        fdr_norm = fdr / FCPS_MAX_NEXT_3_FDR
        fdr_term = (1.0 - fdr_norm).clip(lower=0.0, upper=1.0)
        # A season with no fixture file cached has no FDR term; neutralise it at
        # the weighted mean rather than dropping the row, so coverage is
        # identical across every model in the comparison table.
        fdr_term = fdr_term.fillna(fdr_term.mean() if fdr_term.notna().any() else 0.5)

        score = (
            FCPS_WEIGHTS["total_points_weight"] * (total_points / divisor(total_points))
            + FCPS_WEIGHTS["form_weight"] * (form / divisor(form))
            + FCPS_WEIGHTS["fdr_weight"] * fdr_term
            + FCPS_WEIGHTS["ict_index_weight"] * (ict / divisor(ict))
        ) * 1000.0
        parts.append(pd.Series(score.to_numpy(), index=rows.index))

    if not parts:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(parts).reindex(df.index)


def add_fcps(train, *frames) -> dict:
    """Attach ``bl_fcps`` (raw) and ``bl_fcps_points`` (calibrated) to each frame.

    The calibrator is fit on ``train`` only. Returns a small dict describing the
    fit, for the methodology write-up.
    """
    import numpy as np

    for frame in (train,) + frames:
        frame["bl_fcps"] = fcps_raw(frame)

    x = train["bl_fcps"].to_numpy(dtype="float64")
    y = train[feature_mod.TARGET].to_numpy(dtype="float64")
    mask = ~(np.isnan(x) | np.isnan(y))

    if mask.sum() < 100:
        for frame in (train,) + frames:
            frame["bl_fcps_points"] = float(np.nanmean(y))
        return {
            "calibrator": "constant",
            "n": int(mask.sum()),
            "fitted_on": "train split only",
            "note": "insufficient train rows (<100) for isotonic/linear fit; used train mean",
        }

    try:
        from sklearn.isotonic import IsotonicRegression

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(x[mask], y[mask])
        predict = calibrator.predict
        kind = "isotonic"
    except ImportError:  # pragma: no cover - sklearn is a hard dependency of training
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        predict = lambda values: slope * values + intercept  # noqa: E731
        kind = "linear"

    for frame in (train,) + frames:
        values = frame["bl_fcps"].to_numpy(dtype="float64")
        filled = np.where(np.isnan(values), np.nanmedian(x[mask]), values)
        frame["bl_fcps_points"] = predict(filled)

    return {
        "calibrator": kind,
        "n": int(mask.sum()),
        "fitted_on": "train split only",
        "note": (
            "Rank metrics for FCPS use the raw 0-1000 score; the calibration "
            "affects MAE/RMSE only and is monotone, so it cannot change P@k or "
            "Spearman."
        ),
    }


def add_all(train, *frames) -> dict:
    """Attach every baseline. Returns the calibration report."""
    add_simple(train, *frames)
    return add_fcps(train, *frames)
