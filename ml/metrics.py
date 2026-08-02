"""Evaluation metrics, chosen for what the model is actually used for.

MAE and RMSE are reported because they are the standard and make this comparable
to other work, but on their own they are a poor guide here. FPL points are
zero-inflated and long-tailed: most player-gameweeks are 0-2 points, so a model
that predicts "2" for everyone scores a respectable MAE while being useless. The
metrics that decide whether this ships are the ranking and decision ones.

``mae`` / ``rmse``
    Standard regression error, over all rows and over the ``started`` subset
    separately. The split matters — a model can look good purely by learning who
    doesn't play.
``spearman_by_gameweek``
    Rank correlation between predicted and actual points, computed *within* each
    gameweek and then averaged. Pooling across gameweeks would let cross-gameweek
    variance (a blank week, a double week) inflate the correlation.
``precision_at_k``
    Of the model's top *k* picks in a gameweek, what fraction land in the actual
    top *k*. This is the shape of the real question: the manager picks from a
    shortlist, not from a distribution.
``mean_actual_top_k``
    The mean points actually scored by the model's top *k*. Denominated in
    points, so it can be compared directly between models and against the
    league mean.
``transfer_call_accuracy``
    The decision metric. Over same-position pairs where the model predicts a
    gap of at least ``margin``, how often was it right about the direction, and
    what was the mean realised gain of acting on it. A model can win on MAE and
    lose here, and here is what a transfer engine consumes.
"""

from __future__ import annotations

from typing import Sequence

DEFAULT_K = 20
DEFAULT_MARGIN = 1.0


def mae(y_true, y_pred) -> float:
    import numpy as np

    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def rmse(y_true, y_pred) -> float:
    import numpy as np

    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")


def spearman(y_true, y_pred) -> float:
    """Rank correlation, implemented directly so scipy stays optional."""
    import numpy as np

    y_true, y_pred = _clean(y_true, y_pred)
    if len(y_true) < 3:
        return float("nan")
    rank_true, rank_pred = _rank(y_true), _rank(y_pred)
    if np.std(rank_true) == 0 or np.std(rank_pred) == 0:
        return float("nan")
    return float(np.corrcoef(rank_true, rank_pred)[0, 1])


def _rank(values):
    """Average ranks, which is what makes this Spearman rather than something else."""
    import numpy as np

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype="float64")
    ranks[order] = np.arange(len(values), dtype="float64")

    sorted_values = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def spearman_by_gameweek(frame, pred_column: str, target: str = "y_points") -> float:
    """Mean within-gameweek rank correlation, weighted by gameweek size."""
    import numpy as np

    scores, weights = [], []
    for _key, rows in frame.groupby(["season", "GW"], sort=True):
        value = spearman(rows[target].to_numpy(), rows[pred_column].to_numpy())
        if not np.isnan(value):
            scores.append(value)
            weights.append(len(rows))
    if not scores:
        return float("nan")
    return float(np.average(scores, weights=weights))


def precision_at_k(
    frame, pred_column: str, k: int = DEFAULT_K, target: str = "y_points"
) -> dict:
    """Top-*k* overlap and realised points, averaged over gameweeks."""
    import numpy as np

    overlaps, realised, baselines = [], [], []
    for _key, rows in frame.groupby(["season", "GW"], sort=True):
        if len(rows) < k * 2:
            continue
        picked = rows.nlargest(k, pred_column)
        actual = set(rows.nlargest(k, target).index)
        overlaps.append(len(set(picked.index) & actual) / k)
        realised.append(float(picked[target].mean()))
        baselines.append(float(rows[target].mean()))

    if not overlaps:
        return {"precision_at_k": float("nan"), "mean_actual_top_k": float("nan"),
                "lift_over_mean": float("nan"), "gameweeks": 0}
    return {
        "precision_at_k": float(np.mean(overlaps)),
        "mean_actual_top_k": float(np.mean(realised)),
        "lift_over_mean": float(np.mean(realised) - np.mean(baselines)),
        "gameweeks": len(overlaps),
    }


def transfer_call_accuracy(
    frame,
    pred_column: str,
    target: str = "y_points",
    margin: float = DEFAULT_MARGIN,
    max_pairs_per_gameweek: int = 4000,
    seed: int = 20260801,
) -> dict:
    """How often a predicted same-position upgrade actually was one.

    Pairs are drawn within (gameweek, position), because that is the only swap
    FPL's squad quotas permit. Only pairs where the model claims a gap of at
    least ``margin`` points are counted — below that the model isn't making a
    call, and scoring it on coin-flips understates it.

    ``seed`` is fixed so two runs of the backtest produce the same number.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    calls = 0
    correct = 0
    realised_gain = 0.0
    predicted_gain = 0.0

    for _key, rows in frame.groupby(["season", "GW", "position"], sort=True):
        n = len(rows)
        if n < 4:
            continue
        pred = rows[pred_column].to_numpy(dtype="float64")
        actual = rows[target].to_numpy(dtype="float64")

        pairs = min(max_pairs_per_gameweek, n * (n - 1) // 2)
        left = rng.integers(0, n, size=pairs)
        right = rng.integers(0, n, size=pairs)
        keep = left != right
        left, right = left[keep], right[keep]

        gap = pred[left] - pred[right]
        decisive = np.abs(gap) >= margin
        if not decisive.any():
            continue
        left, right, gap = left[decisive], right[decisive], gap[decisive]

        # Orient every pair so the model is always claiming left > right.
        flip = gap < 0
        left[flip], right[flip] = right[flip], left[flip]
        gap = np.abs(gap)

        outcome = actual[left] - actual[right]
        calls += len(gap)
        correct += int(np.sum(outcome > 0))
        realised_gain += float(np.sum(outcome))
        predicted_gain += float(np.sum(gap))

    if not calls:
        return {"transfer_calls": 0, "accuracy": float("nan"),
                "mean_realised_gain": float("nan"), "mean_predicted_gain": float("nan")}
    return {
        "transfer_calls": calls,
        "accuracy": correct / calls,
        "mean_realised_gain": realised_gain / calls,
        "mean_predicted_gain": predicted_gain / calls,
    }


def evaluate(
    frame,
    pred_column: str,
    target: str = "y_points",
    k: int = DEFAULT_K,
    margin: float = DEFAULT_MARGIN,
) -> dict:
    """Every metric for one prediction column, as a flat dict."""
    y_true = frame[target].to_numpy(dtype="float64")
    y_pred = frame[pred_column].to_numpy(dtype="float64")

    started = frame[frame.get("start_rate", 0) > 0.5] if "start_rate" in frame else frame

    result = {
        "n": int(len(frame)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae_starters": mae(
            started[target].to_numpy(dtype="float64"),
            started[pred_column].to_numpy(dtype="float64"),
        ),
        "spearman_gw": spearman_by_gameweek(frame, pred_column, target),
    }
    result.update(precision_at_k(frame, pred_column, k, target))
    result.update(
        {f"transfer_{k2}": v
         for k2, v in transfer_call_accuracy(frame, pred_column, target, margin).items()}
    )
    return result


def compare(frame, columns: Sequence[str], **kwargs) -> dict[str, dict]:
    """Evaluate several prediction columns on the same rows."""
    return {column: evaluate(frame, column, **kwargs) for column in columns}


def format_table(results: dict[str, dict]) -> str:
    """A markdown table, for pasting straight into the methodology document."""
    if not results:
        return "_(no results)_"
    columns = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("mae_starters", "MAE (starters)"),
        ("spearman_gw", "Spearman (per GW)"),
        ("precision_at_k", "P@20"),
        ("mean_actual_top_k", "Actual pts, top 20"),
        ("transfer_accuracy", "Transfer call acc."),
        ("transfer_mean_realised_gain", "Realised gain/call"),
    ]
    header = "| Model | " + " | ".join(label for _key, label in columns) + " |"
    divider = "|---" * (len(columns) + 1) + "|"
    lines = [header, divider]
    for name, values in results.items():
        cells = []
        for key, _label in columns:
            value = values.get(key)
            cells.append("—" if value is None else f"{value:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _clean(y_true, y_pred):
    import numpy as np

    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return y_true[mask], y_pred[mask]
