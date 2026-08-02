"""The model zoo, and the artifact format.

Four candidates, chosen to span genuinely different inductive biases rather than
to pad a comparison table:

``ridge``
    Linear, imputed and standardised. Not expected to win; it is the control. If
    a gradient-boosted model can't beat a linear one by a clear margin on this
    feature set, the extra complexity isn't earning anything.
``hgb``
    ``HistGradientBoostingRegressor``, squared error. The default choice for
    tabular data of this size, and it consumes NaN natively — which matters here
    because "Opta data didn't exist in this season" is real information that
    imputation would erase.
``hgb_poisson``
    The same, with Poisson deviance. FPL points are a zero-inflated count: the
    modal player-gameweek is 0, the mean is around 2, and the tail runs to 20+.
    Squared error over-weights the tail and drags every prediction upward.
    Poisson deviance is the right likelihood for a count, and this is the
    variant expected to win. Its one wrinkle is that FPL points can be negative
    (own goals, red cards), and Poisson requires a non-negative target, so the
    label is clipped at zero for this model only — costing it accuracy on
    roughly 1% of rows in exchange for a better-specified loss on the other 99%.
``two_stage``
    ``P(plays) x E[points | plays]``. Mirrors how the points are actually
    generated: a player who doesn't feature scores 0 with certainty, and mixing
    those rows into one regression forces a single model to learn both the
    selection problem and the scoring problem at once. Structurally the most
    defensible, and the most likely to be well-calibrated at the top of the
    range, which is where transfer decisions are made.

Whichever wins on the **validation** split is retrained on train+validation and
frozen. The test split is scored exactly once, at the end, and never used to
choose anything.
"""

from __future__ import annotations

import json
from typing import Sequence

from . import config, features as feature_mod

RANDOM_STATE = 20260801

MODEL_NAMES = ("ridge", "hgb", "hgb_poisson", "two_stage")


def build(name: str):
    """Instantiate one candidate by name."""
    if name == "ridge":
        return _ridge()
    if name == "hgb":
        return _hgb(loss="squared_error")
    if name == "hgb_poisson":
        return _PoissonClipped(_hgb(loss="poisson"))
    if name == "two_stage":
        return TwoStage()
    if name == "lgbm":
        return _lgbm()
    raise ValueError(f"unknown model: {name}")


def available() -> tuple[str, ...]:
    """Candidates whose dependencies are actually installed."""
    names = list(MODEL_NAMES)
    try:
        import lightgbm  # noqa: F401

        names.append("lgbm")
    except ImportError:
        pass
    return tuple(names)


def _ridge():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )


def _hgb(loss: str = "squared_error"):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss=loss,
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )


def _lgbm():
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


class _PoissonClipped:
    """Wraps a Poisson-loss regressor, clipping the target at zero to fit it.

    FPL points go negative for own goals and red cards. Rather than dropping
    those rows — which would teach the model that they don't happen — the label
    is floored at 0. The cost is bounded and stated; the benefit is a loss
    function that matches the shape of the data.
    """

    def __init__(self, inner):
        self.inner = inner

    def fit(self, X, y, **kwargs):
        import numpy as np

        self.inner.fit(X, np.clip(np.asarray(y, dtype="float64"), 0.0, None), **kwargs)
        return self

    def predict(self, X):
        return self.inner.predict(X)

    @property
    def feature_importances_(self):
        return getattr(self.inner, "feature_importances_", None)


class TwoStage:
    """``P(plays) x E[points | plays]``.

    "Plays" is defined as recording any minutes, not as starting. A substitute
    who comes on for 20 minutes banks the appearance point and can score, so
    treating him as a non-event would misstate both stages.
    """

    PLAY_THRESHOLD = 0.5

    def __init__(self):
        self.classifier = None
        self.regressor = None

    def fit(self, X, y, minutes=None, **_kwargs):
        import numpy as np
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        y = np.asarray(y, dtype="float64")
        if minutes is None:
            # Without a minutes column, "scored anything at all" is the best
            # available proxy for having played. An appearance is worth at least
            # one point, so the two agree on all but a handful of rows.
            played = (y > 0).astype(int)
        else:
            played = (np.asarray(minutes, dtype="float64") > 0).astype(int)

        self.classifier = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
        self.classifier.fit(X, played)

        mask = played == 1
        if mask.sum() < 100:
            raise ValueError("not enough rows with minutes to fit the second stage")

        self.regressor = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=400,
            learning_rate=0.06,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
        X_played = X[mask] if hasattr(X, "shape") else [x for x, m in zip(X, mask) if m]
        self.regressor.fit(X_played, np.clip(y[mask], 0.0, None))
        return self

    def predict(self, X):
        p_play = self.classifier.predict_proba(X)[:, 1]
        conditional = self.regressor.predict(X)
        return p_play * conditional


# ---------------------------------------------------------------- fit / persist


def fit(name: str, train_frame, feature_names: Sequence[str] | None = None):
    """Fit one candidate on a feature frame. Returns the fitted estimator."""
    feature_names = list(feature_names or feature_mod.FEATURES)
    X = train_frame[feature_names].to_numpy(dtype="float64")
    y = train_frame[feature_mod.TARGET].to_numpy(dtype="float64")

    model = build(name)
    if isinstance(model, TwoStage):
        # `minutes_per_team_game` is the season-to-date rate, not this
        # gameweek's minutes, so it cannot stand in for the play indicator.
        # The panel's own minutes column is passed through when present.
        minutes = (
            train_frame["gw_minutes"].to_numpy(dtype="float64")
            if "gw_minutes" in train_frame.columns
            else None
        )
        model.fit(X, y, minutes=minutes)
    else:
        model.fit(X, y)
    return model


def predict(model, frame, feature_names: Sequence[str] | None = None):
    import numpy as np

    feature_names = list(feature_names or feature_mod.FEATURES)
    X = frame[feature_names].to_numpy(dtype="float64")
    return np.asarray(model.predict(X), dtype="float64")


def save(model, metadata: dict) -> None:
    """Persist the model and its provenance side by side.

    The metadata is not decoration. :func:`load` refuses an artifact whose
    feature list doesn't match the current :data:`ml.features.FEATURES`, which is
    the only thing standing between a reordered feature tuple and silently
    scoring every player on the wrong columns.
    """
    import joblib

    config.ensure_dirs()
    joblib.dump(model, config.artifact_path())
    config.metadata_path().write_text(json.dumps(metadata, indent=2, default=str))


def load():
    """``(model, metadata)``, or ``(None, None)`` if no artifact is present."""
    import joblib

    path = config.artifact_path()
    meta_path = config.metadata_path()
    if not path.exists() or not meta_path.exists():
        return None, None

    metadata = json.loads(meta_path.read_text())
    stored = tuple(metadata.get("features", ()))
    if stored != tuple(feature_mod.FEATURES):
        raise RuntimeError(
            "Model artifact was trained on a different feature set.\n"
            f"  artifact: {stored}\n"
            f"  current:  {tuple(feature_mod.FEATURES)}\n"
            "Retrain with `python -m ml.train`."
        )
    return joblib.load(path), metadata


def importances(model, feature_names: Sequence[str] | None = None) -> dict[str, float]:
    """Best-effort feature importance, for the write-up. Empty when unavailable."""
    feature_names = list(feature_names or feature_mod.FEATURES)
    inner = getattr(model, "inner", model)
    inner = getattr(inner, "regressor", inner)

    values = getattr(inner, "feature_importances_", None)
    if values is None and hasattr(inner, "named_steps"):
        step = inner.named_steps.get("model")
        values = getattr(step, "coef_", None)
    if values is None:
        return {}
    values = list(values)[: len(feature_names)]
    return dict(
        sorted(
            zip(feature_names, (float(abs(v)) for v in values)),
            key=lambda item: -item[1],
        )
    )
