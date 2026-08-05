"""Serve-time inference: a live bootstrap payload in, points per gameweek out.

This is the other half of the parity contract in :mod:`ml.features`. The same
feature names, in the same order, built from the live API instead of from the
archive. If this file and ``build_frame`` ever disagree about what a feature
means, the model is scoring production on a different quantity than it was
trained on, and nothing downstream will notice — which is why
``tests/test_ml_features.py`` asserts they agree on a shared fixture.

Two adjustments are applied *after* the model, deliberately rather than as
features:

**Availability.** The archive carries no injury flags, so the model cannot learn
them. The live API publishes ``status`` and ``chance_of_playing_next_round``,
which are far better evidence than anything a model would infer. Predictions are
therefore conditional on fitness and multiplied by the same availability factor
:mod:`engine.xpts` uses — one definition, both engines.

**Blank gameweeks.** ``n_fixtures = 0`` is a feature, so the model has seen
blanks in training and should already predict near zero. It is nevertheless
floored to exactly zero, because "near zero" in a squad of fifteen still adds up
to a phantom point or two, and a player whose team is not playing scores nothing
with certainty rather than in expectation.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import features as feature_mod, models


class Predictor:
    """A loaded artifact, ready to score live players."""

    def __init__(self, model, metadata: Mapping):
        self.model = model
        self.metadata = dict(metadata)
        self.feature_names = list(metadata.get("features", feature_mod.FEATURES))

    # -------------------------------------------------------------- loading

    @classmethod
    def load(cls) -> "Predictor | None":
        """Load the persisted model, or ``None`` if there isn't one.

        Returning ``None`` rather than raising is the point: a deployment with no
        trained artifact must fall back to :mod:`engine.xpts` and keep serving,
        not 500. A *corrupt* or feature-mismatched artifact still raises, because
        that is a deployment error rather than an absence.
        """
        try:
            model, metadata = models.load()
        except ImportError:
            return None
        if model is None:
            return None
        return cls(model, metadata or {})

    @property
    def name(self) -> str:
        return str(self.metadata.get("model", "unknown"))

    @property
    def trained_on(self) -> list[str]:
        return list(self.metadata.get("train_seasons", [])) + list(
            self.metadata.get("valid_seasons", [])
        )

    # -------------------------------------------------------------- scoring

    def score(
        self,
        elements: Sequence[Mapping],
        fixture_index: Mapping[int, Mapping[int, Sequence[Mapping]]],
        gameweeks: Sequence[int],
        team_games: int,
        use_fpl_xp: bool = True,
    ) -> dict[int, dict[int, float]]:
        """``player_id -> {gameweek: predicted_points}``, before availability.

        One batched call per gameweek rather than one per player: with ~700
        players over an 8-gameweek horizon that is 8 predict calls instead of
        5,600, which is the difference between a fast endpoint and a timeout.
        """
        import numpy as np

        by_player: dict[int, dict[int, float]] = {
            int(e["id"]): {} for e in elements if _is_player(e)
        }

        for gameweek in gameweeks:
            rows = []
            ids = []
            for element in elements:
                if not _is_player(element):
                    continue
                context = feature_mod.fixture_context_for(
                    int(element.get("team", 0)), gameweek, fixture_index
                )
                built = feature_mod.from_bootstrap(
                    element, team_games, context, use_fpl_xp=use_fpl_xp
                )
                rows.append([built[name] for name in self.feature_names])
                ids.append(int(element["id"]))

            if not rows:
                continue

            matrix = np.asarray(rows, dtype="float64")
            predictions = np.asarray(self.model.predict(matrix), dtype="float64")
            # A regressor is unbounded; a gameweek score is not meaningfully
            # negative for our purposes, and a negative projection would let the
            # optimiser "gain" points by benching someone.
            predictions = np.clip(predictions, 0.0, None)

            blank = matrix[:, self.feature_names.index("n_fixtures")] <= 0
            predictions[blank] = 0.0

            # A player with no minutes has every performance feature at zero, so
            # the only informative input left is `fpl_xp` — FPL's own `ep_next`.
            # The model leans on it hard: zeroing that one feature drops such a
            # prediction from ~9 to ~0. Between seasons, when `ep_next` is a
            # placeholder rather than a projection, that produces confident
            # scores for players who have never kicked a ball — a £46m signing
            # and an unused academy squad number score alike, and both outrank a
            # 209-point defender.
            #
            # The model cannot distinguish "rate of zero" from "no observation":
            # `_safe_div` yields 0.0 for both in training and here, so there is
            # no missingness for the tree to learn from. Rather than invent a
            # signal, the projection is capped at FPL's own number. That is a
            # weak estimate, but it is an *honest* one, and it stops the
            # optimiser preferring unknowns to proven players.
            if "minutes_per_team_game" in self.feature_names:
                unseen = matrix[:, self.feature_names.index("minutes_per_team_game")] <= 0
                if "fpl_xp" in self.feature_names:
                    fallback = matrix[:, self.feature_names.index("fpl_xp")]
                    fallback = np.nan_to_num(fallback, nan=0.0)
                    predictions[unseen] = np.minimum(predictions[unseen], fallback[unseen])
                else:
                    predictions[unseen] = 0.0

            for pid, value in zip(ids, predictions):
                by_player[pid][gameweek] = float(value)

        return by_player


def _is_player(element: Mapping) -> bool:
    return int(element.get("element_type", 0)) in (1, 2, 3, 4)


# A process-wide singleton. Loading a joblib artifact costs tens of milliseconds
# and it is immutable once loaded, so doing it per request would be pure waste.
_PREDICTOR: Predictor | None = None
_LOADED = False


def get_predictor() -> Predictor | None:
    global _PREDICTOR, _LOADED
    if not _LOADED:
        try:
            _PREDICTOR = Predictor.load()
        except Exception as error:  # a mismatched artifact must not take the app down
            print(f"[ml] predictor unavailable: {type(error).__name__}: {error}")
            _PREDICTOR = None
        _LOADED = True
    return _PREDICTOR


def reset() -> None:
    """Drop the cached predictor. For tests and for a post-deploy model swap."""
    global _PREDICTOR, _LOADED
    _PREDICTOR = None
    _LOADED = False
