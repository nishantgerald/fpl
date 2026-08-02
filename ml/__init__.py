"""A trained expected-points model for FPL.

This package is the machine-learning half of the recommendation system. It is
deliberately separate from :mod:`engine`:

* ``engine`` is pure, dependency-light and runs inside a web request.
* ``ml`` owns the offline lifecycle — download, assemble, train, backtest — and
  depends on numpy/pandas/scikit-learn, which the web process does not need.

The two meet at exactly one seam: :mod:`ml.predict` turns a live bootstrap
payload into per-player point predictions using a persisted artifact, and
:mod:`engine.ml_scorer` adapts those into the projection shape the optimiser
already consumes. Nothing else in ``engine`` knows this package exists, and if
the artifact is missing the server falls back to :mod:`engine.xpts` silently.

Read ``PRDs/ml-methodology.md`` before changing anything in here. The two rules
that matter:

1. **No lookahead.** A feature for gameweek *g* may only use data observable
   before the gameweek *g* deadline. :mod:`ml.features` is the only place
   features are defined, and :mod:`tests.test_ml_features` asserts the boundary.
2. **Feature parity.** Every feature must be computable two ways — from
   historical per-gameweek rows, and from a live bootstrap payload — and both
   paths must agree. A model trained on features it can't see at serve time is
   a backtest, not a product.
"""

__all__ = ["config", "features", "metrics", "splits"]
