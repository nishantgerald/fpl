#!/usr/bin/env bash
# Refresh the historical archive, the player priors and the trained model.
# Intended for cron, weekly.
#
# Three things go stale on different clocks and are refreshed together because
# they all depend on the same download:
#
#   1. The archive        — a new season's rows appear as it is played.
#   2. The player priors  — engine/data/player_priors.json, the per-90 rates the
#                           projection regresses toward. Shipped in the repo
#                           because Heroku does not run this.
#   3. The trained model  — refit, and deployed only if it beats the incumbent.
#
# Nothing here pushes or deploys. The artifacts land in the working tree and a
# human commits them, because replacing the model that answers every projection
# is not a thing a cron job should do unattended. The log says what changed.
set -euo pipefail

PROJECT_DIR="${FPL_DIR:-$HOME/projects/fpl}"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
PY="$PROJECT_DIR/.venv/bin/python"

echo "=== $(date -Is) refresh-model ==="

# The season the priors currently describe, so the log can say whether this
# changed anything.
before="$("$PY" - <<'EOF'
import json, pathlib
p = pathlib.Path("engine/data/player_priors.json")
try:
    d = json.loads(p.read_text())
    print(f"{d.get('season')} ({len(d.get('players') or {})} players)")
except Exception:
    print("absent")
EOF
)"
echo "[priors] before: $before"

echo "[archive] downloading any new season rows"
"$PY" -m ml.sources

# The most recently completed season is what the priors should describe. A
# season still being played is not a prior; it is the thing being predicted.
season="$("$PY" - <<'EOF'
from ml import config
print(sorted(config.SEASONS)[-1])
EOF
)"
echo "[priors] rebuilding from $season"
"$PY" -m scripts.build_player_priors "$season"

echo "[train] refitting; the gate decides whether it deploys"
"$PY" -m ml.train --refresh --report train_report.json

echo "[git] what changed:"
git status --porcelain -- engine/data ml/artifacts ml/reports || true

echo "=== $(date -Is) done ==="
