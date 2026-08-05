#!/usr/bin/env bash
# Refresh the FPL research digest. Intended for cron.
#
# Kept as a script rather than inlined into the crontab so the schedule and the
# command can change independently, and so the environment is explicit: cron
# runs with a near-empty PATH, which would otherwise hide the Claude CLI that
# `engine.research` shells out to.
set -euo pipefail

PROJECT_DIR="${FPL_DIR:-$HOME/projects/fpl}"
cd "$PROJECT_DIR"

# The CLI authenticates as this user; cron's default PATH does not include it.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

exec "$PROJECT_DIR/.venv/bin/python" -m engine.research
