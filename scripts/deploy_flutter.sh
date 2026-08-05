#!/bin/bash
# Builds the Flutter web app and copies it into the Flask project for deployment.
# Run from the repo root: bash scripts/deploy_flutter.sh
#
# Both paths below were stale and the script failed on its first line: the
# Flutter source moved to `fpl-old`, and the SDK is in /opt rather than
# ~/development. Resolved relative to this script and overridable by
# environment, so a move or a different machine doesn't require an edit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASK_PROJECT="${FLASK_PROJECT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
FLUTTER_PROJECT="${FLUTTER_PROJECT:-$(cd "$FLASK_PROJECT/../fpl-old" 2>/dev/null && pwd || echo "")}"
FLUTTER_BIN="${FLUTTER_BIN:-/opt/flutter/bin/flutter}"
TARGET_DIR="$FLASK_PROJECT/flutter_web"

if [ ! -x "$FLUTTER_BIN" ]; then
  echo "Flutter SDK not found at $FLUTTER_BIN — set FLUTTER_BIN." >&2
  exit 1
fi
if [ -z "$FLUTTER_PROJECT" ] || [ ! -f "$FLUTTER_PROJECT/pubspec.yaml" ]; then
  echo "No Flutter project found — set FLUTTER_PROJECT to the pubspec directory." >&2
  exit 1
fi

echo "🏗  Building Flutter web app from $FLUTTER_PROJECT ..."
cd "$FLUTTER_PROJECT"
"$FLUTTER_BIN" build web --no-tree-shake-icons --base-href /app/

echo "📦 Copying build output to $TARGET_DIR ..."
# Staged then swapped: a failed copy must not leave the served directory empty.
rm -rf "$TARGET_DIR.new"
cp -r "$FLUTTER_PROJECT/build/web" "$TARGET_DIR.new"
rm -rf "$TARGET_DIR"
mv "$TARGET_DIR.new" "$TARGET_DIR"

echo "✅ Done. Served at /app/"
