#!/bin/bash
# Builds the Flutter web app and copies it into the Flask project for deployment.
# Run from the repo root: bash scripts/deploy_flutter.sh

set -e

FLUTTER_PROJECT="../fpl_flutter"
FLASK_PROJECT="$(pwd)"
FLUTTER_BIN="$HOME/development/flutter/bin/flutter"
TARGET_DIR="$FLASK_PROJECT/flutter_web"

echo "🏗  Building Flutter web app..."
cd "$FLUTTER_PROJECT"
"$FLUTTER_BIN" build web --no-tree-shake-icons --base-href /app/

echo "📦 Copying build output to Flask project..."
rm -rf "$TARGET_DIR"
cp -r "$FLUTTER_PROJECT/build/web" "$TARGET_DIR"

echo "✅ Done! Flutter app is at $TARGET_DIR"
echo "   It will be served at: https://fpl.nishantgerald.com/app"
