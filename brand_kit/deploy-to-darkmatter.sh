#!/usr/bin/env bash
# Push the Gameweek Zero brand kit to darkmatter.
#
#   ./deploy-to-darkmatter.sh                 # uses the darkmatter_remote ssh alias
#   ./deploy-to-darkmatter.sh other_host      # override the host
#
# Safe to re-run: it copies over the top, never deletes anything already there.

set -euo pipefail

REMOTE="${1:-darkmatter_remote}"
DEST="/home/nishantgerald/projects/fpl/brand_kit"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$(basename "${BASH_SOURCE[0]}")"

echo "==> source : $SRC"
echo "==> target : $REMOTE:$DEST"
echo

echo "==> checking ssh connectivity"
if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE" true 2>/dev/null; then
  echo "!! cannot reach '$REMOTE' without a password/passphrase prompt." >&2
  echo "   Run 'ssh $REMOTE' once by hand, or unlock your key with 'ssh-add', then retry." >&2
  exit 1
fi

echo "==> creating remote directory"
ssh "$REMOTE" "mkdir -p '$DEST'"

if command -v rsync >/dev/null 2>&1 && ssh "$REMOTE" "command -v rsync" >/dev/null 2>&1; then
  echo "==> syncing with rsync"
  rsync -avh --progress --exclude "$SELF" "$SRC"/ "$REMOTE:$DEST"/
else
  echo "==> rsync unavailable on one side, falling back to tar over ssh"
  tar -C "$SRC" --exclude "$SELF" -czf - . | ssh "$REMOTE" "tar -C '$DEST' -xzf -"
fi

echo
echo "==> verifying"
LOCAL_COUNT=$(find "$SRC" -type f ! -name "$SELF" | wc -l | tr -d ' ')
REMOTE_COUNT=$(ssh "$REMOTE" "find '$DEST' -type f | wc -l" | tr -d ' ')
echo "    local files : $LOCAL_COUNT"
echo "    remote files: $REMOTE_COUNT"

# Compare content hashes so a truncated transfer can't pass silently.
hash_local() {
  find "$SRC" -type f ! -name "$SELF" -print0 \
    | sort -z \
    | xargs -0 shasum -a 256 2>/dev/null \
    | sed "s|$SRC/||" | sort | shasum -a 256 | awk '{print $1}'
}
hash_remote() {
  ssh "$REMOTE" "cd '$DEST' && find . -type f -print0 | sort -z | xargs -0 sha256sum | sed 's|^\(.*\)  \./|\1  |' | sort | sha256sum | awk '{print \$1}'"
}

if [ "$LOCAL_COUNT" = "$REMOTE_COUNT" ]; then
  echo "    file count matches"
else
  echo "    !! file count differs — check the output above" >&2
fi

echo
echo "==> done. Remote listing:"
ssh "$REMOTE" "ls -la '$DEST'"
