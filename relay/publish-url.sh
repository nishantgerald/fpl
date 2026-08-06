#!/usr/bin/env bash
# Tell Heroku where the relay currently is.
#
# A Cloudflare quick tunnel needs no account, but its hostname is random and
# changes every time it restarts — so a URL set once by hand goes stale on the
# next reboot, and the model features quietly switch off with no error anywhere.
#
# This reads the current hostname and, if a Heroku token is available, pushes it
# straight to the app over the Platform API. No Heroku CLI required.
#
# Config, all optional except the token:
#   ~/.config/fpl-relay/heroku.env   HEROKU_API_KEY=...   HEROKU_APP=ng-fpl
#
# Without a token it just prints the URL and the command to run by hand, so it
# is still useful — it never fails silently.

set -uo pipefail

CONFIG_DIR="${FPL_RELAY_CONFIG_DIR:-$HOME/.config/fpl-relay}"
TUNNEL_LOG="$CONFIG_DIR/tunnel.log"
HEROKU_ENV="$CONFIG_DIR/heroku.env"

# The tunnel prints its hostname a few seconds after it starts, so poll rather
# than read once and conclude it has no URL.
url=""
for _ in $(seq 1 30); do
  url=$(grep -ao 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | tail -1)
  [ -n "$url" ] && break
  sleep 2
done

if [ -z "$url" ]; then
  echo "publish-url: no tunnel hostname found in $TUNNEL_LOG" >&2
  exit 1
fi
echo "publish-url: relay is at $url"

if [ ! -r "$HEROKU_ENV" ]; then
  cat >&2 <<EOF
publish-url: no $HEROKU_ENV, so nothing was published.

To let this update itself, create that file with:
  HEROKU_API_KEY=<from https://dashboard.heroku.com/account>
  HEROKU_APP=ng-fpl

Or set it by hand this once:
  heroku config:set --app ng-fpl LLM_RELAY_URL=$url
EOF
  exit 0
fi

# shellcheck disable=SC1090
. "$HEROKU_ENV"
: "${HEROKU_APP:=ng-fpl}"

if [ -z "${HEROKU_API_KEY:-}" ]; then
  echo "publish-url: HEROKU_API_KEY is empty in $HEROKU_ENV" >&2
  exit 1
fi

status=$(curl -s -o /tmp/fpl-relay-publish.out -w '%{http_code}' \
  -X PATCH "https://api.heroku.com/apps/$HEROKU_APP/config-vars" \
  -H "Accept: application/vnd.heroku+json; version=3" \
  -H "Authorization: Bearer $HEROKU_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"LLM_RELAY_URL\":\"$url\"}")

if [ "$status" = "200" ]; then
  echo "publish-url: Heroku now points at $url"
else
  echo "publish-url: Heroku refused the update (HTTP $status)" >&2
  head -c 300 /tmp/fpl-relay-publish.out >&2
  echo >&2
  exit 1
fi
