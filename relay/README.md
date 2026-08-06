# LLM relay

Production has no Claude CLI, so every model-backed feature is off there:
screenshot import, the FCPS column, draft rationale, narrative and research.
Rather than pay a second API bill, the dyno borrows the CLI on the machine that
already has a subscription.

```
Heroku dyno ──signed request──▶ Cloudflare Tunnel ──▶ relay/server.py ──▶ claude
   holds the private key                                holds only the public key
```

## Why signed, not a shared token

A bearer token is a secret both ends hold, so reading either end's disk is
enough to call the relay. Ed25519 splits that: the dyno signs, the relay only
verifies, and the relay's copy of the key is useless to anyone who takes it.

Each request is signed over `timestamp.nonce.sha256(body)` — the body digest is
in there so that catching one request doesn't let anyone swap the prompt inside
it. Requests older than 60s are refused, and each nonce is accepted once.

## What it will and won't run

The relay is on the public internet and the app signing for it accepts
anonymous uploads, so a valid signature proves the caller is our dyno — not
that our dyno hasn't been talked into something. Therefore:

- **Flags are allowlisted.** Anything not in `_ALLOWED_FLAGS` is refused.
- **`Bash`, `Write`, `Edit`, `WebFetch`, `WebSearch` and `Task` can never be
  enabled**, whatever `--allowedTools` asks for.
- **The binary is never taken from the caller.** The relay uses its own.
- **`--add-dir` is rewritten** to a per-call temp directory. Honouring a path
  from the far end would be the whole vulnerability.
- **Attachments are basenamed**, so `../../.ssh/id_rsa` lands in the scratch
  directory like everything else.

## Setup

**1. Generate the key pair** (once):

```bash
python relay/keygen.py
```

**2. Run the relay** on the machine with the CLI:

```bash
LLM_RELAY_PUBLIC_KEY=<public half> python relay/server.py
```

It refuses to start without a key or without a `claude` on PATH — a relay that
401s or 500s every call is worse than one that isn't running.

**3. Expose it.** A Cloudflare Tunnel needs no inbound ports and no router
config:

```bash
cloudflared tunnel --url http://127.0.0.1:8765
```

The quick tunnel above prints a random `*.trycloudflare.com` hostname, and it
**changes on every restart** — so a URL set once by hand goes stale on the next
reboot and the model features switch off with no error anywhere. `publish-url.sh`
closes that: it reads the current hostname and PATCHes it straight to the Heroku
Platform API (no Heroku CLI needed), and the tunnel unit runs it on every start.

Give it a token once, in `~/.config/fpl-relay/heroku.env`:

```
HEROKU_API_KEY=<dashboard.heroku.com/account → Reveal API Key>
HEROKU_APP=ng-fpl
```

Without that file it just prints the URL and the command to run by hand, so it
never fails silently. For a hostname that never moves at all, a named tunnel
against a free Cloudflare account works too — but it needs a domain on
Cloudflare DNS, and this one is on Squarespace.

**4. Point production at it:**

```bash
heroku config:set --app ng-fpl \
  LLM_RELAY_URL=https://<your-tunnel-hostname> \
  LLM_RELAY_PRIVATE_KEY='<private half>'
```

`/api/import/config` flips to `{"screenshot": true}` and the upload button
appears by itself.

## When the machine is asleep

The relay only exists while that machine is awake and the tunnel is up. An
unreachable relay raises `OSError`, which every caller already treats as "no
model available" — so the features hide themselves rather than erroring, which
is the same behaviour as today. Nothing else in the app degrades.

## Keeping it running

```bash
# systemd --user unit, so it survives logout and restarts on failure
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/fpl-relay.service <<'UNIT'
[Unit]
Description=FPL LLM relay
After=network-online.target

[Service]
Environment=LLM_RELAY_PUBLIC_KEY=<public half>
WorkingDirectory=%h/projects/fpl
ExecStart=%h/projects/fpl/.venv/bin/python relay/server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
UNIT
systemctl --user enable --now fpl-relay
loginctl enable-linger "$USER"   # so it runs when you're not logged in
```
