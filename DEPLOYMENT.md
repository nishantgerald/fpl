# Deployment notes

Configuration this app reads, and why each one exists. Nothing here contains a
secret; the values live in `.env` locally (mode 600, gitignored) and in the
host's config vars in production.

## Required in production

| Variable | Why |
|---|---|
| `FPL_APP_VAULT_KEY` | Encrypts stored FPL session cookies. Generated to a file locally, but a hosted filesystem is usually ephemeral — a regenerated key makes every stored cookie permanently unreadable and silently disconnects every user. Generate once with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` and never change it. |
| `FPL_APP_DB` | SQLite path. On an ephemeral filesystem this loses every account on restart; move to Postgres before real traffic. |
| `PUBLIC_BASE_URL` | The origin used to build links in outbound email. Without it, a password-reset link points at whatever host served the request. |
| `TRUST_PROXY_HEADER=true` | Only behind a proxy that rewrites `X-Forwarded-For`. Without it every user shares one rate-limit bucket; with it set on a directly-exposed app, anyone can forge a fresh identity per request. |

## Google sign-in

`GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` — a **Web application**
OAuth client. Authorized redirect URIs must exactly match
`{origin}/api/auth/google/callback` for every origin you serve from. Scopes are
`openid` and `email` only, both non-sensitive, so the consent screen needs no
Google verification review and has no user cap.

## Email (Amazon SES)

`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`,
`SMTP_REPLY_TO`.

Plain SMTP, so any provider works; SES is what this deployment uses.

- The IAM user is **`fpl-companion-ses`**, with an inline policy allowing only
  `ses:SendEmail` and `ses:SendRawEmail`. It can send mail and do nothing else.
- `SMTP_PASSWORD` is *not* the IAM secret key. SES derives it via an HMAC chain
  seeded with a fixed date constant, then region, `ses`, `aws4_request`,
  `SendRawEmail`, prefixed with version byte `0x04`. Starting that chain from
  the region, or using a different version byte, produces a well-formed
  password that fails authentication with `535 Authentication Credentials
  Invalid`.
- Rotating: delete the access key, create a new one, re-derive.

### Sender domain

`us-east-1` already has production access (50,000/day, 14/sec), so there is no
sandbox to escape.

`SMTP_FROM` is currently a verified Gmail address, which works but is a
stopgap: SES cannot produce a valid DKIM signature for `gmail.com`, so those
messages fail alignment checks and land in spam far more often.

`nishantgerald.com` has been created as an SES domain identity. It verifies
once these three CNAMEs exist at the DNS host (Google Domains / Squarespace —
the domain is not in Route 53, so they cannot be added from AWS):

```
4w3knx3ik6a3e6azocyd7wz5ttogree5._domainkey  CNAME  4w3knx3ik6a3e6azocyd7wz5ttogree5.dkim.amazonses.com
tspv2vqixrst3le4gpzkehubdrgsizd4._domainkey  CNAME  tspv2vqixrst3le4gpzkehubdrgsizd4.dkim.amazonses.com
xcngmn267qsc2elfkbvr35jcneexmcld._domainkey  CNAME  xcngmn267qsc2elfkbvr35jcneexmcld.dkim.amazonses.com
```

Then set `SMTP_FROM=noreply@nishantgerald.com`.

## Optional

| Variable | Effect |
|---|---|
| `VISION_MODEL`, `VISION_TIMEOUT_SECONDS` | Screenshot import. Needs the Claude CLI on the host; absent, the feature reports unavailable and the client hides it. |
| `RESET_LINK_TO_LOG=true` | Prints reset links to the server log when mail cannot be sent. Development only — it puts account-recovery links in the log. |
| `IMPORT_ATTEMPTS_PER_WINDOW`, `AUTH_ATTEMPTS_PER_WINDOW` | Throttles. Defaults are 5 and 10 per 15 minutes. |
| `SEASON_LABEL` | Shown in page titles, e.g. `2026/27`. |

## Scheduled work

```
python -m scripts.send_digests --dry-run   # render, send nothing
python -m scripts.send_digests             # send to opted-in users
```

Run a few hours before each deadline. It skips users whose data cannot be read
rather than aborting the run.
