# FPL Companion — API

Flask backend for [fpl.nishantgerald.com](https://fpl.nishantgerald.com). Serves a
deterministic Fantasy Premier League recommendation engine plus the compiled Flutter
web client at `/app`.

Everything is derived from FPL's public, unauthenticated API. There is no database,
no login, and no access to your FPL credentials.

---

## The engine

The headline feature is transfer advice that is **provably legal** — it satisfies
FPL's squad rules by construction and re-verifies every plan before returning it.
It's deterministic: the same squad gets the same advice every time.

```
engine/
  rules.py           squad quotas, 3-per-club, formations, best XI, legality checks
  money.py           integer-tenths money, selling-price estimation, budget checks
  free_transfers.py  derives banked free transfers from public transfer history
  xpts.py            hand-built expected-points model
  ml_scorer.py       adapts the trained model in ml/ into the same projection shape
  fcps.py            the 0-1000 composite score, restored
  fcps_llm.py        the FCPS transfer column, written by a language model
  optimizer.py       candidate pool, staged search, plan verification
  captain.py         armband ranking + flagged-captain warnings
  prices.py          price-change prediction
  ticker.py          fixture grid + swing detection
  fpl_client.py      all upstream I/O, behind a TTL cache
  narrative.py       optional LLM prose over an already-decided plan (off by default)
  service.py         assembles engine output into API responses

ml/                  offline: download, assemble, train, backtest. Not imported
                     by the web process unless an artifact exists.
```

### Three scores, one rules engine

| | `xpts` | `ml` | `fcps` |
|---|---|---|---|
| What | hand-built component model | gradient-boosted, trained on 6 seasons | 0–1000 composite of points/form/FDR/ICT |
| Unit | points | points | unitless |
| Prices a −4 hit? | yes | yes | **no** |
| Feeds the optimiser? | yes | yes | no |
| Dependencies | none | numpy + scikit-learn + a trained artifact | none |

`xpts` and `ml` are interchangeable inputs to the same optimiser — `ml_scorer`
returns projections in exactly the shape `xpts.project_all` produces, so
`optimizer.optimise` is unchanged and can't tell which produced its inputs. An
ML-scored plan therefore passes the same legality gate as an xPts-scored one.
`?engine=blend` averages them.

**FCPS is not one of them.** It ranks players but isn't in points, so it can't
answer "is this worth a −4". It drives its own written-advice endpoint instead.
See `PRDs/prd-fcps-restoration.md`.

Everything except `fpl_client.py` and `narrative.py` is a pure function of plain
dicts — no network, no clock, no randomness — so the whole decision path is testable
offline.

### What it enforces

| Rule | How |
|---|---|
| 15-player squad, 2 GKP / 5 DEF / 5 MID / 3 FWD | Read from `element_types[].squad_select`, checked on every plan |
| Max 3 players per club | Read from `game_settings.squad_team_limit`, checked on every plan |
| Budget = bank + selling price of players sold | Integer tenths throughout; no float touches a budget comparison |
| 50% sell-on tax | Estimated from `last_deadline_value` minus bank; errs conservative |
| Valid formation (1 GKP, 3–5 DEF, 2–5 MID, 1–3 FWD) | Eight legal shapes enumerated; best XI picked across all of them |
| Free transfers (1–5 banked) and −4 hits | Derived from transfer history, user-overridable, priced into every plan |
| Never buy an injured/suspended player | Availability gate on the candidate pool and again at verification |

Selling prices are an estimate — FPL only publishes exact figures to logged-in
managers. The estimate is anchored to the exact squad aggregate and errs on the
conservative side, so it can reject a legal transfer but never admit an illegal one.
Responses carry `budget.confidence` and a warning saying so.

### Expected points

`xpts.py` projects points per player per gameweek from minutes, availability,
underlying xG/xA/xGC, saves, bonus, defensive contribution and cards, adjusted for
fixture difficulty using team strength ratings, and blended with FPL's own `ep_next`.

Blanks and doubles are handled structurally — a gameweek holds a *list* of fixtures,
so a blank scores zero and a double scores both. Every projection returns its
component breakdown, and the total is asserted to equal the sum of its parts.

---

## Endpoints

| Route | Purpose |
|---|---|
| `GET /api/recommendations?user_id=` | Transfer advice. `horizon` (1–8), `max_transfers` (0–3), `free_transfers`, `bank`, `include_hits`, `engine` |
| `GET /api/fcps-recommendations?user_id=` | FCPS shortlist written up by a language model. Also accepts `POST`. `refresh` forces past the 15-min cache |
| `GET /api/engines` | What this deployment can do: which scoring engines exist, whether FCPS is configured, and why not |
| `GET /api/team?user_id=` | Squad with projections, bank and squad value |
| `GET /api/players` | All players with xPts **and FCPS**. `player_id`, `position`, `max_price`, `horizon`, `engine` |
| `GET /api/captain?user_id=` | Armband ranking + flagged-captain warnings |
| `GET /api/price-changes?user_id=` | Tonight's risers/fallers, framed against your squad |
| `GET /api/fixture-ticker?start=&count=` | Team × gameweek difficulty grid + named swings |
| `GET /api/fixtures` | Raw fixture list |
| `GET /api/entry/<id>` | Manager name, team, rank, bank |
| `GET /api/photo/<code>` | Player photo proxy (bypasses Safari ITP) |
| `GET /app/…` | Flutter web client |

Every API response carries a `meta` block with `fetched_at` and `stale`.

`engine` takes `xpts` (default), `ml` or `blend`. An unknown value, or `ml` on a
server with no trained artifact, silently falls back to `xpts` — and the response
says so in its `engine` field and in `warnings`, rather than labelling hand-built
numbers as model output. `GET /api/engines` is how the client knows which to
offer; it never guesses.

### Caching

One TTL cache owns all upstream I/O: bootstrap 10 min, fixtures 60 min, entry/picks
5 min, history 30 min. Every request has a hard timeout. If FPL is unreachable and we
hold stale data, it's served with `stale: true` rather than failing — an FPL outage
degrades the app instead of taking it down.

---

## Running it

```bash
pip install -r requirements.txt
python app.py                     # http://localhost:5001
pytest                            # the engine test suite — no network needed
```

Deploy: `gunicorn app:app` (see `Procfile`). Build and copy the Flutter client with
`bash scripts/deploy_flutter.sh`.

The web process needs none of the ML stack. With no trained artifact present it
starts, serves, and scores with `engine/xpts.py` exactly as before.

### Optional: the trained model

```bash
pip install -r requirements.txt -r requirements-ml.txt
python -m ml.sources     # download + cache 6 seasons of historical FPL data
python -m ml.train       # compare candidates, select on validation, score test once
python -m ml.backtest    # walk-forward evaluation vs every baseline, including FCPS
```

Writes `ml/artifacts/xpts_model.joblib`. Once it exists, `/api/engines` reports
`ml` and `blend` as available and the client's model picker appears — no client
redeploy needed, because the picker is driven by that endpoint.

Full methodology, splits, baselines and results:
`PRDs/ml-methodology.md` in the Flutter repo. Read §13 before reading §12.

### Optional: FCPS advice

FCPS scores are always computed and always returned. The *written column* needs a
model, reached through the **Claude CLI** rather than an HTTP API — the server
holds no API key and spends nothing per call, because the CLI authenticates with
the operator's own Claude subscription.

```bash
# The CLI must be installed and logged in as the user the web process runs as.
claude --version

export FCPS_CLAUDE_BIN=/home/you/.local/bin/claude   # optional, if not on PATH
export FCPS_MODEL=sonnet                             # optional
export FCPS_EFFORT=low                               # optional
```

Without it, `/api/fcps-recommendations` returns **503** with
`code: fcps_not_configured` and the client shows "FCPS advice is turned off"
rather than a retry button for a condition retrying can't fix.

**A subscription is free per call but finite per window**, and each invocation
carries roughly 23k tokens of CLI harness prompt whatever the payload. So the
route is rate-limited by a cache rather than by a counter: one call per
`(entry, gameweek, model)` per **24 hours**, and every later request that day is
served from it.

That gate is deliberately server-side. A browser cache cannot be one — the same
manager on a phone and a laptop is two devices, cleared storage is a third, and a
direct request to the endpoint is none of them. The cache is mirrored to disk
(`FCPS_CACHE_DIR`, default `~/.cache/fpl/fcps`) so a gunicorn restart doesn't
reopen the day's allowance. An unwritable cache directory degrades the gate to
per-process rather than failing the request.

`?refresh=1` bypasses the cache and spends a call.

### Optional: LLM narration

Off by default. When enabled, a language model writes two sentences *about* an
already-decided, already-verified plan — it is never shown alternatives and cannot
change a recommendation. Any failure drops the prose and leaves the response
otherwise identical.

```bash
export ENABLE_LLM_NARRATIVE=true
```

Uses the same Claude CLI as FCPS, so it needs no key either. **Only the first
plan is narrated.** It used to narrate all five the optimiser returns, on a route
with no LLM cache — one request was five calls, and the same request repeated was
five more.

### Optional: research digest

The projections know last season's numbers and the fixture list. They don't know
that a defender's 209-point season came largely from a scoring rule the ML model
was never trained on, that a £46m signing has no Premier League minutes because
he spent the year on loan, or that a club changed manager in February. All of
that is published every August and none of it reaches an FPL API endpoint.

A scheduled job fetches it, distils it once, and caches it; FCPS advice then
receives it as context.

```bash
export ENABLE_RESEARCH_DIGEST=true
python -m engine.research            # refresh now
scripts/refresh-research.sh          # the same thing, for cron
```

**Why a digest rather than search-per-request.** Giving the request path web
tools would undo two properties bought deliberately above. It would put
arbitrary web text into a prompt on an internet-facing endpoint once per
visitor, and it would turn one model call into a call plus an unbounded number
of searches against a rate limit shared with the operator's own work. The digest
runs **twice a day for the whole site** — 2 of the 250 daily calls — adds no
latency to a request, and confines web content to one artefact you can read
before it is used (`~/.cache/fpl/research/digest.json`).

Pages are fetched **in Python, against a hardcoded allowlist**. The model is
never given web tools; it only summarises text this codebase chose to hand it.
That is the difference between "the model researched something" and "the model
was handed a page we picked", and only the second is auditable. Override the
list with `RESEARCH_SOURCES` (JSON `[{name, url}]`) to follow a season's article
slugs without a redeploy.

The digest is the **only** content in an FCPS prompt not sourced from FPL's own
API, so it is fenced in `<reference_notes>` and labelled as quoted material, and
the system prompt states that the data tables win any disagreement about a
price, a points total or a status. A stale digest is dropped rather than served:
yesterday's "expected to start" is exactly the claim that becomes misinformation
once a team sheet lands.

---

## Protecting the model access

The model is reached through a CLI authenticated with the operator's **personal
Claude subscription**, on routes exposed to the open internet. That inverts the
usual threat model: there is no bill to cap, and the scarce resource — the
subscription's rate-limit window — is *shared with the operator's own Claude
Code sessions*. Someone abusing the endpoint doesn't run up a charge; they take
out the operator's own tooling as collateral.

Two properties of the naive design made that easy, and both are fixed:

**Caching is not rate limiting.** FCPS advice is cached per `(entry, gameweek,
model)` for a day, which stops one manager re-triggering a call and does nothing
about someone walking `user_id=1,2,3,…` through the several million entry IDs FPL
hands out — every one a fresh key and a fresh call. Query parameters
(`horizon`, `engine`, `max_transfers`) widen the same hole.

**Narration multiplied.** Five plans, five calls, no cache, per request.

### What enforces the limit

| Control | Default | What it stops |
|---|---|---|
| Global daily ceiling | 250 calls | Enumeration, in aggregate. No caller can raise it; it is the hard backstop. |
| Concurrency cap | 2 processes | Subprocess pile-up. Each call is a Node runtime at a couple of hundred MB; unbounded spawning exhausts memory and PIDs long before any quota. |
| Per-client hourly limit | 10 calls/hr | One abuser spending the whole day's ceiling and locking out real users. |
| Tool denial | all denied | The CLI is a coding agent by default. No shell, no filesystem, no network. |
| Scratch working directory | per call | The CLI picking up this repo's `CLAUDE.md` and settings as context. |

All three counters live **on disk under a lock**, not in memory. The app is
served by gunicorn with several workers, and a per-process counter would multiply
every limit by the worker count — a limit that reads as enforced while not being
one. `tests/test_llm_budget.py` asserts the ceiling holds across real processes.

Concurrency slots are `flock`s rather than a counter, because a counter has to be
decremented and a process killed mid-call never gets to decrement it. The kernel
releases a `flock` when the descriptor closes, however it closed.

The ceiling is charged **on entry, not on success**: a call that failed after the
model ran still spent the tokens, and refunding failures would make an error loop
an unmetered retry loop.

### Failure behaviour

Both routes fail closed, and they fail differently on purpose:

- FCPS returns **429** `fcps_budget_exhausted` or **503** `fcps_busy`. It is the
  whole feature, so the client is told why rather than shown a spinner.
- Narration **silently drops** the prose. It is additive garnish; a spent budget
  must not fail a request whose real content was computed without a model.

`/api/engines` reports `budget.remaining_today` so the client can explain a 429
instead of offering a retry that cannot work. It exposes counts only — nothing
identifies a caller.

### What is *not* claimed

- **`X-Forwarded-For` is ignored by default.** It is attacker-controlled unless a
  trusted proxy overwrites it; honouring it on a directly-exposed app would let
  anyone mint a fresh identity per request. Set `TRUST_PROXY_HEADER=true` only
  when a reverse proxy in front of this app rewrites it.
- **Per-client limiting is a speed bump, not an identity control.** IPs are
  spoofable, shared behind NAT, and cheap to rotate. The global ceiling is the
  guarantee; the per-client limit only makes it *fairly* distributed.
- **Prompt injection surface is one artefact, not every request.** Every other
  string in both prompts — player names, teams, prices, positions — comes from
  FPL's own bootstrap, and the only user-controlled input is an integer entry ID.
  The research digest is the sole exception, and it is deliberately shaped to be
  the *only* one: fetched from an allowlist rather than searched, distilled on a
  schedule rather than per request, fenced and labelled in the prompt, and
  overridden by the data tables on any factual disagreement. It is readable on
  disk before use. **Keep the rest that way**: adding the manager's own team name
  (which managers choose freely) to a prompt would open a second, per-request
  surface with none of those controls.
- Command injection is structurally absent rather than filtered: arguments are
  passed as a list with no shell, and the prompt goes in on stdin.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `5001` | Listen port |
| `ALLOWED_ORIGINS` | production + localhost | Comma-separated CORS origins for `/api/*` |
| `ENABLE_LLM_NARRATIVE` | `false` | Turn on narration |
| `FCPS_CLAUDE_BIN` | `claude` on `PATH` | Path to the Claude CLI |
| `FCPS_MODEL` | `sonnet` | Model used for the FCPS column |
| `NARRATIVE_MODEL` | `sonnet` | Model used for narration |
| `FCPS_EFFORT` | `low` | Effort level passed to the CLI |
| `FCPS_CACHE_DIR` | `~/.cache/fpl/fcps` | Where the 24h gate is persisted |
| `LLM_DAILY_CEILING` | `250` | Global model calls per day, all routes |
| `LLM_MAX_CONCURRENT` | `2` | Simultaneous CLI processes |
| `LLM_CLIENT_HOURLY` | `10` | Per-client model calls per hour |
| `LLM_BUDGET_DIR` | `~/.cache/fpl/llm-budget` | Where the counters live |
| `ENABLE_RESEARCH_DIGEST` | `false` | Inject the news/consensus digest into FCPS advice |
| `RESEARCH_SOURCES` | built-in allowlist | JSON `[{name, url}]` of pages to distil |
| `RESEARCH_TTL_SECONDS` | `43200` | How long a digest stays fresh (12h) |
| `RESEARCH_CACHE_DIR` | `~/.cache/fpl/research` | Where the digest is cached |
| `RESEARCH_EFFORT` | `medium` | Effort level for the distillation call |
| `TRUST_PROXY_HEADER` | `false` | Honour `X-Forwarded-For` (only behind a trusted proxy) |
| `FPL_ML_ARTIFACTS` | `ml/artifacts` | Where to look for the trained model |
| `FPL_ML_DATA` | `ml/_data` | Historical data cache (training only) |

---

## Design notes

Specifications live in `PRDs/` in the Flutter repo (`~/projects/fpl-old`), one per
feature, starting with `prd-transfer-engine.md`.

**Why the LLM doesn't decide anything.** The original version interpolated ~85
player records into a prompt and asked `gpt-4o-mini` to respect FPL's rules in
English. Nothing verified budget, quotas or the club limit; the model was never told
the manager's bank balance or free-transfer count; the output was prose no program
could parse; and it gave different answers on refresh. A recommendation you can't
legally execute is worse than none, because it costs you the time to find out. The
optimiser decides and proves it.

**Why FCPS is still here.** It was removed in an earlier pass, on the reasoning
that two competing scores is worse than either alone. That is right about the
*optimiser's objective* — which must be single-valued and in points — and wrong
about the *product*. FCPS is a number this app's users recognise. It is back, with
its weights bit-for-bit unchanged, and the written column it feeds is reachable
from the app for the first time: the old route rendered a template that was never
committed, so every GET returned 500, and the only working path was a `POST` the
client never called. Both the score and the column are labelled as opinion
throughout, next to the plans that were actually verified.

**Known limits.** Selling prices are estimated (see above). The free-transfer count is
derived, not published, and is user-overridable. Search is exhaustive at one transfer
and near-exhaustive at two and three. Price-change prediction approximates an
undocumented algorithm and says so in every response. The trained model is limited
to features computable from a live `bootstrap-static` payload, which costs it most
of its recency signal — `PRDs/ml-methodology.md` §5 and §14 explain why and what
the fix is.

## Licence

MIT — see `LICENSE`.
