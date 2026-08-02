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
key:

```bash
export OPENAI_API_KEY=sk-...
export FCPS_MODEL=gpt-4o-mini     # optional
```

Without one, `/api/fcps-recommendations` returns **503** with
`code: fcps_not_configured` and the client shows "FCPS advice is turned off"
rather than a retry button for a condition retrying can't fix.

### Optional: LLM narration

Off by default. When enabled, a language model writes two sentences *about* an
already-decided, already-verified plan — it is never shown alternatives and cannot
change a recommendation. Any failure drops the prose and leaves the response
otherwise identical.

```bash
export ENABLE_LLM_NARRATIVE=true
export OPENAI_API_KEY=sk-...
```

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `5001` | Listen port |
| `ALLOWED_ORIGINS` | production + localhost | Comma-separated CORS origins for `/api/*` |
| `ENABLE_LLM_NARRATIVE` | `false` | Turn on narration |
| `OPENAI_API_KEY` | — | Required for FCPS advice and for narration |
| `FCPS_MODEL` | `gpt-4o-mini` | Model used for the FCPS column |
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
