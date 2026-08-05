#!/usr/bin/env python3
"""Send the pre-deadline briefing to everyone who opted in.

Run from cron a few hours before each deadline::

    python -m scripts.send_digests            # send
    python -m scripts.send_digests --dry-run  # render, send nothing

Deliberately a script rather than a route. A mailing run is not a request: it
takes as long as it takes, must not be triggered by a stranger hitting a URL,
and needs to survive one user's data being broken without abandoning everyone
after them in the list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import accounts, digest, mailer, service  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render every digest and print it, sending nothing",
    )
    parser.add_argument(
        "--only",
        type=int,
        help="restrict to one FPL entry id, for checking a single account",
    )
    args = parser.parse_args()

    accounts.init_db()
    subscribers = accounts.digest_subscribers()
    if args.only:
        subscribers = [s for s in subscribers if s["entry_id"] == args.only]

    if not subscribers:
        print("[digest] nobody subscribed; nothing to do")
        return 0
    if not args.dry_run and not mailer.is_configured():
        print("[digest] SMTP not configured — re-run with --dry-run to preview")
        return 1

    sent = failed = skipped = 0
    for subscriber in subscribers:
        label = f"{subscriber['email']} (entry {subscriber['entry_id']})"
        try:
            briefing = service.deadline_digest(
                subscriber["entry_id"],
                manager_name=subscriber.get("manager_name") or "",
            )
        except service.ServiceError as error:
            # One broken account must not abandon everyone after it in the list.
            print(f"[digest] skip {label}: {error.code}")
            skipped += 1
            continue
        except Exception as error:  # noqa: BLE001
            print(f"[digest] skip {label}: {type(error).__name__}: {error}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"\n=== {label} ===")
            print(f"Subject: {briefing['subject']}")
            print(digest.render_text(briefing))
            continue

        delivery = mailer.send(
            to=subscriber["email"],
            subject=briefing["subject"],
            text=digest.render_text(briefing),
            html=digest.render_html(briefing),
        )
        if delivery.sent:
            sent += 1
        else:
            failed += 1
            print(f"[digest] failed {label}: {delivery.reason}")

    if not args.dry_run:
        print(f"[digest] sent={sent} failed={failed} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
