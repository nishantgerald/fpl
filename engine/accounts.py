"""User accounts: registration, sessions, FPL links, and the cookie vault.

The design decision that shapes everything here: **we never see a Premier
League password.** Users register with an email and a password *for this app*,
then link their FPL entry by Team ID — which is how every large FPL tool works,
because the public API serves squads, transfers and leagues from the ID alone
once the season starts. The optional extra is a per-user FPL session cookie
(for pre-deadline squad reads), stored encrypted and never returned once
stored.

Storage is chosen by `DATABASE_URL`: Postgres when one is set, otherwise a
SQLite file. That is not a preference but a correctness requirement — Heroku's
filesystem is ephemeral, so a SQLite file in production is destroyed on every
restart, taking every account with it. `engine/db.py` hides the difference, and
the queries below are the same either way. Passwords are scrypt-hashed via
Werkzeug; session tokens are opaque random values stored server-side so logout
actually revokes.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path
from typing import Sequence

from cryptography.fernet import Fernet, InvalidToken
from werkzeug.security import check_password_hash, generate_password_hash

from . import db

DB_PATH = Path(
    os.getenv("FPL_APP_DB", Path.home() / ".local" / "share" / "fpl" / "app.db")
)

# Encrypts stored FPL cookies at rest. Generated once and chmod 600; an env
# override exists so a hosted deployment can supply it as a config var instead
# of a disk file.
KEY_PATH = Path(
    os.getenv("FPL_APP_KEY_FILE", Path.home() / ".local" / "share" / "fpl" / "vault.key")
)

SESSION_TTL_SECONDS = 30 * 24 * 3600
MIN_PASSWORD_LENGTH = 8

# Deliberately loose: its job is catching typos, not adjudicating RFC 5322.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Writer serialisation lives in `db.lock()`, which applies it on SQLite (one
# file, one writer) and not on Postgres. Holding a process-wide mutex across a
# Postgres query would serialise every request in a worker behind a network
# round-trip to the database — the opposite of what a server built for
# concurrency needs.


class AccountError(Exception):
    """A user-correctable problem; `code` keys the client's message."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


# ---------------------------------------------------------------- storage


def _connect() -> db.Connection:
    """Postgres when DATABASE_URL is set, else the SQLite file at DB_PATH.

    DB_PATH is still consulted so the tests, which point it at a tmp_path, keep
    working unchanged.
    """
    return db.connect(sqlite_path=DB_PATH)


def init_db() -> None:
    """Create the schema, and upgrade a database that predates a column.

    The migration is expressed against `information_schema` rather than
    SQLite's `PRAGMA table_info`, because it has to run on both backends: a
    long-lived Postgres database is exactly the thing that will still be here
    when the next column is added.
    """
    with db.lock(), _connect() as conn:
        # Before any DDL. Every gunicorn worker runs this on import, and
        # CREATE TABLE IF NOT EXISTS is not atomic on Postgres — against an
        # empty database the workers race, one dies, and gunicorn shuts the
        # master down because a worker failed to boot.
        db.schema_lock(conn)
        db.init_schema(conn)

        columns = _existing_columns(conn, "users")
        # Databases created before Google sign-in lack the column; CREATE TABLE
        # IF NOT EXISTS won't add it to an existing table.
        if "google_sub" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN google_sub TEXT")
        # On by default for new accounts: the briefing is the reason to have an
        # account at all, and burying it behind a toggle nobody finds means the
        # feature may as well not exist. Every message carries the way out, and
        # one switch on the account screen turns it off for good.
        if "deadline_email" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN deadline_email INTEGER NOT NULL DEFAULT 1"
            )

        # Last: an index names a column, so it cannot be created before the
        # migration that adds one.
        db.init_indexes(conn)


def _existing_columns(conn: db.Connection, table: str) -> set[str]:
    if conn.postgres:
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_name = ?",
            (table,),
        ).fetchall()
        return {row["column_name"] for row in rows}
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _fernet() -> Fernet:
    from_env = os.getenv("FPL_APP_VAULT_KEY", "").strip()
    if from_env:
        return Fernet(from_env.encode())
    if not KEY_PATH.exists():
        KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        KEY_PATH.write_bytes(Fernet.generate_key())
        KEY_PATH.chmod(0o600)
    return Fernet(KEY_PATH.read_bytes().strip())


# ---------------------------------------------------------------- users


def register(email: str, password: str) -> dict:
    email = (email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        raise AccountError("invalid_email", "That doesn't look like an email address.")
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AccountError(
            "weak_password",
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
        )

    with db.lock(), _connect() as conn:
        existing = conn.execute(
            "SELECT password_hash, google_sub FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing is not None:
            if not existing["password_hash"] and existing["google_sub"]:
                # The account exists but only Google can prove it's theirs.
                # Attaching this password blind would let anyone who knows the
                # email capture a Google user's account — we have no email
                # verification to say otherwise. The owner adds a password
                # from inside a signed-in session instead (set_password).
                raise AccountError(
                    "email_taken",
                    "That email already signed in with Google. Continue with "
                    "Google, then add a password under Account.",
                )
            raise AccountError(
                "email_taken", "An account with that email already exists."
            )
        cursor = conn.execute(
            "INSERT INTO users (email, password_hash, created_at, deadline_email)"
            " VALUES (?, ?, ?, 1)",
            (email, generate_password_hash(password), time.time()),
        )
        return {"id": cursor.lastrowid, "email": email, "is_new": True}


def _mint_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with db.lock(), _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at)"
            " VALUES (?, ?, ?, ?)",
            (token, user_id, now, now + SESSION_TTL_SECONDS),
        )
    return token


def authenticate(email: str, password: str) -> str:
    """Verify credentials and mint a session token."""
    email = (email or "").strip().lower()
    with db.lock(), _connect() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()

    # One error for wrong-email, wrong-password and google-only accounts
    # alike: distinguishing them hands an enumeration oracle to anyone probing
    # for registered emails.
    if (
        row is None
        or not row["password_hash"]
        or not check_password_hash(row["password_hash"], password or "")
    ):
        raise AccountError("bad_credentials", "Email or password is incorrect.")

    return _mint_session(row["id"])


def login_google(sub: str, email: str) -> str:
    """Sign in (or up) with a Google-verified identity; mint a session token.

    Consolidation happens here, in the safe direction: Google has verified
    that this person controls ``email``, so an existing password account with
    the same address is *their* account, and the Google identity is attached
    to it — one record, both login methods. The unsafe direction (a password
    registration claiming a Google account's email) is refused in
    :func:`register` and satisfied by :func:`set_password` instead.
    """
    email = (email or "").strip().lower()
    if not sub or not _EMAIL_RE.match(email):
        raise AccountError("bad_google_identity", "Google returned an unusable identity.")

    created = False
    with db.lock(), _connect() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE google_sub = ?", (sub,)
        ).fetchone()
        if row is not None:
            user_id = row["id"]
        else:
            by_email = conn.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()
            if by_email is not None:
                conn.execute(
                    "UPDATE users SET google_sub = ? WHERE id = ?",
                    (sub, by_email["id"]),
                )
                user_id = by_email["id"]
            else:
                cursor = conn.execute(
                    "INSERT INTO users (email, password_hash, google_sub,"
                    " created_at, deadline_email) VALUES (?, '', ?, ?, 1)",
                    (email, sub, time.time()),
                )
                user_id = cursor.lastrowid
                created = True

    return _mint_session(user_id), created


def set_password(user_id: int, password: str) -> None:
    """Add or change the password on an existing account.

    This is the legitimate route to "Google account that also takes a
    password": the caller is inside an authenticated session, so ownership is
    already proven — unlike a cold registration with the same email.
    """
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AccountError(
            "weak_password",
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
        )
    with db.lock(), _connect() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (generate_password_hash(password), user_id),
        )


def login_methods(user_id: int) -> dict:
    """Which ways this account can sign in. Booleans only — no secrets."""
    with db.lock(), _connect() as conn:
        row = conn.execute(
            "SELECT password_hash, google_sub FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if row is None:
        return {"password": False, "google": False}
    return {"password": bool(row["password_hash"]), "google": bool(row["google_sub"])}


def user_for_token(token: str | None) -> dict | None:
    """The user a bearer token belongs to, or None. Expired tokens are reaped."""
    if not token:
        return None
    with db.lock(), _connect() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.email, s.expires_at
            FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] < time.time():
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None
    return {"id": row["id"], "email": row["email"]}


def logout(token: str | None) -> None:
    if not token:
        return
    with db.lock(), _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ---------------------------------------------------------------- FPL link


def link_fpl(user_id: int, entry_id: int, team_name: str, manager_name: str) -> dict:
    with db.lock(), _connect() as conn:
        conn.execute(
            """
            INSERT INTO fpl_links (user_id, entry_id, team_name, manager_name, linked_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (user_id) DO UPDATE SET
                entry_id = excluded.entry_id,
                team_name = excluded.team_name,
                manager_name = excluded.manager_name,
                linked_at = excluded.linked_at
            """,
            (user_id, entry_id, team_name, manager_name, time.time()),
        )
    return {"entry_id": entry_id, "team_name": team_name, "manager_name": manager_name}


def fpl_link(user_id: int) -> dict | None:
    with db.lock(), _connect() as conn:
        row = conn.execute(
            "SELECT entry_id, team_name, manager_name FROM fpl_links WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------- cookie vault


def store_cookie(user_id: int, cookie: str) -> None:
    """Encrypt and store a user's FPL session cookie. Write-only from outside:
    nothing in the API surface ever returns it, only whether one exists."""
    ciphertext = _fernet().encrypt(cookie.encode("utf-8"))
    with db.lock(), _connect() as conn:
        conn.execute(
            """
            INSERT INTO fpl_cookies (user_id, ciphertext, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT (user_id) DO UPDATE SET
                ciphertext = excluded.ciphertext,
                updated_at = excluded.updated_at
            """,
            (user_id, ciphertext, time.time()),
        )


def cookie_for(user_id: int) -> str | None:
    with db.lock(), _connect() as conn:
        row = conn.execute(
            "SELECT ciphertext FROM fpl_cookies WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row is None:
        return None
    try:
        return _fernet().decrypt(row["ciphertext"]).decode("utf-8")
    except InvalidToken:
        # A rotated key makes old ciphertexts unreadable. Treat as absent so the
        # user is prompted to reconnect rather than shown a 500.
        return None


def delete_cookie(user_id: int) -> None:
    with db.lock(), _connect() as conn:
        conn.execute("DELETE FROM fpl_cookies WHERE user_id = ?", (user_id,))


# ---------------------------------------------------------------- saved squads


def store_squad(
    user_id: int,
    element_ids: Sequence[int],
    bench_ids: Sequence[int] = (),
    source: str = "screenshot",
) -> None:
    """Remember the fifteen a user told us they own.

    Before the first deadline a squad is private to FPL, so without a session
    cookie there is nothing to give advice about — which is exactly when advice
    is most useful, because everything is still changeable. A squad read off a
    screenshot is a perfectly good answer to that, but only if it outlives the
    request that parsed it.
    """
    with db.lock(), _connect() as conn:
        conn.execute(
            """
            INSERT INTO saved_squads (user_id, element_ids, bench_ids, source, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (user_id) DO UPDATE SET
                element_ids = excluded.element_ids,
                bench_ids = excluded.bench_ids,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                json.dumps([int(i) for i in element_ids]),
                json.dumps([int(i) for i in bench_ids]),
                source,
                time.time(),
            ),
        )


def saved_squad(user_id: int) -> dict | None:
    with db.lock(), _connect() as conn:
        row = conn.execute(
            "SELECT element_ids, bench_ids, source, updated_at FROM saved_squads"
            " WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "element_ids": json.loads(row["element_ids"]),
        "bench_ids": json.loads(row["bench_ids"]),
        "source": row["source"],
        "updated_at": row["updated_at"],
    }


def has_cookie(user_id: int) -> bool:
    with db.lock(), _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM fpl_cookies WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row is not None


# ---------------------------------------------------------------- digests


def set_deadline_email(user_id: int, enabled: bool) -> None:
    """Opt in or out of the pre-deadline briefing."""
    with db.lock(), _connect() as conn:
        conn.execute(
            "UPDATE users SET deadline_email = ? WHERE id = ?",
            (1 if enabled else 0, user_id),
        )


def wants_deadline_email(user_id: int) -> bool:
    with db.lock(), _connect() as conn:
        row = conn.execute(
            "SELECT deadline_email FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return bool(row and row["deadline_email"])


def digest_subscribers() -> list[dict]:
    """Everyone who opted in *and* linked a team.

    Both conditions matter: a subscriber with no linked entry has nothing to be
    briefed about, and sending them an empty digest teaches them to ignore the
    next one.
    """
    with db.lock(), _connect() as conn:
        rows = conn.execute(
            """
            SELECT u.id, u.email, l.entry_id, l.team_name, l.manager_name
            FROM users u JOIN fpl_links l ON l.user_id = u.id
            WHERE u.deadline_email = 1
            ORDER BY u.id
            """
        ).fetchall()
    return [dict(row) for row in rows]


# ---------------------------------------------------------- password resets

# Long enough to survive a slow inbox, short enough that a link found later in
# a forwarded email or a shared screen is already dead.
RESET_TTL_SECONDS = 3600

# One live link per account. Requesting a second invalidates the first, so a
# user who clicks "forgot password" three times cannot leave three working keys
# to their account lying in an inbox.
def create_reset_token(email: str) -> tuple[str, dict] | None:
    """Mint a reset token for ``email``, or ``None`` if nobody has that address.

    The caller must **not** vary its response on the return value. Telling an
    anonymous caller whether an address is registered is an enumeration oracle,
    and the whole point of a reset flow is that it is reachable by someone who
    is not signed in.

    Only a hash of the token is stored. A leaked database then yields no usable
    reset links, the same reasoning that applies to passwords.
    """
    email = (email or "").strip().lower()
    with db.lock(), _connect() as conn:
        row = conn.execute(
            "SELECT id, email FROM users WHERE email = ?", (email,)
        ).fetchone()
        if row is None:
            return None

        token = secrets.token_urlsafe(32)
        now = time.time()
        conn.execute("DELETE FROM password_resets WHERE user_id = ?", (row["id"],))
        conn.execute(
            "INSERT INTO password_resets (token_hash, user_id, created_at, expires_at)"
            " VALUES (?, ?, ?, ?)",
            (_hash_token(token), row["id"], now, now + RESET_TTL_SECONDS),
        )
    return token, {"id": row["id"], "email": row["email"]}


def _hash_token(token: str) -> str:
    import hashlib

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def reset_password(token: str, password: str) -> dict:
    """Consume a reset token and set a new password.

    Every existing session for that user is revoked. Someone resetting a
    password may be doing it because they think somebody else has it, and
    leaving the intruder's session alive would defeat the exercise.
    """
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AccountError(
            "weak_password",
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
        )

    token_hash = _hash_token(token or "")
    now = time.time()
    with db.lock(), _connect() as conn:
        row = conn.execute(
            "SELECT user_id, expires_at, used_at FROM password_resets"
            " WHERE token_hash = ?",
            (token_hash,),
        ).fetchone()
        # One message for missing, expired and already-used. Distinguishing
        # them tells someone holding a stale link which kind of stale it is,
        # which helps nobody but them.
        if row is None or row["used_at"] is not None or row["expires_at"] < now:
            raise AccountError(
                "invalid_reset_token",
                "That reset link is invalid or has expired. Request a new one.",
            )

        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (generate_password_hash(password), row["user_id"]),
        )
        conn.execute(
            "UPDATE password_resets SET used_at = ? WHERE token_hash = ?",
            (now, token_hash),
        )
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (row["user_id"],))
        user = conn.execute(
            "SELECT id, email FROM users WHERE id = ?", (row["user_id"],)
        ).fetchone()

    return {"id": user["id"], "email": user["email"]}


def purge_expired_resets() -> int:
    """Drop spent and expired tokens. Cheap hygiene, safe to call any time."""
    with db.lock(), _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM password_resets WHERE expires_at < ? OR used_at IS NOT NULL",
            (time.time(),),
        )
        return cursor.rowcount
