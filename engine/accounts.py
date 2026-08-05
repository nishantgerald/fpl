"""User accounts: registration, sessions, FPL links, and the cookie vault.

The design decision that shapes everything here: **we never see a Premier
League password.** Users register with an email and a password *for this app*,
then link their FPL entry by Team ID — which is how every large FPL tool works,
because the public API serves squads, transfers and leagues from the ID alone
once the season starts. The optional extra is a per-user FPL session cookie
(for pre-deadline squad reads), stored encrypted and never returned once
stored.

SQLite on purpose. One process, thousands of users, read-heavy: well within
SQLite's envelope, and moving to Postgres later is a change to this module
alone. Passwords are scrypt-hashed via Werkzeug; session tokens are opaque
random values stored server-side so logout actually revokes.
"""

from __future__ import annotations

import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from werkzeug.security import check_password_hash, generate_password_hash

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

_lock = threading.Lock()


class AccountError(Exception):
    """A user-correctable problem; `code` keys the client's message."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


# ---------------------------------------------------------------- storage


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                -- Empty string means "no password set" (a Google-only account).
                -- NULL would say the same thing more idiomatically, but the
                -- column predates Google sign-in and SQLite cannot drop a NOT
                -- NULL without rebuilding the table.
                password_hash TEXT NOT NULL,
                google_sub TEXT,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fpl_links (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                entry_id INTEGER NOT NULL,
                team_name TEXT,
                manager_name TEXT,
                linked_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fpl_cookies (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                ciphertext BLOB NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        # Databases created before Google sign-in lack the column; CREATE IF
        # NOT EXISTS won't add it to an existing table.
        columns = [row[1] for row in conn.execute("PRAGMA table_info(users)")]
        if "google_sub" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN google_sub TEXT")
        # Opt-in, not opt-out. Nobody signed up to be emailed, and a deadline
        # reminder nobody asked for is spam however useful it is.
        if "deadline_email" not in columns:
            conn.execute(
                "ALTER TABLE users ADD COLUMN deadline_email INTEGER NOT NULL DEFAULT 0"
            )
        # SQLite unique indexes permit any number of NULLs, so password-only
        # accounts coexist fine.
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub"
            " ON users (google_sub)"
        )


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

    with _lock, _connect() as conn:
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
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, generate_password_hash(password), time.time()),
        )
        return {"id": cursor.lastrowid, "email": email}


def _mint_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _lock, _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at)"
            " VALUES (?, ?, ?, ?)",
            (token, user_id, now, now + SESSION_TTL_SECONDS),
        )
    return token


def authenticate(email: str, password: str) -> str:
    """Verify credentials and mint a session token."""
    email = (email or "").strip().lower()
    with _lock, _connect() as conn:
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

    with _lock, _connect() as conn:
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
                    "INSERT INTO users (email, password_hash, google_sub, created_at)"
                    " VALUES (?, '', ?, ?)",
                    (email, sub, time.time()),
                )
                user_id = cursor.lastrowid

    return _mint_session(user_id)


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
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (generate_password_hash(password), user_id),
        )


def login_methods(user_id: int) -> dict:
    """Which ways this account can sign in. Booleans only — no secrets."""
    with _lock, _connect() as conn:
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
    with _lock, _connect() as conn:
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
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ---------------------------------------------------------------- FPL link


def link_fpl(user_id: int, entry_id: int, team_name: str, manager_name: str) -> dict:
    with _lock, _connect() as conn:
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
    with _lock, _connect() as conn:
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
    with _lock, _connect() as conn:
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
    with _lock, _connect() as conn:
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
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM fpl_cookies WHERE user_id = ?", (user_id,))


def has_cookie(user_id: int) -> bool:
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM fpl_cookies WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row is not None


# ---------------------------------------------------------------- digests


def set_deadline_email(user_id: int, enabled: bool) -> None:
    """Opt in or out of the pre-deadline briefing."""
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE users SET deadline_email = ? WHERE id = ?",
            (1 if enabled else 0, user_id),
        )


def wants_deadline_email(user_id: int) -> bool:
    with _lock, _connect() as conn:
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
    with _lock, _connect() as conn:
        rows = conn.execute(
            """
            SELECT u.id, u.email, l.entry_id, l.team_name, l.manager_name
            FROM users u JOIN fpl_links l ON l.user_id = u.id
            WHERE u.deadline_email = 1
            ORDER BY u.id
            """
        ).fetchall()
    return [dict(row) for row in rows]
