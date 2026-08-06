"""One seam for every Claude CLI invocation: run it here, or relay it home.

The five model-backed features all shell out to the same binary in the same
shape. On a Heroku dyno that binary does not exist, so all five are simply off
in production — `/api/import/config` reports `{"screenshot": false}` and the
upload button hides itself.

Rather than pay a second API bill, the dyno can borrow the CLI on the machine
that already has a subscription. `relay/server.py` exposes exactly this call
over HTTP; this module decides which side of the wire runs it, and every caller
keeps the `subprocess.CompletedProcess` interface it already had.

The relay is authenticated with Ed25519 over the request, not with a shared
bearer token: the dyno holds a private key that never leaves it, and the relay
holds only the public half, so reading the relay's disk gives an attacker
nothing that lets them call it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

# Signature freshness. Long enough to survive ordinary clock drift between a
# dyno and a home machine, short enough that a captured request is useless by
# the time anyone finds it.
MAX_SKEW_SECONDS = 60


class RelayError(RuntimeError):
    """The relay could not be reached, or refused the call."""


def local_binary() -> str | None:
    """The Claude CLI on *this* machine, if there is one."""
    override = os.getenv("FCPS_CLAUDE_BIN", "").strip()
    if override:
        return override if Path(override).exists() else None
    return shutil.which("claude")


def relay_url() -> str:
    return os.getenv("LLM_RELAY_URL", "").strip().rstrip("/")


def _private_key():
    raw = os.getenv("LLM_RELAY_PRIVATE_KEY", "").strip()
    if not raw:
        return None
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    try:
        return Ed25519PrivateKey.from_private_bytes(base64.b64decode(raw))
    except Exception:
        # A malformed key is a misconfiguration, not a runtime condition: treat
        # it as "no relay" so the app degrades instead of 500ing every call.
        return None


def relay_configured() -> bool:
    return bool(relay_url()) and _private_key() is not None


def is_available() -> bool:
    """Whether a model can be reached at all, by either route."""
    return local_binary() is not None or relay_configured()


def sign(private_key, timestamp: str, nonce: str, body: bytes) -> str:
    """The signature both sides compute.

    Covers the body digest as well as the timestamp and nonce — signing only
    the envelope would let anyone who caught one request swap the prompt.
    """
    digest = hashlib.sha256(body).hexdigest()
    message = f"{timestamp}.{nonce}.{digest}".encode()
    return base64.b64encode(private_key.sign(message)).decode()


def verify(public_key, timestamp: str, nonce: str, body: bytes, signature: str) -> bool:
    from cryptography.exceptions import InvalidSignature

    digest = hashlib.sha256(body).hexdigest()
    message = f"{timestamp}.{nonce}.{digest}".encode()
    try:
        public_key.verify(base64.b64decode(signature), message)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def run(
    command: list[str],
    *,
    input: str,
    timeout: int,
    cwd: str | None = None,
    attachments: dict[str, bytes] | None = None,
) -> subprocess.CompletedProcess:
    """Run a Claude CLI invocation locally, or via the relay.

    [command] is the full argv including the binary at index 0; when relaying,
    the binary is replaced by whatever the relay has. [attachments] are files
    the prompt refers to by name — the screenshot importer's image is the only
    current case, and it is why this cannot be a plain "send a prompt" call.

    The rule is simply "if I can run this here, I do". That is decided from
    `command[0]` as handed in rather than re-resolved from the environment:
    the caller already chose a binary, and quietly substituting a different one
    would make the choice untestable and the behaviour surprising.
    """
    if _runnable(command[0]):
        return subprocess.run(
            command,
            input=input,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    if not relay_configured():
        raise OSError("No Claude CLI available and no relay configured.")
    return _run_via_relay(command, input, timeout, attachments or {})


def _runnable(binary: str) -> bool:
    if not binary:
        return False
    if "/" in binary:
        return os.path.isfile(binary) and os.access(binary, os.X_OK)
    return shutil.which(binary) is not None


def _run_via_relay(
    command: list[str],
    prompt: str,
    timeout: int,
    attachments: dict[str, bytes],
) -> subprocess.CompletedProcess:
    payload = {
        # The binary is deliberately dropped: the relay uses its own, and
        # accepting a path from the caller would be a remote-execution hole
        # dressed up as a config option.
        "args": list(command[1:]),
        "input": prompt,
        "timeout": timeout,
        "attachments": {
            name: base64.b64encode(data).decode() for name, data in attachments.items()
        },
    }
    body = json.dumps(payload).encode()
    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(16)

    request = urlrequest.Request(
        f"{relay_url()}/invoke",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Relay-Timestamp": timestamp,
            "X-Relay-Nonce": nonce,
            "X-Relay-Signature": sign(_private_key(), timestamp, nonce, body),
        },
    )
    try:
        # A little beyond the CLI's own timeout: the relay should be the one to
        # give up first, so a timeout is reported as a model timeout rather than
        # as an unreachable relay.
        with urlrequest.urlopen(request, timeout=timeout + 15) as response:
            answer = json.loads(response.read())
    except urlerror.HTTPError as error:
        detail = error.read().decode(errors="replace")[:200]
        raise RelayError(f"Relay refused the call ({error.code}): {detail}") from error
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
        # The home machine being asleep is the expected failure, not an
        # exceptional one. Callers already handle OSError as "no model".
        raise OSError(f"Relay unreachable: {type(error).__name__}") from error

    return subprocess.CompletedProcess(
        args=command,
        returncode=int(answer.get("returncode", 1)),
        stdout=str(answer.get("stdout", "")),
        stderr=str(answer.get("stderr", "")),
    )
