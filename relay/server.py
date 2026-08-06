"""The home end of the LLM relay: lends the local Claude CLI to the dyno.

Runs on the machine that already has a Claude subscription. Production signs
each request with a private key it never transmits; this process holds only the
public half, so an attacker who reads this machine's disk still cannot call it.

Run it:

    LLM_RELAY_PUBLIC_KEY=... python relay/server.py

and expose it with a Cloudflare Tunnel (no inbound ports, no router config):

    cloudflared tunnel --url http://127.0.0.1:8765

Everything it will run is bounded by _ALLOWED_FLAGS below. The dyno cannot ask
for a tool that is not on the list, cannot name a binary, and cannot reach a
path outside the per-call scratch directory — because a signed request from a
compromised app should still not be able to run shell commands on a home
machine.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import claude_cli  # noqa: E402

# Flags the CLI is allowed to be invoked with. An allowlist rather than a
# denylist: a new upstream flag should arrive switched off, not switched on.
_ALLOWED_FLAGS = {
    "-p",
    "--model",
    "--effort",
    "--output-format",
    "--system-prompt",
    "--allowedTools",
    "--disallowedTools",
    "--add-dir",
}

# Tools that must never be reachable through the relay, whatever the caller
# asks for. Reading one uploaded image is a feature; running shell commands on
# a home machine on behalf of an anonymous web upload is not.
_FORBIDDEN_TOOLS = {"bash", "write", "edit", "webfetch", "websearch", "task"}

MAX_BODY_BYTES = 12 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 300

# Nonces already spent, with the time they were seen. Bounded by the skew
# window: anything older cannot be replayed anyway, because the timestamp check
# would reject it first.
_seen_nonces: dict[str, float] = {}


def _public_key():
    raw = os.getenv("LLM_RELAY_PUBLIC_KEY", "").strip()
    if not raw:
        return None
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        return Ed25519PublicKey.from_public_bytes(base64.b64decode(raw))
    except Exception:
        return None


def _check_nonce(nonce: str) -> bool:
    now = time.time()
    horizon = claude_cli.MAX_SKEW_SECONDS * 2
    for stale in [n for n, seen in _seen_nonces.items() if now - seen > horizon]:
        _seen_nonces.pop(stale, None)
    if nonce in _seen_nonces:
        return False
    _seen_nonces[nonce] = now
    return True


def validate_args(args: list[str]) -> str | None:
    """None if this argv is safe to run, otherwise why it isn't."""
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("-"):
            # A bare positional would be a prompt or, worse, a path. Prompts
            # arrive on stdin; nothing else is expected here.
            return f"unexpected positional argument: {token[:40]}"
        if token not in _ALLOWED_FLAGS:
            return f"flag not allowed: {token[:40]}"
        if token == "-p":
            index += 1
            continue
        if index + 1 >= len(args):
            return f"{token} is missing its value"
        value = args[index + 1]
        if token == "--allowedTools":
            for tool in value.replace(",", " ").split():
                if tool.strip().lower() in _FORBIDDEN_TOOLS:
                    return f"tool not allowed through the relay: {tool}"
        index += 2
    return None


def invoke(payload: dict) -> dict:
    args = [str(a) for a in payload.get("args", [])]
    problem = validate_args(args)
    if problem:
        raise ValueError(problem)

    binary = claude_cli.local_binary()
    if binary is None:
        raise RuntimeError("This machine has no Claude CLI on PATH.")

    timeout = min(int(payload.get("timeout", 120) or 120), MAX_TIMEOUT_SECONDS)

    with tempfile.TemporaryDirectory(prefix="fpl-relay-") as scratch:
        rewritten = list(args)
        for name, encoded in (payload.get("attachments") or {}).items():
            # Basename only: an attachment called "../../.ssh/id_rsa" must land
            # in the scratch directory like everything else.
            target = Path(scratch) / Path(str(name)).name
            target.write_bytes(base64.b64decode(encoded))
        # Any --add-dir the caller asked for is replaced with the scratch
        # directory. The dyno's idea of a path is meaningless here, and honouring
        # it would be the whole vulnerability.
        for index, token in enumerate(rewritten):
            if token == "--add-dir" and index + 1 < len(rewritten):
                rewritten[index + 1] = scratch

        completed = subprocess.run(
            [binary, *rewritten],
            input=str(payload.get("input", "")),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=scratch,
        )

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr[-4000:],
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "fpl-relay/1.0"

    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path != "/health":
            return self._reply(404, {"error": "not found"})
        # Unauthenticated on purpose, and it reveals nothing but liveness: the
        # tunnel needs something to probe, and a signed health check would make
        # "is it up?" require the very key we're protecting.
        self._reply(
            200,
            {
                "ok": True,
                "cli": claude_cli.local_binary() is not None,
                "key": _public_key() is not None,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/invoke":
            return self._reply(404, {"error": "not found"})

        public_key = _public_key()
        if public_key is None:
            return self._reply(500, {"error": "relay has no public key configured"})

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            return self._reply(413, {"error": "body missing or too large"})
        body = self.rfile.read(length)

        timestamp = self.headers.get("X-Relay-Timestamp", "")
        nonce = self.headers.get("X-Relay-Nonce", "")
        signature = self.headers.get("X-Relay-Signature", "")
        try:
            skew = abs(time.time() - int(timestamp))
        except ValueError:
            return self._reply(401, {"error": "bad timestamp"})
        if skew > claude_cli.MAX_SKEW_SECONDS:
            return self._reply(401, {"error": "stale request"})
        if not claude_cli.verify(public_key, timestamp, nonce, body, signature):
            return self._reply(401, {"error": "bad signature"})
        if not _check_nonce(nonce):
            return self._reply(401, {"error": "replayed nonce"})

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return self._reply(400, {"error": "bad json"})

        try:
            self._reply(200, invoke(payload))
        except ValueError as error:
            self._reply(400, {"error": str(error)})
        except subprocess.TimeoutExpired:
            self._reply(504, {"error": "the model timed out"})
        except Exception as error:  # noqa: BLE001 - the relay must not die
            self._reply(500, {"error": f"{type(error).__name__}: {error}"})

    def log_message(self, fmt: str, *args) -> None:
        # One line per call, on stderr, without the client address: this is a
        # home machine and the log is likely to outlive anyone's interest in it.
        sys.stderr.write(f"[relay] {fmt % args}\n")


def main() -> int:
    if _public_key() is None:
        print(
            "LLM_RELAY_PUBLIC_KEY is not set or is not a valid Ed25519 key.\n"
            "Generate a pair with:  python relay/keygen.py",
            file=sys.stderr,
        )
        return 2
    if shutil.which("claude") is None and not os.getenv("FCPS_CLAUDE_BIN"):
        print("No `claude` on PATH — the relay would refuse every call.", file=sys.stderr)
        return 2

    host = os.getenv("LLM_RELAY_HOST", "127.0.0.1")
    port = int(os.getenv("LLM_RELAY_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[relay] listening on http://{host}:{port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
