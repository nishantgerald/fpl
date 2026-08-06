"""The LLM relay: lending a home machine's Claude CLI to a dyno.

Two things are under test. That the dyno can reach a model at all — the reason
screenshot import was dead in production — and that a *signed* request still
cannot make the home machine do anything it wasn't asked to. The second matters
more: the relay is reachable from the internet, and the app that signs for it is
one that accepts anonymous uploads.
"""

import base64
import json
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from engine import claude_cli
from relay import server as relay_server


@pytest.fixture
def keypair(monkeypatch):
    private = Ed25519PrivateKey.generate()
    private_raw = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_raw = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv("LLM_RELAY_PUBLIC_KEY", base64.b64encode(public_raw).decode())
    monkeypatch.setenv("LLM_RELAY_PRIVATE_KEY", base64.b64encode(private_raw).decode())
    return private


@pytest.fixture
def relay(keypair, monkeypatch, tmp_path):
    """A real relay on a real socket, with a stub CLI that echoes its argv."""
    fake_cli = tmp_path / "claude"
    fake_cli.write_text(
        "#!/bin/sh\n"
        "echo \"ARGS: $*\"\n"
        "cat >/dev/null\n"
        "ls . | sed 's/^/FILE: /'\n"
    )
    fake_cli.chmod(0o755)
    monkeypatch.setenv("FCPS_CLAUDE_BIN", str(fake_cli))

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), relay_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "LLM_RELAY_URL", f"http://127.0.0.1:{httpd.server_address[1]}"
    )
    relay_server._seen_nonces.clear()
    yield httpd
    httpd.shutdown()


def _call(prompt="hello", attachments=None):
    """Force the relay path by naming a binary that does not exist locally."""
    return claude_cli.run(
        ["/nonexistent/claude", "-p", "--model", "claude-sonnet-5"],
        input=prompt,
        timeout=30,
        attachments=attachments,
    )


# ------------------------------------------------------------- it works at all


def test_a_dyno_with_no_cli_reaches_a_model_through_the_relay(relay):
    result = _call()

    assert result.returncode == 0
    assert "--model claude-sonnet-5" in result.stdout


def test_an_image_travels_with_the_request(relay):
    """A path means nothing at the far end — the bytes have to go too. This is
    what makes screenshot import possible from a dyno at all."""
    result = _call(attachments={"squad.png": b"\x89PNG\r\n\x1a\n fake"})

    assert "FILE: squad.png" in result.stdout


def test_availability_reflects_the_relay_not_just_a_local_binary(
    keypair, monkeypatch
):
    monkeypatch.setenv("FCPS_CLAUDE_BIN", "/nonexistent/claude")
    monkeypatch.delenv("LLM_RELAY_URL", raising=False)
    assert claude_cli.is_available() is False

    monkeypatch.setenv("LLM_RELAY_URL", "https://relay.example")
    assert claude_cli.is_available() is True


def test_a_sleeping_home_machine_is_an_ordinary_outage(keypair, monkeypatch):
    """Callers already treat OSError as "no model available" and degrade. An
    unreachable relay must look like that, not like a crash."""
    monkeypatch.setenv("FCPS_CLAUDE_BIN", "/nonexistent/claude")
    # Port 9 is discard; nothing listens, so the connection is refused at once.
    monkeypatch.setenv("LLM_RELAY_URL", "http://127.0.0.1:9")

    with pytest.raises(OSError):
        _call()


# ------------------------------------------------------------------- it is safe


def test_an_unsigned_request_is_refused(relay):
    import urllib.error
    import urllib.request

    body = json.dumps({"args": ["-p"], "input": "hi"}).encode()
    request = urllib.request.Request(
        f"{claude_cli.relay_url()}/invoke", data=body, method="POST"
    )

    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=5)
    assert caught.value.code == 401


def test_a_tampered_body_is_refused(relay, keypair):
    """The signature covers the body digest. Signing only the envelope would
    let anyone who caught one request swap the prompt inside it."""
    import urllib.error
    import urllib.request

    honest = json.dumps({"args": ["-p"], "input": "hi"}).encode()
    timestamp, nonce = str(int(time.time())), "nonce-1"
    signature = claude_cli.sign(keypair, timestamp, nonce, honest)

    tampered = json.dumps({"args": ["-p"], "input": "something else"}).encode()
    request = urllib.request.Request(
        f"{claude_cli.relay_url()}/invoke",
        data=tampered,
        method="POST",
        headers={
            "X-Relay-Timestamp": timestamp,
            "X-Relay-Nonce": nonce,
            "X-Relay-Signature": signature,
        },
    )

    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=5)
    assert caught.value.code == 401


def test_a_replayed_request_is_refused(relay, keypair):
    import urllib.error
    import urllib.request

    body = json.dumps({"args": ["-p"], "input": "hi", "timeout": 20}).encode()
    timestamp, nonce = str(int(time.time())), "nonce-replay"
    headers = {
        "X-Relay-Timestamp": timestamp,
        "X-Relay-Nonce": nonce,
        "X-Relay-Signature": claude_cli.sign(keypair, timestamp, nonce, body),
    }

    def send():
        return urllib.request.urlopen(
            urllib.request.Request(
                f"{claude_cli.relay_url()}/invoke",
                data=body,
                method="POST",
                headers=headers,
            ),
            timeout=10,
        )

    assert send().status == 200
    with pytest.raises(urllib.error.HTTPError) as caught:
        send()
    assert caught.value.code == 401


@pytest.mark.parametrize(
    "args, why",
    [
        (["--allowedTools", "Bash"], "shell access"),
        (["--allowedTools", "Read,Bash"], "shell access hidden in a list"),
        (["--allowedTools", "Write"], "writing to the home machine"),
        (["--dangerously-skip-permissions"], "an unknown flag"),
        (["/etc/passwd"], "a bare path"),
    ],
)
def test_the_relay_refuses_what_it_was_not_asked_to_do(args, why):
    """Defence in depth: the app that signs these requests is one that accepts
    anonymous uploads, so a signature proves the caller is our dyno — not that
    our dyno has not been talked into something."""
    assert relay_server.validate_args(args) is not None, why


def test_the_ordinary_calls_the_app_makes_are_allowed():
    for args in (
        ["-p", "--model", "claude-sonnet-5", "--output-format", "text",
         "--allowedTools", "Read", "--add-dir", "/tmp/x"],
        ["-p", "--model", "claude-sonnet-5", "--effort", "medium",
         "--output-format", "json", "--system-prompt", "be brief",
         "--disallowedTools", "Bash"],
    ):
        assert relay_server.validate_args(args) is None, args


def test_an_attachment_cannot_escape_the_scratch_directory(relay):
    """A filename is attacker-controlled in the sense that matters: it comes
    from another machine. Only the basename is honoured."""
    result = _call(attachments={"../../escaped.png": b"x"})

    assert "FILE: escaped.png" in result.stdout
    assert ".." not in result.stdout


def test_add_dir_is_rewritten_to_the_relays_own_scratch(relay):
    """The dyno's paths mean nothing here, and honouring one would be the whole
    vulnerability — it is how a caller would name a directory to read."""
    result = claude_cli.run(
        ["/nonexistent/claude", "-p", "--add-dir", "/home/someone/.ssh"],
        input="x",
        timeout=30,
    )

    assert "/home/someone/.ssh" not in result.stdout
    assert "fpl-relay-" in result.stdout


def test_the_health_check_reveals_liveness_and_nothing_else(relay):
    import urllib.request

    with urllib.request.urlopen(
        f"{claude_cli.relay_url()}/health", timeout=5
    ) as response:
        body = json.loads(response.read())

    assert body == {"ok": True, "cli": True, "key": True}


def test_a_local_binary_is_never_bypassed_in_favour_of_the_relay(
    relay, tmp_path
):
    """A developer machine must not silently ship its work to a relay."""
    local = tmp_path / "local-claude"
    local.write_text("#!/bin/sh\necho LOCAL\ncat >/dev/null\n")
    local.chmod(0o755)

    result = claude_cli.run([str(local), "-p"], input="x", timeout=30)

    assert result.stdout.strip() == "LOCAL"


def test_the_relay_will_not_start_without_a_key(monkeypatch, capsys):
    monkeypatch.delenv("LLM_RELAY_PUBLIC_KEY", raising=False)

    assert relay_server.main() == 2
    assert "LLM_RELAY_PUBLIC_KEY" in capsys.readouterr().err


def test_keygen_produces_a_pair_that_verifies():
    """The two halves have to actually match, or every call 401s and the cause
    is invisible from either side."""
    generated = subprocess.run(
        [sys.executable, "relay/keygen.py"], capture_output=True, text=True, check=True
    ).stdout
    private_b64 = generated.split("LLM_RELAY_PRIVATE_KEY=")[1].split()[0]
    public_b64 = generated.split("LLM_RELAY_PUBLIC_KEY=")[1].split()[0]

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    private = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_b64))
    public = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_b64))
    body, timestamp, nonce = b"{}", "1700000000", "n"

    assert claude_cli.verify(
        public, timestamp, nonce, body, claude_cli.sign(private, timestamp, nonce, body)
    )
