"""Vision calls, through the same Claude CLI the other model features use.

Same binary, same subscription, same budget ledger as :mod:`engine.fcps_llm` —
so a screenshot upload competes for the operator's rate-limit window with
everything else rather than having a private allowance nobody is watching.

The one structural difference: this call needs the model to *see* a file, so
the Read tool is allowed where the text features deny every tool. It is scoped
to a per-call temporary directory containing exactly one image, and every other
tool stays denied. Bash in particular is never available — an anonymous upload
route that can run shell commands is not a feature.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import claude_cli
from . import llm_budget

TIMEOUT_SECONDS = int(os.getenv("VISION_TIMEOUT_SECONDS", 120))

# Read is required — the model cannot look at an image it cannot open. Nothing
# else is: the scratch directory holds one file and there is nothing to search,
# write, fetch or execute.
_ALLOWED_TOOLS = "Read"


class VisionUnavailable(Exception):
    """The model could not be reached, or declined. Never a crash."""

    def __init__(self, code: str, message: str, status: int = 502):
        self.code = code
        self.message = message
        self.status = status
        super().__init__(message)

    def to_dict(self) -> dict:
        return {"code": self.code, "error": self.message}


def _binary() -> str | None:
    """The CLI this call will use as argv[0].

    Falls back to the bare name when only the relay is available: the relay
    substitutes its own binary and ignores this, but the argv still has to be
    well-formed on the way out.
    """
    if claude_cli.local_binary() is not None:
        return claude_cli.local_binary()
    return "claude" if claude_cli.relay_configured() else None


def is_configured() -> bool:
    """Whether a model is reachable — locally, or over the relay.

    This is what `/api/import/config` reports, and therefore what decides
    whether the upload button is shown at all. Before the relay it was false on
    every dyno, which is why the feature was dead in production.
    """
    return claude_cli.is_available()


def model_name() -> str:
    return os.getenv("VISION_MODEL", "claude-sonnet-5").strip() or "claude-sonnet-5"


def read_image(image: bytes, prompt: str, suffix: str = ".png") -> str:
    """Show the model one image and return its raw text response.

    Raises :class:`VisionUnavailable` for every failure mode, so a caller on an
    HTTP route never has to catch a subprocess exception.
    """
    binary = _binary()
    if binary is None:
        raise VisionUnavailable(
            "vision_not_configured",
            "Screenshot import needs the Claude CLI on the server.",
            status=503,
        )

    try:
        with llm_budget.reserve("vision"), tempfile.TemporaryDirectory(
            prefix="fpl-vision-"
        ) as scratch:
            image_path = Path(scratch) / f"squad{suffix}"
            image_path.write_bytes(image)

            completed = claude_cli.run(
                [
                    binary,
                    "-p",
                    "--model",
                    model_name(),
                    "--output-format",
                    "text",
                    "--allowedTools",
                    _ALLOWED_TOOLS,
                    "--add-dir",
                    scratch,
                ],
                # Named relative to the working directory, never absolutely.
                # The scratch directory is the cwd on both sides, but it is a
                # *different* directory on each: run through the relay, an
                # absolute path from this machine does not exist on the one
                # holding the file, and the model is left guessing. It
                # sometimes recovered by listing the directory and sometimes
                # reported it could not find the image — which surfaced as an
                # intermittent "no player names could be read".
                input=f"{prompt}\n\nThe image is the file ./{image_path.name}\n",
                timeout=TIMEOUT_SECONDS,
                cwd=scratch,
                # The relay runs on another machine, so the bytes have to travel
                # with the request — a path means nothing at the far end.
                attachments={image_path.name: image},
            )
    except llm_budget.BudgetExhausted as error:
        raise VisionUnavailable("vision_budget_exhausted", str(error), 429) from error
    except llm_budget.TooBusy as error:
        raise VisionUnavailable("vision_busy", str(error), 503) from error
    except subprocess.TimeoutExpired as error:
        raise VisionUnavailable(
            "vision_timeout",
            f"Reading the screenshot timed out after {TIMEOUT_SECONDS}s.",
            504,
        ) from error
    except OSError as error:
        raise VisionUnavailable(
            "vision_upstream_error",
            f"The Claude CLI could not be run: {type(error).__name__}.",
        ) from error

    if completed.returncode != 0:
        # stderr can contain paths and account details; the code is enough for
        # the client and the detail belongs in the server log.
        print(f"[vision] exit {completed.returncode}: {completed.stderr[:400]}", flush=True)
        raise VisionUnavailable(
            "vision_upstream_error", "The model could not read that screenshot."
        )
    return completed.stdout or ""
