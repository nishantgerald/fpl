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
    override = os.getenv("FCPS_CLAUDE_BIN", "").strip()
    if override:
        return override if Path(override).exists() else None
    return shutil.which("claude")


def is_configured() -> bool:
    return _binary() is not None


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

            completed = subprocess.run(
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
                input=f"{prompt}\n\nThe image is at: {image_path}\n",
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=scratch,
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
        print(f"[vision] exit {completed.returncode}: {completed.stderr[:400]}")
        raise VisionUnavailable(
            "vision_upstream_error", "The model could not read that screenshot."
        )
    return completed.stdout or ""
