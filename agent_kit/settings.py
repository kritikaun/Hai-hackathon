"""Run settings read from the environment (./helper loads .env)."""

from __future__ import annotations

import os
from pathlib import Path

_TRUE = {"on", "1", "true", "yes"}


def model() -> str:
    """Fallback default model (MODEL_API_MODEL, default muse-spark-1.3).

    Kept for callers without a task-specific setting (legal agent, document
    jobs, run records) so existing setups keep working.
    """
    return os.environ.get("MODEL_API_MODEL") or "muse-spark-1.3"


def agent_model() -> str:
    """The interactive family agent model (MODEL_AGENT, default muse-wonder-1.0)."""
    return os.environ.get("MODEL_AGENT") or "muse-wonder-1.0"


def verify_model() -> str:
    """The verification-judge model (MODEL_VERIFY, default muse-spark-1.3)."""
    return os.environ.get("MODEL_VERIFY") or "muse-spark-1.3"


def fast_model() -> str:
    """The fast model (MODEL_FAST, default muse-spark-1.3): short text jobs and
    the family loop's fallback when the agent model fails or times out."""
    return os.environ.get("MODEL_FAST") or "muse-spark-1.3"


def sam_model() -> str:
    """The segmentation model check_signatures uses (SAM_MODEL, default sam-3.1)."""
    return os.environ.get("SAM_MODEL") or "sam-3.1"


def image_model() -> str:
    """The image model invitations use (IMAGE_MODEL, default muse-image-1.0)."""
    return os.environ.get("IMAGE_MODEL") or "muse-image-1.0"


def transcribe_model() -> str:
    """The speech-to-text model (TRANSCRIBE_MODEL, default muse-voice-transcribe-1.0)."""
    return os.environ.get("TRANSCRIBE_MODEL") or "muse-voice-transcribe-1.0"


def verification_enabled() -> bool:
    """VERIFICATION=on: new runs get the verification layer (agent/verification/): the
    checks run at fixed points, three more tools, escalation, and attorney sign-off."""
    return os.environ.get("VERIFICATION", "off").strip().lower() in _TRUE


def case_dir(root: Path) -> Path:
    """The case folder (CASE_DIR, absolute or relative to the agent folder): objective.md and documents/."""
    value = os.environ.get("CASE_DIR", "").strip()
    if not value:
        raise RuntimeError("CASE_DIR is not set: point it at a case folder in .env")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path(root) / path).resolve()
