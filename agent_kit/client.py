"""The Meta Model API client: the official OpenAI Python SDK, pointed at api.meta.ai.

    from agent_kit.client import make_client
    client = make_client()
    response = client.responses.create(model="muse-spark-1.3", input="Hello")
    print(response.output_text)

The API key stays inside the client; traces never see it. The SDK retries rate
limits (429) and overloads (503) on its own, waiting as long as the API asks.
"""

from __future__ import annotations

import os

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
MAX_RETRIES = 4
TIMEOUT_SECONDS = 180
# Interactive family chat: fail fast so the loop can fall back to MODEL_FAST.
# No SDK retry on the agent model: the first timeout goes straight to fallback.
INTERACTIVE_MAX_RETRIES = 0
INTERACTIVE_TIMEOUT_SECONDS = 25


class MissingKeyError(RuntimeError):
    pass


def make_client(
    timeout_seconds: float = TIMEOUT_SECONDS, max_retries: int = MAX_RETRIES
) -> OpenAI:
    """An OpenAI client for the Meta Model API, from MODEL_API_KEY and MODEL_API_BASE_URL.

    Defaults (180s, 4 retries) suit background and legal jobs. Interactive
    family chat passes INTERACTIVE_TIMEOUT_SECONDS / INTERACTIVE_MAX_RETRIES.
    """
    key = os.environ.get("MODEL_API_KEY", "")
    if not key:
        raise MissingKeyError("MODEL_API_KEY is not set: add it to agent/.env")
    return OpenAI(
        api_key=key,
        base_url=os.environ.get("MODEL_API_BASE_URL") or DEFAULT_BASE_URL,
        max_retries=max_retries,
        timeout=timeout_seconds,
    )
