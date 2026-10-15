"""Shared Gemini helpers used by every lab.

Setup:  put GEMINI_API_KEY=... in a .env file (in course/ or the folder above it),
        or `export GEMINI_API_KEY=...` in your terminal. Free key: https://aistudio.google.com

Labs call two functions:
    ask(prompt, system="")        → str
    parse(prompt, Schema)         → validated Pydantic object (structured output)
"""
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

_COURSE = Path(__file__).resolve().parents[1]
for env_file in (_COURSE / ".env", _COURSE.parent / ".env"):  # first match wins; real env vars are never overwritten
    load_dotenv(env_file)

MODEL = "gemini-3.8-flash"            # main model
FAST_MODEL = "gemini-3.5-flash-lite"  # cheaper model for high-volume work (synthetic data, judging)

client = genai.Client(  # reads GEMINI_API_KEY from the environment
    # Free tier allows only a few requests per minute and an agent makes one request per step.
    # On 429 (rate limit) or 5xx, wait and retry: 15s, 30s, 60s, 60s... instead of crashing.
    http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(
        attempts=8, initial_delay=15, max_delay=60, exp_base=2, http_status_codes=[429, 500, 503])),
)


def ask(prompt: str, system: str = "", fast: bool = False, max_tokens: int = 16000) -> str:
    """Single call → text. The simplest possible LLM 'program'."""
    response = client.models.generate_content(
        model=FAST_MODEL if fast else MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system or None, max_output_tokens=max_tokens),
    )
    return response.text or ""


def parse(prompt: str, schema: type[BaseModel], fast: bool = False, system: str = ""):
    """Structured output: the answer is forced into `schema` (JSON) and returned as a validated object."""
    response = client.models.generate_content(
        model=FAST_MODEL if fast else MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system or None,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
        ),
    )
    return schema.model_validate_json(response.text)
