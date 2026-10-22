import time
import random
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# How many times to retry on rate-limit or transient errors
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds

# Wait between Gemini calls if a throttle is configured
_last_call_time: float = 0.0

def _throttle(min_seconds: float) -> None:
    # Sleep if needed to honor the minimum time between API calls
    global _last_call_time
    if min_seconds > 0:
        elapsed = time.time() - _last_call_time
        if elapsed < min_seconds:
            time.sleep(min_seconds - elapsed)
    _last_call_time = time.time()

def call_structured(prompt: str, model_class: Type[T], settings) -> T:
    # Call Gemini with structured output using google.genai and retry on transient errors
    from google import genai
    _throttle(settings.min_seconds_between_calls)
    client = genai.Client(api_key=settings.api_key)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=settings.chat_model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": model_class,
                },
            )
            if response.parsed is not None:
                return response.parsed
            return model_class.model_validate_json(response.text)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if any(k in error_str for k in ("rate", "quota", "429", "500", "503", "unavailable")):
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
            else:
                break
    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_error}")

def call_text(prompt: str, settings) -> str:
    # Call Gemini for a plain text response with retry on transient errors
    from google import genai
    _throttle(settings.min_seconds_between_calls)
    client = genai.Client(api_key=settings.api_key)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=settings.chat_model,
                contents=prompt,
            )
            return response.text or ""
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if any(k in error_str for k in ("rate", "quota", "429", "500", "503", "unavailable")):
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
            else:
                break
    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_error}")
