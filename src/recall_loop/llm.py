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
    # Call Gemini with structured output and retry on transient errors
    from langchain_google_genai import ChatGoogleGenerativeAI
    _throttle(settings.min_seconds_between_calls)
    llm = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.api_key,
    )
    structured_llm = llm.with_structured_output(model_class)
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            result = structured_llm.invoke(prompt)
            return result
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Retry on rate limits or transient server errors
            if "rate" in error_str or "quota" in error_str or "429" in error_str or "500" in error_str or "503" in error_str:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
            else:
                break
    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_error}")

def call_text(prompt: str, settings) -> str:
    # Call Gemini for a plain text response with retry on transient errors
    from langchain_google_genai import ChatGoogleGenerativeAI
    _throttle(settings.min_seconds_between_calls)
    llm = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.api_key,
    )
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            result = llm.invoke(prompt)
            return result.content
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "rate" in error_str or "quota" in error_str or "429" in error_str or "500" in error_str or "503" in error_str:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
            else:
                break
    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_error}")
