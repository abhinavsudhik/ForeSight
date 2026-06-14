"""
Gemini Client Helper for ForeSight.
Manages primary and fallback API keys (Gemini / Groq), providing fallback client
initialization and execution mechanisms if the primary key is rate-limited or missing.
"""

import os
import base64
import logging
import requests
from io import BytesIO
from PIL import Image
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY_FALLBACK")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Groq fallback models
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

class GroqResponse:
    """Wrapper to mimic the Gemini Response structure (specifically .text)."""
    def __init__(self, text: str):
        self.text = text

def is_gemini_available() -> bool:
    """Check if either the primary Gemini API key or Groq fallback API key is configured."""
    return bool(GEMINI_API_KEY and GEMINI_API_KEY.strip()) or bool(GROQ_API_KEY and GROQ_API_KEY.strip())

def _is_groq_key(key: str) -> bool:
    return key.strip().startswith("gsk_")

def _call_groq(contents: list, **kwargs) -> GroqResponse:
    """
    Call Groq API using standard HTTP POST requests.
    """
    key = (GROQ_API_KEY or "").strip()
    if not key:
        raise Exception("Out of tokens / Rate limit exceeded. Groq API key is not configured.")

    # 1. Determine if this is a vision request
    is_vision = False
    text_prompt = ""
    image_data_url = None

    for item in contents:
        if isinstance(item, Image.Image):
            is_vision = True
            # Convert PIL Image to Base64 JPEG URL
            buffered = BytesIO()
            item.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data_url = f"data:image/jpeg;base64,{img_b64}"
        elif isinstance(item, str):
            text_prompt += item + "\n"

    text_prompt = text_prompt.strip()

    # 2. Select appropriate model and payload
    if is_vision:
        model = GROQ_VISION_MODEL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ]
    else:
        model = GROQ_TEXT_MODEL
        messages = [
            {
                "role": "user",
                "content": text_prompt
            }
        ]

    # 3. Call the API (single attempt, no retry)
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0
    }

    try:
        logger.info("Calling Groq API (model: %s)...", model)
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            return GroqResponse(content)
        else:
            logger.error("Groq API error (status code %d): %s", response.status_code, response.text)
            raise Exception("Out of tokens")
    except Exception as exc:
        logger.error("Groq API call failed: %s", exc)
        raise Exception("Out of tokens / Rate limit exceeded. Please check API quotas or try again later.") from exc

def generate_content(contents: list, model: str = None, **kwargs):
    """
    Call models.generate_content using GEMINI_API_KEY.
    If it fails, call using Groq fallback (or Gemini fallback key if not a Groq key).
    No retry loops. Try primary first, then fallback.
    """
    model_name = model or GEMINI_MODEL
    primary_key = (GEMINI_API_KEY or "").strip()
    fallback_key = (GROQ_API_KEY or "").strip()

    # If primary key is empty, try fallback directly
    if not primary_key:
        if fallback_key:
            logger.warning("Primary GEMINI_API_KEY is not set. Trying fallback key directly.")
            return _execute_with_fallback(contents, fallback_key, model_name, **kwargs)
        else:
            raise Exception("Out of tokens / Rate limit exceeded. No API keys are configured.")

    # Try primary key first
    try:
        logger.info("Initializing Gemini Client with primary API key.")
        client = genai.Client(api_key=primary_key)
        return client.models.generate_content(model=model_name, contents=contents, **kwargs)
    except Exception as exc:
        logger.warning("Gemini primary key failed: %s. Attempting fallback...", exc)
        if fallback_key:
            try:
                return _execute_with_fallback(contents, fallback_key, model_name, **kwargs)
            except Exception as exc_fallback:
                raise exc_fallback
        else:
            logger.error("Primary key failed and no fallback key is configured.")
            raise Exception("Out of tokens / Rate limit exceeded. Please check API quotas or try again later.") from exc

def _execute_with_fallback(contents: list, fallback_key: str, model_name: str, **kwargs):
    """Determine client type and call fallback API."""
    if _is_groq_key(fallback_key):
        return _call_groq(contents, **kwargs)
    else:
        logger.info("Initializing Gemini Client with fallback API key.")
        client = genai.Client(api_key=fallback_key)
        try:
            return client.models.generate_content(model=model_name, contents=contents, **kwargs)
        except Exception as exc:
            logger.error("Gemini fallback key failed: %s", exc)
            raise Exception("Out of tokens / Rate limit exceeded. Please check API quotas or try again later.") from exc
