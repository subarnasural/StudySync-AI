import logging
import os
from typing import List

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_AVAILABLE_MODELS_CACHE: dict = {}
_FALLBACK_LLM_SINGLETON = None
_EMBEDDINGS_SINGLETON = None


def get_api_keys() -> List[str]:
    """Retrieve multiple API keys from .env separated by comma."""
    keys_str = os.getenv("GEMINI_API_KEYS")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if keys:
            return keys

    # Fallback to single key setup
    single_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if single_key:
        return [single_key.strip()]

    raise ValueError(
        "Missing Gemini API keys. Set GEMINI_API_KEYS (comma separated) "
        "or GEMINI_API_KEY environment variable."
    )


def get_chat_model_names() -> List[str]:
    """Resolve chat model priority from env with safe defaults."""
    default_models = [
        "gemini-flash-latest",
        "gemini-3.6-flash",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-pro-latest",
    ]

    configured = os.getenv("GEMINI_CHAT_MODELS", "")
    models = [m.strip() for m in configured.split(",") if m.strip()]
    if not models:
        models = list(default_models)

    preferred = os.getenv("GEMINI_CHAT_MODEL", "").strip()
    if preferred:
        if preferred in models:
            models.remove(preferred)
        models.insert(0, preferred)

    # Deduplicate while preserving order; strip "models/" prefix if present.
    deduped: List[str] = []
    seen: set = set()
    for m in models:
        if m.startswith("models/"):
            m = m.split("/", 1)[1]
        if m not in seen:
            deduped.append(m)
            seen.add(m)
    return deduped


def get_available_generate_models(api_key: str) -> List[str]:
    """Query the Gemini API to list models that support generateContent."""
    if api_key in _AVAILABLE_MODELS_CACHE:
        return _AVAILABLE_MODELS_CACHE[api_key]

    try:
        from google import genai  # type: ignore[import]

        client = genai.Client(api_key=api_key)
        available = []
        for model in client.models.list():
            name = getattr(model, "name", "")
            if name.startswith("models/"):
                name = name.split("/", 1)[1]

            methods = (
                getattr(model, "supported_actions", None)
                or getattr(model, "supported_generation_methods", None)
                or []
            )
            if any("generate" in str(m).lower() for m in methods):
                available.append(name)

        _AVAILABLE_MODELS_CACHE[api_key] = available
        return available
    except Exception as exc:
        logger.warning("Model discovery failed: %s", exc)
        _AVAILABLE_MODELS_CACHE[api_key] = []
        return []


def _should_discover_models() -> bool:
    """Model discovery adds startup latency; opt-in via env flag."""
    flag = os.getenv("GEMINI_DISCOVER_MODELS", "false").strip().lower()
    return flag in {"1", "true", "yes", "on"}


class GeminiFallbackLLM:
    """
    Thin wrapper around ChatGoogleGenerativeAI that automatically retries
    across multiple model names and API keys on recoverable errors.
    """

    def __init__(self) -> None:
        keys = get_api_keys()
        models = get_chat_model_names()
        self.clients: List[dict] = []

        # Fallback list – prioritize verified models with high quota
        fallback_priority = [
            "gemini-flash-latest",
            "gemini-3.6-flash",
            "gemini-flash-lite-latest",
            "gemini-3.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-pro-latest",
        ]

        for key in keys:
            available = (
                set(get_available_generate_models(key))
                if _should_discover_models()
                else set()
            )

            if available:
                selected_models = [m for m in models if m in available]
                if not selected_models:
                    selected_models = [m for m in fallback_priority if m in available]
            else:
                selected_models = models

            for model in selected_models:
                self.clients.append(
                    {
                        "model": model,
                        "client": ChatGoogleGenerativeAI(
                            model=model,
                            google_api_key=key,
                            max_retries=1,
                        ),
                    }
                )

    def invoke(self, prompt: str):
        """Invoke the LLM, retrying across model/key combinations on failure."""
        last_error = None

        for entry in self.clients:
            model = entry["model"]
            client = entry["client"]
            try:
                return client.invoke(prompt)
            except Exception as e:
                last_error = e
                message = str(e).lower()

                # Retry on common recoverable API and model-availability failures.
                recoverable = any(
                    token in message
                    for token in [
                        "not_found",
                        "not found",
                        "resource_exhausted",
                        "quota",
                        "rate",
                        "429",
                        "timeout",
                        "unavailable",
                        "503",
                        "internal",
                        "permission",
                        "403",
                        "401",
                    ]
                )

                if recoverable:
                    logger.warning("Model/key fallback triggered from %s: %s", model, e)
                    continue

                # Unknown error — still try the remaining options.
                logger.error("Model invoke error on %s: %s", model, e)
                continue

        if last_error:
            raise RuntimeError(
                f"All configured Gemini models/keys failed. Last error: {last_error}"
            )
        raise RuntimeError("No Gemini clients were initialized.")


def get_fallback_llm() -> GeminiFallbackLLM:
    """Return the shared GeminiFallbackLLM singleton."""
    global _FALLBACK_LLM_SINGLETON
    if _FALLBACK_LLM_SINGLETON is None:
        _FALLBACK_LLM_SINGLETON = GeminiFallbackLLM()
    return _FALLBACK_LLM_SINGLETON


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Return the shared HuggingFace embedding model singleton."""
    global _EMBEDDINGS_SINGLETON
    if _EMBEDDINGS_SINGLETON is None:
        _EMBEDDINGS_SINGLETON = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _EMBEDDINGS_SINGLETON