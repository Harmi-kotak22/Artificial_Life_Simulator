"""Local RAG utilities for FAQ retrieval.

This module implements a lightweight local retriever using TF-IDF + cosine
similarity so the app can retrieve grounded context without external services.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalHit:
    text: str
    score: float


def load_faq_chunks(corpus_path: str | Path) -> List[str]:
    """Load and split corpus text into retrieval chunks.

    Splits on blank lines and keeps reasonably informative passages.
    """
    path = Path(corpus_path)
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    raw_chunks = [c.strip() for c in text.split("\n\n")]

    # Keep chunks with sufficient content for retrieval and skip policy text
    # that can cause generic non-answers.
    chunks = [
        c
        for c in raw_chunks
        if len(c) >= 40
        and not c.upper().startswith("SECTION: CHATBOT USAGE POLICY")
    ]
    return chunks


class LocalFAQRetriever:
    """Simple local retriever for FAQ corpus."""

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.chunk_matrix = None

        if self.chunks:
            self.chunk_matrix = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.08) -> List[Dict[str, float | str]]:
        """Return top matching chunks with cosine scores."""
        if not query.strip() or self.chunk_matrix is None:
            return []

        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.chunk_matrix).flatten()

        ranked_idx = scores.argsort()[::-1]
        hits: List[Dict[str, float | str]] = []

        for idx in ranked_idx[:top_k]:
            score = float(scores[idx])
            if score < min_score:
                continue
            hits.append({"text": self.chunks[idx], "score": score})

        return hits


def _read_key_from_dotenv(key_name: str, dotenv_path: Path) -> Optional[str]:
    """Read a key from .env without requiring python-dotenv dependency."""
    if not dotenv_path.exists():
        return None

    try:
        # utf-8-sig removes BOM automatically if present.
        for line in dotenv_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().replace("\ufeff", "")
            if k != key_name:
                continue
            value = v.strip().strip('"').strip("'")
            return value or None
    except Exception:
        return None

    return None


def resolve_gemini_api_key() -> Optional[str]:
    """Resolve Gemini key from environment or project .env file."""
    key = sanitize_api_key(os.getenv("GEMINI_API_KEY"))
    if key:
        try:
            f"Bearer {key}".encode("latin-1")
            return key
        except UnicodeEncodeError:
            # Fall through to .env when shell env contains masked/non-ASCII chars.
            pass

    # Project root is parent of app/ directory.
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    return sanitize_api_key(_read_key_from_dotenv("GEMINI_API_KEY", dotenv_path))


def sanitize_api_key(raw_key: Optional[str]) -> Optional[str]:
    """Normalize API key text and remove common hidden formatting artifacts."""
    if not raw_key:
        return None

    key = raw_key.strip()

    # Remove common smart quotes and zero-width characters copied from docs/chats.
    replacements = {
        "\u201c": "",
        "\u201d": "",
        "\u2018": "",
        "\u2019": "",
        "\ufeff": "",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\u2060": "",
    }
    for bad, good in replacements.items():
        key = key.replace(bad, good)

    # Strip wrapping quotes if present.
    key = key.strip().strip('"').strip("'")

    return key or None


def validate_gemini_api_key(api_key: Optional[str]) -> Tuple[bool, str]:
    """Validate Gemini key formatting for API usage."""
    if not api_key:
        return False, "GEMINI_API_KEY missing"

    if "\n" in api_key or "\r" in api_key:
        return False, "GEMINI_API_KEY contains newline characters"

    if not api_key.isascii():
        return False, "GEMINI_API_KEY contains non-ASCII characters"

    return True, "ok"


def resolve_gemini_api_key_with_source() -> Tuple[Optional[str], str]:
    """Resolve Gemini key and return source label for debugging."""
    key = sanitize_api_key(os.getenv("GEMINI_API_KEY"))
    if key:
        try:
            f"Bearer {key}".encode("latin-1")
            return key, "env"
        except UnicodeEncodeError:
            pass

    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    key = sanitize_api_key(_read_key_from_dotenv("GEMINI_API_KEY", dotenv_path))
    if key:
        return key, ".env"

    return None, "missing"


def _detect_alt_llm_keys() -> List[str]:
    """Detect non-Gemini keys to help diagnose provider mismatch quickly."""
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    alt_names = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    found = []

    for name in alt_names:
        if os.getenv(name):
            found.append(name)
            continue
        if _read_key_from_dotenv(name, dotenv_path):
            found.append(name)

    return found


def gemini_key_diagnostics() -> Dict[str, object]:
    """Return safe diagnostics about Gemini key resolution (no secret values)."""
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    env_raw = os.getenv("GEMINI_API_KEY")
    dotenv_raw = _read_key_from_dotenv("GEMINI_API_KEY", dotenv_path)

    env_key = sanitize_api_key(env_raw)
    dotenv_key = sanitize_api_key(dotenv_raw)

    return {
        "dotenv_path": str(dotenv_path),
        "dotenv_exists": dotenv_path.exists(),
        "env_key_present": bool(env_key),
        "dotenv_key_present": bool(dotenv_key),
        "env_key_len": len(env_key) if env_key else 0,
        "dotenv_key_len": len(dotenv_key) if dotenv_key else 0,
    }


def _gemini_model_candidates(preferred_model: str) -> List[str]:
    """Return ordered Gemini model candidates for generateContent calls."""
    defaults = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.5-pro-latest",
        "gemini-1.5-pro",
    ]
    ordered = [preferred_model] + defaults

    # Preserve order while removing duplicates.
    unique = []
    seen = set()
    for m in ordered:
        if m not in seen:
            unique.append(m)
            seen.add(m)
    return unique


def _list_gemini_generate_models(api_key: str, api_version: str = "v1beta") -> List[str]:
    """List models that support generateContent for the provided API key."""
    try:
        response = requests.get(
            f"https://generativelanguage.googleapis.com/{api_version}/models",
            params={"key": api_key},
            timeout=20,
        )
        if response.status_code != 200:
            return []

        data = response.json()
        models = data.get("models", [])
        supported = []
        for model in models:
            methods = model.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue
            name = model.get("name", "")
            # API returns names like "models/gemini-1.5-flash".
            if name.startswith("models/"):
                name = name.split("models/", 1)[1]
            if name:
                supported.append(name)
        return supported
    except Exception:
        return []


def _build_model_attempts(api_key: str, preferred_model: str) -> List[Tuple[str, str]]:
    """Build ordered (api_version, model) attempts from discovered + fallback models."""
    discovered_v1beta = _list_gemini_generate_models(api_key, "v1beta")
    discovered_v1 = _list_gemini_generate_models(api_key, "v1")

    fallback = _gemini_model_candidates(preferred_model)

    attempts: List[Tuple[str, str]] = []
    seen = set()

    for model in fallback:
        key = ("v1beta", model)
        if key not in seen:
            attempts.append(key)
            seen.add(key)
        key = ("v1", model)
        if key not in seen:
            attempts.append(key)
            seen.add(key)

    for model in discovered_v1beta:
        key = ("v1beta", model)
        if key not in seen:
            attempts.insert(0, key)
            seen.add(key)

    for model in discovered_v1:
        key = ("v1", model)
        if key not in seen:
            attempts.insert(0, key)
            seen.add(key)

    return attempts


def _gemini_generate_content(
    api_key: str,
    model: str,
    api_version: str,
    system_prompt: str,
    user_prompt: str,
    timeout_sec: int,
) -> Tuple[Optional[str], int, str]:
    """Call Gemini generateContent and return (answer, status_code, error_text)."""
    merged_prompt = (
        f"System instructions:\n{system_prompt}\n\n"
        f"User request:\n{user_prompt}"
    )

    # Keep payload minimal for broad compatibility across Gemini model variants.
    payload = {
        "contents": [
            {
                "parts": [{"text": merged_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
        },
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(
        f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent",
        params={"key": api_key},
        json=payload,
        headers=headers,
        timeout=timeout_sec,
    )

    if response.status_code != 200:
        return None, response.status_code, response.text[:220]

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return None, response.status_code, "No candidates returned"

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [p.get("text", "") for p in parts if p.get("text")]
    content = "\n".join(text_parts).strip()
    if not content:
        return None, response.status_code, "Empty content from model"

    return content, response.status_code, ""


def generate_rag_answer(
    query: str,
    retrieved_chunks: List[Dict[str, float | str]],
    model: str = "gemini-1.5-flash",
    timeout_sec: int = 25,
) -> Optional[str]:
    """Generate grounded answer using Gemini generateContent API.

    Returns None when API key is missing, retrieval is empty, or request fails.
    """
    if not query.strip() or not retrieved_chunks:
        return None

    api_key = sanitize_api_key(resolve_gemini_api_key())
    if not api_key:
        return None

    ok, _ = validate_gemini_api_key(api_key)
    if not ok:
        return None

    context_block = "\n\n".join(
        [f"Context {i+1}: {str(hit['text']).strip()}" for i, hit in enumerate(retrieved_chunks)]
    )

    system_prompt = (
        "You are an FAQ assistant for a disease forecasting application. "
        "Answer only from the provided context. If the context is insufficient, "
        "say so clearly and ask the user to refine the question. Keep answers concise and practical."
    )
    user_prompt = (
        f"User question:\n{query}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        "Instructions: Give a direct answer and include 2-4 short bullet points when useful."
    )

    try:
        for api_version, candidate_model in _build_model_attempts(api_key, model):
            content, status, _ = _gemini_generate_content(
                api_key=api_key,
                model=candidate_model,
                api_version=api_version,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_sec=timeout_sec,
            )
            if status == 200 and content:
                return content
    except Exception:
        return None


def generate_rag_answer_with_debug(
    query: str,
    retrieved_chunks: List[Dict[str, float | str]],
    model: str = "gemini-1.5-flash",
    timeout_sec: int = 25,
) -> Dict[str, object]:
    """Generate answer and return debug metadata for troubleshooting."""
    debug: Dict[str, object] = {
        "mode": "unknown",
        "model": model,
        "model_used": None,
        "models_tried": [],
        "api_versions_tried": [],
        "retrieved_count": len(retrieved_chunks),
        "api_key_source": "missing",
        "api_http_status": None,
        "error": None,
        "alt_keys_detected": [],
    }

    if not query.strip():
        debug["mode"] = "no-query"
        debug["error"] = "Empty query"
        return {"answer": None, "debug": debug}

    if not retrieved_chunks:
        debug["mode"] = "no-retrieval"
        debug["error"] = "No retrieved context"
        return {"answer": None, "debug": debug}

    api_key, source = resolve_gemini_api_key_with_source()
    api_key = sanitize_api_key(api_key)
    debug["api_key_source"] = source

    if not api_key:
        debug["mode"] = "missing-key"
        debug["alt_keys_detected"] = _detect_alt_llm_keys()
        if debug["alt_keys_detected"]:
            debug["error"] = "GEMINI_API_KEY missing; found other LLM keys"
        else:
            debug["error"] = "GEMINI_API_KEY missing"
        return {"answer": None, "debug": debug}

    ok, msg = validate_gemini_api_key(api_key)
    if not ok:
        debug["mode"] = "invalid-key-format"
        debug["error"] = msg
        return {"answer": None, "debug": debug}

    context_block = "\n\n".join(
        [f"Context {i+1}: {str(hit['text']).strip()}" for i, hit in enumerate(retrieved_chunks)]
    )
    system_prompt = (
        "You are an FAQ assistant for a disease forecasting application. "
        "Answer only from the provided context. If the context is insufficient, "
        "say so clearly and ask the user to refine the question. Keep answers concise and practical."
    )
    user_prompt = (
        f"User question:\n{query}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        "Instructions: Give a direct answer and include 2-4 short bullet points when useful."
    )

    try:
        last_error = None
        last_status = None

        for api_version, candidate_model in _build_model_attempts(api_key, model):
            debug["models_tried"].append(candidate_model)
            debug["api_versions_tried"].append(api_version)
            content, status, err = _gemini_generate_content(
                api_key=api_key,
                model=candidate_model,
                api_version=api_version,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_sec=timeout_sec,
            )

            last_status = status
            debug["api_http_status"] = status

            if status == 200 and content:
                debug["mode"] = "llm"
                debug["model_used"] = candidate_model
                debug["api_version_used"] = api_version
                return {"answer": content, "debug": debug}

            # 404 model-not-found: try next model automatically.
            last_error = err

        debug["mode"] = "api-error"
        debug["error"] = last_error or "No successful Gemini model response"
        debug["api_http_status"] = last_status
        return {"answer": None, "debug": debug}
    except Exception as exc:
        debug["mode"] = "exception"
        debug["error"] = str(exc)
        return {"answer": None, "debug": debug}
