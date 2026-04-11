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


def resolve_groq_api_key() -> Optional[str]:
    """Resolve Groq key from environment or project .env file."""
    key = sanitize_api_key(os.getenv("groq-api-key"))
    if key:
        try:
            f"Bearer {key}".encode("latin-1")
            return key
        except UnicodeEncodeError:
            # Fall through to .env when shell env contains masked/non-ASCII chars.
            pass

    # Project root is parent of app/ directory.
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    return sanitize_api_key(_read_key_from_dotenv("groq-api-key", dotenv_path))


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


def validate_groq_api_key(api_key: Optional[str]) -> Tuple[bool, str]:
    """Validate Groq key formatting for API usage."""
    if not api_key:
        return False, "groq-api-key missing"

    if "\n" in api_key or "\r" in api_key:
        return False, "groq-api-key contains newline characters"

    if not api_key.isascii():
        return False, "groq-api-key contains non-ASCII characters"

    return True, "ok"


def resolve_groq_api_key_with_source() -> Tuple[Optional[str], str]:
    """Resolve Groq key and return source label for debugging."""
    key = sanitize_api_key(os.getenv("groq-api-key"))
    if key:
        try:
            f"Bearer {key}".encode("latin-1")
            return key, "env"
        except UnicodeEncodeError:
            pass

    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    key = sanitize_api_key(_read_key_from_dotenv("groq-api-key", dotenv_path))
    if key:
        return key, ".env"

    return None, "missing"


def _detect_alt_llm_keys() -> List[str]:
    """Detect non-Groq keys to help diagnose provider mismatch quickly."""
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


def groq_key_diagnostics() -> Dict[str, object]:
    """Return safe diagnostics about Groq key resolution (no secret values)."""
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    env_raw = os.getenv("groq-api-key")
    dotenv_raw = _read_key_from_dotenv("groq-api-key", dotenv_path)

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


def _groq_model_candidates(preferred_model: str) -> List[str]:
    """Return ordered Groq model candidates for chat completions."""
    defaults = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
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


def _list_groq_generate_models(api_key: str, api_version: str = "v1beta") -> List[str]:
    """List available Groq chat models for the provided API key."""
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
        )
        if response.status_code != 200:
            return []

        data = response.json()
        models = data.get("data", [])
        supported = []
        for model in models:
            name = str(model.get("id", "")).strip()
            if name:
                supported.append(name)
        return supported
    except Exception:
        return []


def _build_model_attempts(api_key: str, preferred_model: str) -> List[Tuple[str, str]]:
    """Build ordered (api_style, model) attempts from discovered + fallback models."""
    discovered = _list_groq_generate_models(api_key)

    fallback = _groq_model_candidates(preferred_model)

    attempts: List[Tuple[str, str]] = []
    seen = set()

    for model in fallback:
        key = ("openai", model)
        if key not in seen:
            attempts.append(key)
            seen.add(key)

    for model in discovered:
        key = ("openai", model)
        if key not in seen:
            attempts.insert(0, key)
            seen.add(key)

    return attempts


def _groq_generate_content(
    api_key: str,
    model: str,
    api_version: str,
    system_prompt: str,
    user_prompt: str,
    timeout_sec: int,
) -> Tuple[Optional[str], int, str]:
    """Call Groq chat completions API and return (answer, status_code, error_text)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 700,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout_sec,
    )

    if response.status_code != 200:
        return None, response.status_code, response.text[:220]

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return None, response.status_code, "No choices returned"

    content = str(choices[0].get("message", {}).get("content", "")).strip()
    if not content:
        return None, response.status_code, "Empty content from model"

    return content, response.status_code, ""


def _extract_recent_user_queries(
    conversation_history: Optional[List[Dict[str, str]]],
    max_queries: int = 5,
) -> List[str]:
    """Extract last N user messages for lightweight conversation awareness."""
    if not conversation_history:
        return []

    user_msgs = [
        str(m.get("text", "")).strip()
        for m in conversation_history
        if isinstance(m, dict) and m.get("role") == "user" and str(m.get("text", "")).strip()
    ]
    return user_msgs[-max_queries:]


def _dedupe_and_filter_hits(
    hits: List[Dict[str, float | str]],
    min_score: float,
) -> List[Dict[str, float | str]]:
    """Remove low-quality/noisy duplicates from retrieval results."""
    cleaned: List[Dict[str, float | str]] = []
    seen_texts = set()

    for h in hits:
        text = str(h.get("text", "")).strip()
        score = float(h.get("score", 0.0))

        if not text or len(text) < 40:
            continue
        if score < min_score:
            continue

        normalized = " ".join(text.lower().split())
        if normalized in seen_texts:
            continue

        seen_texts.add(normalized)
        cleaned.append({"text": text, "score": score})

    return cleaned


def retrieve_context(
    query: str,
    retriever: LocalFAQRetriever,
    top_k: int = 5,
    min_score: float = 0.18,
) -> List[Dict[str, float | str]]:
    """Retrieve relevant context chunks with filtering for hybrid RAG."""
    if not query.strip() or retriever is None:
        return []

    raw_hits = retriever.retrieve(query, top_k=top_k, min_score=min_score)
    return _dedupe_and_filter_hits(raw_hits, min_score=min_score)


def _build_hybrid_prompts(
    query: str,
    retrieved_chunks: List[Dict[str, float | str]],
    recent_user_queries: List[str],
    allow_general_knowledge: bool,
) -> Tuple[str, str]:
    """Build hybrid prompts that prefer context but allow helpful reasoning."""
    recent_block = "\n".join([f"- {q}" for q in recent_user_queries]) if recent_user_queries else "- (no prior user turns)"

    context_block = "\n\n".join(
        [f"Context {i+1} (score={float(hit.get('score', 0.0)):.3f}):\n{str(hit.get('text', '')).strip()}" for i, hit in enumerate(retrieved_chunks)]
    )
    if not context_block:
        context_block = "(No retrieved context)"

    if allow_general_knowledge:
        knowledge_rule = (
            "Use retrieved context as the primary source when relevant. "
            "If context is weak or missing, answer using your general knowledge clearly and helpfully. "
            "Do not hallucinate specific dataset values not present in context."
        )
    else:
        knowledge_rule = (
            "Use retrieved context as the primary source. "
            "If context is insufficient, provide a best-effort general explanation and mention uncertainty."
        )

    system_prompt = (
        "You are an expert disease forecasting assistant for a Streamlit application. "
        "Give practical, clear, and non-repetitive answers. "
        f"{knowledge_rule}"
    )

    user_prompt = (
        f"Current user question:\n{query}\n\n"
        f"Recent user questions (conversation memory):\n{recent_block}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        "Answer format:\n"
        "1) Start with a direct answer in 1-2 sentences.\n"
        "2) Add 2-4 concise bullet points with useful details if needed.\n"
        "3) If uncertain, state uncertainty briefly and suggest a better follow-up question."
    )

    return system_prompt, user_prompt


def _format_llm_failure_message(last_status: Optional[int], last_error: Optional[str]) -> str:
    """Return user-safe fallback message for LLM failures."""
    err = (last_error or "").upper()
    if last_status == 403 or "PERMISSION_DENIED" in err:
        return (
            "I cannot access Gemini for this project right now due to permission limits. "
            "I can still help using the local knowledge base."
        )

    if last_status in (401, 429):
        return "The language model is temporarily unavailable due to auth/rate limits. Please try again shortly."

    return "I could not generate a model response right now. Please try again in a moment."


def generate_rag_answer(
    query: str,
    retrieved_chunks: List[Dict[str, float | str]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    model: str = "llama-3.1-8b-instant",
    timeout_sec: int = 12,
    max_attempts: int = 5,
) -> str:
    """Generate hybrid RAG answer with robust fallback.

    This function never fails silently: it returns either model output
    or a meaningful fallback message.
    """
    if not query.strip():
        return "Please enter a valid question so I can help."

    api_key = sanitize_api_key(resolve_groq_api_key())
    if not api_key:
        return "I could not access the language model right now. Please verify groq-api-key and try again."

    ok, msg = validate_groq_api_key(api_key)
    if not ok:
        return f"I could not use the language model due to API key validation issue: {msg}."

    recent_user_queries = _extract_recent_user_queries(conversation_history, max_queries=5)
    system_prompt, user_prompt = _build_hybrid_prompts(
        query=query,
        retrieved_chunks=retrieved_chunks,
        recent_user_queries=recent_user_queries,
        allow_general_knowledge=True,
    )

    attempts = _build_model_attempts(api_key, model)
    if max_attempts > 0:
        attempts = attempts[:max_attempts]

    last_error = None
    last_status = None

    try:
        for api_version, candidate_model in attempts:
            content, status, err = _groq_generate_content(
                api_key=api_key,
                model=candidate_model,
                api_version=api_version,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_sec=timeout_sec,
            )
            if status == 200 and content:
                return content
            last_status = status
            last_error = err
    except Exception:
        return "I ran into a temporary model error while generating your answer. Please try again."

    return _format_llm_failure_message(last_status, last_error)


def generate_general_answer(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    model: str = "llama-3.1-8b-instant",
    timeout_sec: int = 12,
    max_attempts: int = 5,
) -> str:
    """Generate answer without retrieval context as fallback path."""
    if not query.strip():
        return "Please enter a valid question so I can help."

    api_key = sanitize_api_key(resolve_groq_api_key())
    if not api_key:
        return "I could not access the language model right now. Please verify groq-api-key and try again."

    ok, msg = validate_groq_api_key(api_key)
    if not ok:
        return f"I could not use the language model due to API key validation issue: {msg}."

    recent_user_queries = _extract_recent_user_queries(conversation_history, max_queries=5)
    system_prompt, user_prompt = _build_hybrid_prompts(
        query=query,
        retrieved_chunks=[],
        recent_user_queries=recent_user_queries,
        allow_general_knowledge=True,
    )

    attempts = _build_model_attempts(api_key, model)
    if max_attempts > 0:
        attempts = attempts[:max_attempts]

    last_error = None
    last_status = None
    try:
        for api_version, candidate_model in attempts:
            content, status, err = _groq_generate_content(
                api_key=api_key,
                model=candidate_model,
                api_version=api_version,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_sec=timeout_sec,
            )
            if status == 200 and content:
                return content
            last_status = status
            last_error = err
    except Exception:
        return "I ran into a temporary model error while generating your answer. Please try again."

    return _format_llm_failure_message(last_status, last_error)


def generate_rag_answer_with_debug(
    query: str,
    retrieved_chunks: List[Dict[str, float | str]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    model: str = "llama-3.1-8b-instant",
    timeout_sec: int = 12,
    max_attempts: int = 5,
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

    # Retrieval may be empty in hybrid flow. We still try model generation.

    api_key, source = resolve_groq_api_key_with_source()
    api_key = sanitize_api_key(api_key)
    debug["api_key_source"] = source

    if not api_key:
        debug["mode"] = "missing-key"
        debug["alt_keys_detected"] = _detect_alt_llm_keys()
        if debug["alt_keys_detected"]:
            debug["error"] = "groq-api-key missing; found other LLM keys"
        else:
            debug["error"] = "groq-api-key missing"
        return {"answer": None, "debug": debug}

    ok, msg = validate_groq_api_key(api_key)
    if not ok:
        debug["mode"] = "invalid-key-format"
        debug["error"] = msg
        return {"answer": None, "debug": debug}

    recent_user_queries = _extract_recent_user_queries(conversation_history, max_queries=5)
    system_prompt, user_prompt = _build_hybrid_prompts(
        query=query,
        retrieved_chunks=retrieved_chunks,
        recent_user_queries=recent_user_queries,
        allow_general_knowledge=True,
    )

    try:
        last_error = None
        last_status = None

        attempts = _build_model_attempts(api_key, model)
        if max_attempts > 0:
            attempts = attempts[:max_attempts]

        for api_version, candidate_model in attempts:
            debug["models_tried"].append(candidate_model)
            debug["api_versions_tried"].append(api_version)
            content, status, err = _groq_generate_content(
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


def get_final_answer(
    query: str,
    retriever: LocalFAQRetriever,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    model: str = "llama-3.1-8b-instant",
    top_k: int = 5,
    min_score: float = 0.18,
    timeout_sec: int = 12,
    max_attempts: int = 5,
) -> Dict[str, object]:
    """Hybrid orchestrator: retrieve context, try RAG, then fallback to pure LLM.

    Returns dict with keys: answer, mode, debug, retrieved_chunks.
    """
    if not query.strip():
        return {
            "answer": "Please enter a valid question so I can help.",
            "mode": "empty-query",
            "debug": {"error": "empty-query"},
            "retrieved_chunks": [],
        }

    retrieved = retrieve_context(
        query=query,
        retriever=retriever,
        top_k=top_k,
        min_score=min_score,
    )

    if retrieved:
        rag_result = generate_rag_answer_with_debug(
            query=query,
            retrieved_chunks=retrieved,
            conversation_history=conversation_history,
            model=model,
            timeout_sec=timeout_sec,
            max_attempts=max_attempts,
        )
        answer = rag_result.get("answer") if isinstance(rag_result, dict) else None
        debug = rag_result.get("debug", {}) if isinstance(rag_result, dict) else {}

        if isinstance(answer, str) and answer.strip():
            debug["fallback_used"] = False
            debug["retrieval_used"] = True
            print(f"[chat-debug] mode=rag model={debug.get('model_used')} status={debug.get('api_http_status')} fallback=False")
            return {
                "answer": answer.strip(),
                "mode": "rag",
                "debug": debug,
                "retrieved_chunks": retrieved,
            }

        # RAG path failed -> pure LLM fallback
        general_answer = generate_general_answer(
            query=query,
            conversation_history=conversation_history,
            model=model,
            timeout_sec=timeout_sec,
            max_attempts=max_attempts,
        )

        # If general LLM also fails, return strongest retrieved context instead of raw failure text.
        if (
            isinstance(general_answer, str)
            and (
                "cannot access gemini" in general_answer.lower()
                or "language model is temporarily unavailable" in general_answer.lower()
                or "could not generate a model response" in general_answer.lower()
            )
            and retrieved
        ):
            best_context = str(retrieved[0].get("text", "")).strip()
            if best_context:
                general_answer = best_context[:420]

        debug["fallback_used"] = True
        debug["retrieval_used"] = True
        print(f"[chat-debug] mode=rag-fallback-general model={debug.get('model_used')} status={debug.get('api_http_status')} fallback=True")
        return {
            "answer": general_answer,
            "mode": "rag-fallback-general",
            "debug": debug,
            "retrieved_chunks": retrieved,
        }

    # No relevant retrieval -> pure LLM path
    general_answer = generate_general_answer(
        query=query,
        conversation_history=conversation_history,
        model=model,
        timeout_sec=timeout_sec,
        max_attempts=max_attempts,
    )
    debug = {
        "fallback_used": True,
        "retrieval_used": False,
        "retrieved_count": 0,
    }
    print("[chat-debug] mode=general-no-retrieval fallback=True")
    return {
        "answer": general_answer,
        "mode": "general",
        "debug": debug,
        "retrieved_chunks": [],
    }
