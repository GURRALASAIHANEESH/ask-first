from __future__ import annotations
import os
from typing import Optional

from engine.models import DetectedPattern, EngineOutput


_LLM_ENABLED = os.environ.get("ASKFIRST_LLM_ENABLED", "false").lower() == "true"
_LLM_PROVIDER = os.environ.get("ASKFIRST_LLM_PROVIDER", "none")
_LLM_API_KEY = os.environ.get("ASKFIRST_LLM_API_KEY", "")


def is_enabled() -> bool:
    return _LLM_ENABLED and _LLM_PROVIDER != "none" and len(_LLM_API_KEY) > 0


def summarize_pattern(pattern: DetectedPattern) -> Optional[str]:
    if not is_enabled():
        return None

    prompt = _build_prompt(pattern)

    if _LLM_PROVIDER == "openai":
        return _call_openai(prompt)

    return None


def summarize_output(output: EngineOutput) -> Optional[dict[str, str]]:
    if not is_enabled():
        return None

    summaries: dict[str, str] = {}
    for p in output.detected_patterns:
        result = summarize_pattern(p)
        if result:
            summaries[p.pattern_id] = result
    return summaries if summaries else None


def _build_prompt(pattern: DetectedPattern) -> str:
    sessions_str = ", ".join(pattern.evidence_sessions)
    signals_str = "; ".join(
        f"{s.signal_type}: {s.extracted_text} ({s.session_id})"
        for s in pattern.supporting_signals[:10]
    )
    return (
        f"You are a health pattern analyst. Summarize this detected pattern in 2-3 "
        f"clear sentences for a non-medical user. Be precise and do not add information "
        f"not present in the evidence.\n\n"
        f"Pattern: {pattern.title}\n"
        f"User: {pattern.user_id}\n"
        f"Confidence: {pattern.confidence.value}\n"
        f"Sessions: {sessions_str}\n"
        f"Temporal reasoning: {pattern.temporal_reasoning}\n"
        f"Signals: {signals_str}\n"
        f"Alternatives considered: {'; '.join(pattern.alternative_explanations)}\n\n"
        f"Summary:"
    )


def _call_openai(prompt: str) -> Optional[str]:
    try:
        import openai
        client = openai.OpenAI(api_key=_LLM_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None