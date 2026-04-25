from __future__ import annotations
import json
import sys
from typing import TextIO

from engine.models import (
    ChunkingStrategy,
    DetectedPattern,
    EngineOutput,
    ReasoningStep,
    UserTimeline,
    UserWindowInfo,
)
from engine.timeline import compute_session_windows


def build_chunking_strategy(timelines: list[UserTimeline]) -> ChunkingStrategy:
    per_user: dict[str, list[UserWindowInfo]] = {}

    for tl in timelines:
        windows = compute_session_windows(tl)
        user_windows: list[UserWindowInfo] = []
        for w in windows:
            user_windows.append(
                UserWindowInfo(window=w.label, sessions=w.session_ids)
            )
        per_user[tl.user.user_id] = user_windows

    return ChunkingStrategy(
        method="temporal_windowing_per_user",
        description=(
            "Each user's conversations are sorted chronologically, then grouped into "
            "half-month windows (early: days 1-15, late: days 16-31) within each calendar month. "
            "This preserves session ordering and allows the detector to reason across windows "
            "while keeping related sessions grouped. All prior sessions are retained in context "
            "when analyzing a new window so that cross-session patterns spanning months are not "
            "missed. The detector operates on the full user timeline but uses windows to structure "
            "its temporal reasoning and to report which time periods contributed to each finding."
        ),
        per_user_windows=per_user,
    )


def build_failure_notes(
    patterns: list[DetectedPattern],
    timelines: list[UserTimeline],
) -> list[str]:
    notes: list[str] = []

    notes.append(
        "Signal extraction relies on phrase matching against a fixed taxonomy. "
        "Symptoms or triggers expressed in language not covered by the taxonomy will be missed. "
        "Novel phrasing or metaphorical descriptions of health issues are not captured."
    )

    notes.append(
        "Negation detection is limited to a small set of known phrases per trigger. "
        "Complex negations, sarcasm, or conditional statements may cause false positives "
        "where a trigger is detected despite being explicitly denied by the user."
    )

    notes.append(
        "The system cannot distinguish correlation from causation. Co-occurring signals "
        "across sessions are reported as correlated patterns, not proven causal links. "
        "The temporal ordering provides directional evidence but not proof."
    )

    notes.append(
        "Confidence scores are calibrated using session count, month span, and intervention "
        "confirmation. These heuristics may over-weight patterns with more sessions regardless "
        "of the quality of evidence in each session."
    )

    notes.append(
        "The compounding cascade strategy assumes that symptoms appearing after a trigger "
        "and not present before it are downstream effects. This assumption fails if the "
        "symptom existed before the dataset's time range or was simply not reported earlier."
    )

    notes.append(
        "Variable isolation requires at least 3 sessions with the same symptom and at least "
        "2 co-occurring triggers to compare. Patterns with fewer data points will not be "
        "analyzed for isolated drivers, which may cause under-detection."
    )

    total_sessions = sum(len(tl.sessions) for tl in timelines)
    covered_sessions: set[str] = set()
    for p in patterns:
        covered_sessions.update(p.evidence_sessions)
    uncovered = total_sessions - len(covered_sessions)
    if uncovered > 0:
        notes.append(
            f"{uncovered} of {total_sessions} sessions are not referenced by any detected "
            f"pattern. These sessions may contain signals that the detector did not link to "
            f"a cross-session pattern, or they may be genuinely isolated events."
        )

    return notes


def build_output(
    patterns: list[DetectedPattern],
    reasoning: list[ReasoningStep],
    timelines: list[UserTimeline],
) -> EngineOutput:
    chunking = build_chunking_strategy(timelines)
    failures = build_failure_notes(patterns, timelines)
    return EngineOutput(
        detected_patterns=patterns,
        reasoning_trace=reasoning,
        chunking_strategy=chunking,
        failure_notes=failures,
    )


def stream_progress(message: str, stream: TextIO = sys.stderr) -> None:
    stream.write(f"[PROGRESS] {message}\n")
    stream.flush()


def stream_pattern_found(pattern: DetectedPattern, stream: TextIO = sys.stderr) -> None:
    stream.write(
        f"[PATTERN] {pattern.pattern_id} | {pattern.user_id} | "
        f"{pattern.confidence.value.upper()} | {pattern.title}\n"
    )
    stream.flush()


def emit_json(output: EngineOutput, stream: TextIO = sys.stdout, indent: int = 2) -> None:
    json_str = json.dumps(output.to_dict(), indent=indent, ensure_ascii=False)
    stream.write(json_str)
    stream.write("\n")
    stream.flush()


def write_json_file(output: EngineOutput, path: str, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dumps(output.to_dict(), indent=indent, ensure_ascii=False)
        f.write(json.dumps(output.to_dict(), indent=indent, ensure_ascii=False))
        f.write("\n")