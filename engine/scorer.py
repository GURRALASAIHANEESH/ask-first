from __future__ import annotations
from collections import defaultdict
from datetime import datetime

from engine.models import (
    Confidence,
    DetectedPattern,
    ReasoningStep,
    UserTimeline,
)


_CONFIDENCE_RANK = {
    Confidence.VERY_LOW: 0,
    Confidence.LOW: 1,
    Confidence.MEDIUM: 2,
    Confidence.HIGH: 3,
    Confidence.VERY_HIGH: 4,
}

_RANK_TO_CONFIDENCE = {v: k for k, v in _CONFIDENCE_RANK.items()}


def _bump(current: Confidence, steps: int = 1) -> Confidence:
    rank = _CONFIDENCE_RANK[current]
    new_rank = min(rank + steps, 4)
    return _RANK_TO_CONFIDENCE[new_rank]


def _demote(current: Confidence, steps: int = 1) -> Confidence:
    rank = _CONFIDENCE_RANK[current]
    new_rank = max(rank - steps, 0)
    return _RANK_TO_CONFIDENCE[new_rank]


def _count_distinct_months(pattern: DetectedPattern, timelines: dict[str, UserTimeline]) -> int:
    tl = timelines.get(pattern.user_id)
    if tl is None:
        return 0
    months: set[str] = set()
    evidence_set = set(pattern.evidence_sessions)
    for s in tl.sessions:
        if s.conversation.session_id in evidence_set:
            months.add(s.conversation.timestamp.strftime("%Y-%m"))
    return len(months)


def _has_intervention_confirmation(pattern: DetectedPattern) -> bool:
    lower = pattern.temporal_reasoning.lower()
    return "intervention" in lower or "improvement" in lower or "resolved" in lower


def _has_worsening_confirmation(pattern: DetectedPattern) -> bool:
    lower = pattern.temporal_reasoning.lower()
    return "worsening" in lower or "worse" in lower or "breakout" in lower


def _evidence_count(pattern: DetectedPattern) -> int:
    return len(pattern.evidence_sessions)


def recalibrate(
    patterns: list[DetectedPattern],
    timelines: list[UserTimeline],
    reasoning: list[ReasoningStep],
) -> tuple[list[DetectedPattern], list[ReasoningStep]]:
    step_counter = max((r.step for r in reasoning), default=0)
    tl_map = {tl.user.user_id: tl for tl in timelines}

    def log(action: str, detail: str) -> None:
        nonlocal step_counter
        step_counter += 1
        reasoning.append(ReasoningStep(step=step_counter, action=action, detail=detail))

    log("scoring_start", f"Recalibrating confidence for {len(patterns)} detected patterns")

    for p in patterns:
        original = p.confidence
        months = _count_distinct_months(p, tl_map)
        evidence = _evidence_count(p)
        has_intervention = _has_intervention_confirmation(p)
        has_worsening = _has_worsening_confirmation(p)

        adjustments: list[str] = []

        if months >= 3 and evidence >= 3:
            if _CONFIDENCE_RANK[p.confidence] < _CONFIDENCE_RANK[Confidence.HIGH]:
                p.confidence = _bump(p.confidence)
                adjustments.append(f"bumped: evidence spans {months} months with {evidence} sessions")

        if months >= 2 and evidence >= 4:
            if _CONFIDENCE_RANK[p.confidence] < _CONFIDENCE_RANK[Confidence.VERY_HIGH]:
                p.confidence = _bump(p.confidence)
                adjustments.append(f"bumped: {evidence} sessions across {months} months")

        if has_intervention and has_worsening:
            if _CONFIDENCE_RANK[p.confidence] < _CONFIDENCE_RANK[Confidence.VERY_HIGH]:
                p.confidence = _bump(p.confidence)
                adjustments.append("bumped: both intervention improvement and worsening confirmation present")
        elif has_intervention:
            if _CONFIDENCE_RANK[p.confidence] < _CONFIDENCE_RANK[Confidence.HIGH]:
                p.confidence = _bump(p.confidence)
                adjustments.append("bumped: intervention followed by improvement")

        if evidence == 2 and months == 1:
            if _CONFIDENCE_RANK[p.confidence] > _CONFIDENCE_RANK[Confidence.MEDIUM]:
                p.confidence = _demote(p.confidence)
                adjustments.append("demoted: only 2 sessions in a single month")

        if evidence == 1:
            p.confidence = Confidence.LOW
            adjustments.append("set to low: only 1 evidence session")

        if len(p.alternative_explanations) >= 3:
            if _CONFIDENCE_RANK[p.confidence] > _CONFIDENCE_RANK[Confidence.HIGH]:
                pass
            elif _CONFIDENCE_RANK[p.confidence] == _CONFIDENCE_RANK[Confidence.HIGH]:
                if not has_intervention:
                    p.confidence = _demote(p.confidence)
                    adjustments.append("demoted: multiple alternative explanations without intervention confirmation")

        _validate_ceiling(p, months, evidence, has_intervention)

        if p.confidence != original:
            log(
                "confidence_adjusted",
                f"{p.pattern_id} ({p.user_id}): {original.value} → {p.confidence.value}. "
                f"Reasons: {'; '.join(adjustments) if adjustments else 'ceiling validation'}",
            )
        else:
            log(
                "confidence_unchanged",
                f"{p.pattern_id} ({p.user_id}): remains {p.confidence.value}",
            )

        p.justification = _rebuild_justification(p, months, evidence, has_intervention, has_worsening, tl_map)

    log("scoring_complete", f"Recalibration complete for {len(patterns)} patterns")
    return patterns, reasoning


def _validate_ceiling(
    p: DetectedPattern,
    months: int,
    evidence: int,
    has_intervention: bool,
) -> None:
    if p.confidence == Confidence.VERY_HIGH:
        if evidence < 3 or months < 2:
            p.confidence = Confidence.HIGH
    if p.confidence == Confidence.HIGH:
        if evidence < 2:
            p.confidence = Confidence.MEDIUM


def _rebuild_justification(
    p: DetectedPattern,
    months: int,
    evidence: int,
    has_intervention: bool,
    has_worsening: bool,
    tl_map: dict[str, UserTimeline],
) -> str:
    parts: list[str] = []

    parts.append(f"{evidence} evidence sessions across {months} month(s)")

    tl = tl_map.get(p.user_id)
    if tl and evidence >= 2:
        timestamps: list[datetime] = []
        evidence_set = set(p.evidence_sessions)
        for s in tl.sessions:
            if s.conversation.session_id in evidence_set:
                timestamps.append(s.conversation.timestamp)
        if len(timestamps) >= 2:
            timestamps.sort()
            total_span = (timestamps[-1] - timestamps[0]).days
            parts.append(f"spanning {total_span} days")

    if has_intervention and has_worsening:
        parts.append("confirmed by both intervention improvement and recurrence on re-exposure")
    elif has_intervention:
        parts.append("confirmed by intervention leading to improvement")
    elif has_worsening:
        parts.append("supported by symptom worsening on trigger re-exposure")

    trigger_signals = {
        s.extracted_text for s in p.supporting_signals if s.signal_type == "trigger"
    }
    symptom_signals = {
        s.extracted_text for s in p.supporting_signals if s.signal_type == "symptom"
    }
    if trigger_signals and symptom_signals:
        parts.append(
            f"linking {', '.join(sorted(symptom_signals))} to "
            f"{', '.join(sorted(trigger_signals))}"
        )

    return ". ".join(parts)