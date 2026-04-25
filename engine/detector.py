from __future__ import annotations
from collections import defaultdict
from datetime import datetime

from engine.models import (
    AnnotatedSession,
    Confidence,
    DetectedPattern,
    ReasoningStep,
    SignalType,
    SupportingSignal,
    UserTimeline,
)
from engine.timeline import (
    compute_gap_days,
    find_co_occurring_signals,
    get_sessions_with_signal_value,
)


_VALUE_LABELS: dict[str, str] = {
    "stomach_pain": "stomach pain",
    "headache": "headache",
    "back_pain": "lower back pain",
    "fatigue": "fatigue",
    "dizziness": "dizziness",
    "acne": "acne",
    "hair_fall": "hair fall",
    "brain_fog": "brain fog",
    "menstrual_cramps": "menstrual cramps",
    "anxiety": "anxiety",
    "mood_low": "low mood",
    "post_lunch_crash": "post-lunch energy crash",
    "late_eating": "late night eating",
    "dehydration": "low water intake",
    "work_stress": "work deadline stress",
    "dairy_intake": "dairy consumption",
    "caffeine": "caffeine intake",
    "calorie_restriction": "severe calorie restriction",
    "high_carb_meal": "high carbohydrate meal",
    "screen_use_late": "late night screen use",
    "sedentary": "sedentary behavior",
    "sleep_deprivation": "sleep deprivation",
    "intermittent_fasting": "intermittent fasting",
    "dairy_reduction": "dairy reduction",
    "dairy_reintroduction": "dairy reintroduction",
    "protein_added": "adding protein to lunch",
    "water_increase": "increasing water intake",
    "calorie_increase": "increasing calorie intake",
    "screen_limit": "screen time limit",
    "posture_break": "posture breaks",
    "symptom_improved": "symptom improvement",
    "condition_resolved": "condition resolved",
    "symptom_worse": "symptom worsening",
    "symptom_recurring": "symptom recurring",
    "eating_out": "eating out frequently",
    "poor_diet": "poor diet",
    "work_long_hours": "long work hours",
    "screen_all_day": "all-day screen exposure",
    "irregular_sleep": "irregular sleep schedule",
}

_NEGATION_INDICATORS: dict[str, list[str]] = {
    "work_stress": [
        "stress is actually low",
        "stress is low",
        "work was actually fine",
        "work is a bit calmer",
        "no big deadlines",
        "work is fine",
        "low stress",
    ],
    "sleep_deprivation": [
        "sleep is okay",
        "sleep is fine",
        "sleep is good",
        "sleeping well",
    ],
}


def _label(value: str) -> str:
    return _VALUE_LABELS.get(value, value.replace("_", " "))


def _is_negated(session: AnnotatedSession, signal_value: str) -> bool:
    if signal_value not in _NEGATION_INDICATORS:
        return False
    text = session.conversation.full_text().lower()
    tags_lower = [t.lower() for t in session.conversation.tags]
    for phrase in _NEGATION_INDICATORS[signal_value]:
        if phrase.lower() in text:
            return True
    if signal_value == "work_stress":
        if "stress low" in tags_lower:
            return True
    return False


def _build_supporting(session: AnnotatedSession, values: set[str]) -> list[SupportingSignal]:
    result = []
    for sig in session.signals:
        if sig.value in values:
            result.append(
                SupportingSignal(
                    session_id=sig.session_id,
                    signal_type=sig.signal_type.value,
                    extracted_text=sig.raw_text,
                    timestamp=sig.timestamp.isoformat(),
                )
            )
    return result


def _session_dates_str(sessions: list[AnnotatedSession]) -> str:
    dates = [s.conversation.timestamp.strftime("%b %d") for s in sessions]
    return ", ".join(dates)


def _sessions_to_ids(sessions: list[AnnotatedSession]) -> list[str]:
    return [s.conversation.session_id for s in sessions]


def _signal_values_in_session(session: AnnotatedSession, signal_type: SignalType) -> set[str]:
    return {
        sig.value for sig in session.signals
        if sig.signal_type == signal_type
    }


def _get_claimed_pairs(patterns: list[DetectedPattern], user_id: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for p in patterns:
        if p.user_id != user_id:
            continue
        symptoms = set()
        triggers = set()
        for sig in p.supporting_signals:
            if sig.signal_type == "symptom":
                symptoms.add(sig.extracted_text)
            elif sig.signal_type == "trigger":
                triggers.add(sig.extracted_text)
        for sy in symptoms:
            for tr in triggers:
                pairs.add((sy, tr))
    return pairs


class PatternDetector:
    def __init__(self):
        self._patterns: list[DetectedPattern] = []
        self._reasoning: list[ReasoningStep] = []
        self._step_counter = 0
        self._pattern_counter = 0
        self._cooccurrence_keys: set[tuple[str, str, str]] = set()

    def _log(self, action: str, detail: str) -> None:
        self._step_counter += 1
        self._reasoning.append(
            ReasoningStep(step=self._step_counter, action=action, detail=detail)
        )

    def _next_id(self) -> str:
        self._pattern_counter += 1
        return f"P-{self._pattern_counter:03d}"

    def detect(self, timelines: list[UserTimeline]) -> tuple[list[DetectedPattern], list[ReasoningStep]]:
        for tl in timelines:
            uid = tl.user.user_id
            name = tl.user.name
            self._log(
                "begin_user_analysis",
                f"Analyzing {name} ({uid}): {len(tl.sessions)} sessions from "
                f"{tl.sessions[0].conversation.timestamp.strftime('%Y-%m-%d')} to "
                f"{tl.sessions[-1].conversation.timestamp.strftime('%Y-%m-%d')}",
            )
            self._strategy_recurring_cooccurrence(tl)
            self._strategy_delayed_sequence(tl)
            self._strategy_intervention_outcome(tl)
            self._strategy_compounding_cascade(tl)
            self._strategy_variable_isolation(tl)

        self._deduplicate()
        return self._patterns, self._reasoning

    def _strategy_recurring_cooccurrence(self, tl: UserTimeline) -> None:
        self._log("strategy_start", f"Running recurring co-occurrence analysis for {tl.user.user_id}")

        symptom_sessions: dict[str, list[AnnotatedSession]] = defaultdict(list)
        trigger_sessions: dict[str, list[AnnotatedSession]] = defaultdict(list)

        for s in tl.sessions:
            for sig in s.signals:
                if sig.signal_type == SignalType.SYMPTOM:
                    symptom_sessions[sig.value].append(s)
                elif sig.signal_type == SignalType.TRIGGER:
                    trigger_sessions[sig.value].append(s)

        for symptom, sym_sessions in symptom_sessions.items():
            if len(sym_sessions) < 2:
                continue

            sym_ids = {s.conversation.session_id for s in sym_sessions}
            correlated: list[tuple[str, list[AnnotatedSession], float]] = []

            for trigger, trg_sessions in trigger_sessions.items():
                valid_trg = [
                    s for s in trg_sessions
                    if not _is_negated(s, trigger)
                ]
                valid_trg_ids = {s.conversation.session_id for s in valid_trg}
                overlap_ids = sym_ids & valid_trg_ids
                if len(overlap_ids) < 2:
                    continue
                ratio = len(overlap_ids) / len(sym_ids)
                overlap_sessions = [
                    s for s in sym_sessions if s.conversation.session_id in overlap_ids
                ]
                correlated.append((trigger, overlap_sessions, ratio))

            if not correlated:
                self._log(
                    "no_cooccurrence",
                    f"Symptom '{_label(symptom)}' in {len(sym_sessions)} sessions "
                    f"but no consistent trigger found for {tl.user.user_id}",
                )
                continue

            correlated.sort(key=lambda x: (-x[2], -len(x[1])))
            primary_trigger, primary_sessions, primary_ratio = correlated[0]
            secondary = [
                (t, sess, r) for t, sess, r in correlated[1:]
                if r >= 0.5 and len(sess) >= 2
            ]

            all_evidence_sessions = list(primary_sessions)
            all_evidence_ids = {s.conversation.session_id for s in all_evidence_sessions}
            for _, sec_sessions, _ in secondary:
                for s in sec_sessions:
                    if s.conversation.session_id not in all_evidence_ids:
                        all_evidence_sessions.append(s)
                        all_evidence_ids.add(s.conversation.session_id)
            all_evidence_sessions.sort(key=lambda s: s.conversation.timestamp)

            title = f"Recurring {_label(symptom)} correlated with {_label(primary_trigger)}"
            if secondary:
                sec_labels = [_label(t) for t, _, _ in secondary[:2]]
                title += f" during {' and '.join(sec_labels)}"

            dates_str = _session_dates_str(all_evidence_sessions)
            temporal = (
                f"{_label(symptom).capitalize()} appeared in {len(sym_sessions)} sessions. "
                f"{_label(primary_trigger).capitalize()} co-occurred in "
                f"{len(primary_sessions)} of those ({primary_ratio:.0%}). "
                f"Dates: {dates_str}."
            )
            if secondary:
                for t, sess, r in secondary:
                    temporal += (
                        f" {_label(t).capitalize()} also present in "
                        f"{len(sess)} sessions ({r:.0%})."
                    )

            months = {s.conversation.timestamp.strftime("%Y-%m") for s in primary_sessions}
            if len(primary_sessions) >= 3 and primary_ratio >= 0.75 and len(months) >= 2:
                confidence = Confidence.VERY_HIGH
            elif len(primary_sessions) >= 3 and primary_ratio >= 0.6:
                confidence = Confidence.HIGH
            elif len(primary_sessions) >= 2 and primary_ratio >= 0.5:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            gaps = []
            sorted_primary = sorted(primary_sessions, key=lambda s: s.conversation.timestamp)
            for i in range(1, len(sorted_primary)):
                gap = compute_gap_days(
                    sorted_primary[i - 1].conversation.timestamp,
                    sorted_primary[i].conversation.timestamp,
                )
                gaps.append(gap)

            justification = (
                f"{_label(primary_trigger).capitalize()} appears in "
                f"{len(primary_sessions)}/{len(sym_sessions)} {_label(symptom)} episodes "
                f"across {len(months)} month(s)"
            )
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                justification += f" with average {avg_gap:.0f}-day gap between episodes"

            all_values = {symptom, primary_trigger}
            for t, _, _ in secondary:
                all_values.add(t)
            supporting = []
            for s in all_evidence_sessions:
                supporting.extend(_build_supporting(s, all_values))

            alternatives = [
                f"Coincidental timing of {_label(symptom)} and {_label(primary_trigger)}",
                f"Unknown confounding variable driving both {_label(symptom)} and {_label(primary_trigger)}",
            ]
            if secondary:
                alternatives.append(
                    f"One of the secondary factors ({', '.join(_label(t) for t, _, _ in secondary)}) "
                    f"may be the actual driver"
                )

            self._cooccurrence_keys.add((tl.user.user_id, symptom, primary_trigger))
            for t, _, _ in secondary:
                self._cooccurrence_keys.add((tl.user.user_id, symptom, t))

            self._patterns.append(
                DetectedPattern(
                    pattern_id=self._next_id(),
                    user_id=tl.user.user_id,
                    title=title,
                    evidence_sessions=_sessions_to_ids(all_evidence_sessions),
                    temporal_reasoning=temporal,
                    confidence=confidence,
                    justification=justification,
                    supporting_signals=supporting,
                    alternative_explanations=alternatives,
                )
            )
            self._log(
                "cooccurrence_found",
                f"{tl.user.user_id}: {title} — {confidence.value} confidence",
            )

    def _strategy_delayed_sequence(self, tl: UserTimeline) -> None:
        self._log("strategy_start", f"Running delayed sequence analysis for {tl.user.user_id}")

        trigger_first: dict[str, AnnotatedSession] = {}
        trigger_count: dict[str, int] = defaultdict(int)
        for s in tl.sessions:
            for sig in s.signals:
                if sig.signal_type == SignalType.TRIGGER:
                    trigger_count[sig.value] += 1
                    if sig.value not in trigger_first:
                        trigger_first[sig.value] = s

        symptom_first: dict[str, AnnotatedSession] = {}
        for s in tl.sessions:
            for sig in s.signals:
                if sig.signal_type == SignalType.SYMPTOM and sig.value not in symptom_first:
                    symptom_first[sig.value] = s

        for trigger_val, trigger_sess in trigger_first.items():
            if trigger_count[trigger_val] < 2:
                self._log(
                    "delayed_skip",
                    f"{tl.user.user_id}: Skipping trigger '{_label(trigger_val)}' — "
                    f"appears in only {trigger_count[trigger_val]} session(s)",
                )
                continue

            for symptom_val, symptom_sess in symptom_first.items():
                t_time = trigger_sess.conversation.timestamp
                s_time = symptom_sess.conversation.timestamp
                if s_time <= t_time:
                    continue

                gap = compute_gap_days(t_time, s_time)
                if gap < 14 or gap > 90:
                    continue

                if (tl.user.user_id, symptom_val, trigger_val) in self._cooccurrence_keys:
                    continue

                cooc_sessions = find_co_occurring_signals(tl, trigger_val, symptom_val)
                if len(cooc_sessions) == 0:
                    self._log(
                        "delayed_skip",
                        f"{tl.user.user_id}: No co-occurrence of '{_label(trigger_val)}' and "
                        f"'{_label(symptom_val)}' in any session — skipping",
                    )
                    continue
                if len(cooc_sessions) >= 3:
                    continue

                later_symptom_sessions = [
                    s for s in tl.sessions
                    if any(sig.value == symptom_val for sig in s.signals)
                    and s.conversation.timestamp > t_time
                ]

                evidence = [trigger_sess] + later_symptom_sessions
                evidence_deduped = []
                seen_ids: set[str] = set()
                for s in sorted(evidence, key=lambda s: s.conversation.timestamp):
                    if s.conversation.session_id not in seen_ids:
                        seen_ids.add(s.conversation.session_id)
                        evidence_deduped.append(s)

                title = (
                    f"{_label(trigger_val).capitalize()} triggering delayed "
                    f"{_label(symptom_val)} after {gap:.0f} days"
                )

                temporal = (
                    f"{_label(trigger_val).capitalize()} first noted on "
                    f"{trigger_sess.conversation.timestamp.strftime('%b %d')}. "
                    f"{_label(symptom_val).capitalize()} first appeared on "
                    f"{symptom_sess.conversation.timestamp.strftime('%b %d')}, "
                    f"a gap of {gap:.0f} days."
                )
                if len(later_symptom_sessions) > 1:
                    temporal += (
                        f" {_label(symptom_val).capitalize()} reported in "
                        f"{len(later_symptom_sessions)} subsequent session(s)."
                    )

                confidence = Confidence.MEDIUM
                if gap >= 28 and len(later_symptom_sessions) >= 2:
                    confidence = Confidence.HIGH

                justification = (
                    f"{_label(trigger_val).capitalize()} began "
                    f"{trigger_sess.conversation.timestamp.strftime('%b %d')} and "
                    f"{_label(symptom_val)} emerged {gap:.0f} days later, "
                    f"with co-occurrence in {len(cooc_sessions)} session(s) validating the link"
                )

                values = {trigger_val, symptom_val}
                supporting = []
                for s in evidence_deduped:
                    supporting.extend(_build_supporting(s, values))

                alternatives = [
                    f"{_label(symptom_val).capitalize()} may be caused by an unrelated condition",
                    f"Temporal proximity does not establish causation",
                    f"Other lifestyle changes during the same period could explain {_label(symptom_val)}",
                ]

                self._patterns.append(
                    DetectedPattern(
                        pattern_id=self._next_id(),
                        user_id=tl.user.user_id,
                        title=title,
                        evidence_sessions=[s.conversation.session_id for s in evidence_deduped],
                        temporal_reasoning=temporal,
                        confidence=confidence,
                        justification=justification,
                        supporting_signals=supporting,
                        alternative_explanations=alternatives,
                    )
                )
                self._log(
                    "delayed_sequence_found",
                    f"{tl.user.user_id}: {title} — {confidence.value} confidence",
                )

    def _strategy_intervention_outcome(self, tl: UserTimeline) -> None:
        self._log("strategy_start", f"Running intervention outcome analysis for {tl.user.user_id}")

        intervention_sessions: dict[str, list[AnnotatedSession]] = defaultdict(list)
        for s in tl.sessions:
            for sig in s.signals:
                if sig.signal_type == SignalType.INTERVENTION:
                    intervention_sessions[sig.value].append(s)

        for intervention_val, i_sessions in intervention_sessions.items():
            for i_sess in i_sessions:
                i_time = i_sess.conversation.timestamp
                following = [
                    s for s in tl.sessions
                    if s.conversation.timestamp > i_time
                ]
                following.sort(key=lambda s: s.conversation.timestamp)

                for f_sess in following[:4]:
                    has_improvement = any(
                        sig.signal_type == SignalType.IMPROVEMENT
                        for sig in f_sess.signals
                    )
                    has_worsening = any(
                        sig.signal_type == SignalType.WORSENING
                        for sig in f_sess.signals
                    )

                    if not has_improvement and not has_worsening:
                        continue

                    gap = compute_gap_days(i_time, f_sess.conversation.timestamp)

                    enriched = False
                    for p in self._patterns:
                        if p.user_id != tl.user.user_id:
                            continue
                        p_session_set = set(p.evidence_sessions)
                        i_id = i_sess.conversation.session_id
                        f_id = f_sess.conversation.session_id

                        if i_id in p_session_set or f_id in p_session_set:
                            if f_id not in p_session_set:
                                p.evidence_sessions.append(f_id)
                            outcome_type = "improvement" if has_improvement else "worsening"
                            p.temporal_reasoning += (
                                f" Intervention '{_label(intervention_val)}' on "
                                f"{i_sess.conversation.timestamp.strftime('%b %d')} "
                                f"followed by {outcome_type} after {gap:.0f} days."
                            )
                            values = {intervention_val, "symptom_improved", "condition_resolved"}
                            p.supporting_signals.extend(
                                _build_supporting(f_sess, values)
                            )
                            enriched = True
                            self._log(
                                "intervention_enrichment",
                                f"{tl.user.user_id}: Enriched existing pattern with "
                                f"'{_label(intervention_val)}' → "
                                f"{'improvement' if has_improvement else 'worsening'} "
                                f"({gap:.0f} days)",
                            )
                            break

                    if not enriched:
                        outcome = "improvement" if has_improvement else "worsening"
                        title = (
                            f"{_label(intervention_val).capitalize()} followed by "
                            f"symptom {outcome} after {gap:.0f} days"
                        )
                        temporal = (
                            f"Intervention '{_label(intervention_val)}' applied on "
                            f"{i_sess.conversation.timestamp.strftime('%b %d')}. "
                            f"Outcome ({outcome}) observed on "
                            f"{f_sess.conversation.timestamp.strftime('%b %d')}, "
                            f"a gap of {gap:.0f} days."
                        )
                        confidence = Confidence.MEDIUM if has_improvement else Confidence.LOW
                        justification = (
                            f"{outcome.capitalize()} observed {gap:.0f} days after "
                            f"'{_label(intervention_val)}' intervention"
                        )
                        values = {intervention_val, "symptom_improved", "condition_resolved",
                                  "symptom_worse", "symptom_recurring"}
                        supporting = (
                            _build_supporting(i_sess, values) +
                            _build_supporting(f_sess, values)
                        )
                        alternatives = [
                            f"Natural symptom resolution unrelated to intervention",
                            f"Other concurrent changes may explain the {outcome}",
                        ]
                        self._patterns.append(
                            DetectedPattern(
                                pattern_id=self._next_id(),
                                user_id=tl.user.user_id,
                                title=title,
                                evidence_sessions=[
                                    i_sess.conversation.session_id,
                                    f_sess.conversation.session_id,
                                ],
                                temporal_reasoning=temporal,
                                confidence=confidence,
                                justification=justification,
                                supporting_signals=supporting,
                                alternative_explanations=alternatives,
                            )
                        )
                        self._log(
                            "intervention_outcome_found",
                            f"{tl.user.user_id}: {title} — {confidence.value}",
                        )
                    break

    def _strategy_compounding_cascade(self, tl: UserTimeline) -> None:
        self._log("strategy_start", f"Running compounding cascade analysis for {tl.user.user_id}")

        trigger_sessions_map: dict[str, list[AnnotatedSession]] = defaultdict(list)
        for s in tl.sessions:
            for sig in s.signals:
                if sig.signal_type == SignalType.TRIGGER and not _is_negated(s, sig.value):
                    trigger_sessions_map[sig.value].append(s)

        trigger_first_session: dict[str, AnnotatedSession] = {}
        for trigger_val, sessions in trigger_sessions_map.items():
            if len(sessions) < 2:
                continue
            sessions.sort(key=lambda s: s.conversation.timestamp)
            trigger_first_session[trigger_val] = sessions[0]

        for trigger_val, first_sess in trigger_first_session.items():
            t_time = first_sess.conversation.timestamp

            trigger_session_ids = {
                s.conversation.session_id for s in trigger_sessions_map[trigger_val]
            }

            downstream_symptoms: dict[str, AnnotatedSession] = {}
            for s in tl.sessions:
                if s.conversation.timestamp <= t_time:
                    continue
                for sig in s.signals:
                    if sig.signal_type != SignalType.SYMPTOM:
                        continue
                    if sig.value in downstream_symptoms:
                        continue

                    pre_trigger = any(
                        earlier_s for earlier_s in tl.sessions
                        if earlier_s.conversation.timestamp < t_time
                        and any(
                            es.value == sig.value and es.signal_type == SignalType.SYMPTOM
                            for es in earlier_s.signals
                        )
                    )
                    if pre_trigger:
                        continue

                    in_trigger_session = any(
                        s.conversation.session_id == ts_id
                        for ts_id in trigger_session_ids
                    )
                    if not in_trigger_session:
                        cooc = find_co_occurring_signals(tl, trigger_val, sig.value)
                        if len(cooc) == 0:
                            continue

                    downstream_symptoms[sig.value] = s

            if len(downstream_symptoms) < 2:
                self._log(
                    "cascade_skip",
                    f"{tl.user.user_id}: Trigger '{_label(trigger_val)}' has "
                    f"{len(downstream_symptoms)} downstream symptom(s) — need at least 2",
                )
                continue

            sorted_symptoms = sorted(
                downstream_symptoms.items(),
                key=lambda x: x[1].conversation.timestamp,
            )

            evidence = [first_sess] + [s for _, s in sorted_symptoms]
            evidence_deduped = []
            seen_ids: set[str] = set()
            for s in sorted(evidence, key=lambda s: s.conversation.timestamp):
                if s.conversation.session_id not in seen_ids:
                    seen_ids.add(s.conversation.session_id)
                    evidence_deduped.append(s)

            symptom_labels = [_label(sv) for sv, _ in sorted_symptoms]
            title = (
                f"{_label(trigger_val).capitalize()} causing compounding symptoms: "
                f"{', '.join(symptom_labels)}"
            )

            temporal_parts = [
                f"{_label(trigger_val).capitalize()} started on "
                f"{first_sess.conversation.timestamp.strftime('%b %d')}."
            ]
            for sv, ss in sorted_symptoms:
                gap = compute_gap_days(t_time, ss.conversation.timestamp)
                temporal_parts.append(
                    f"{_label(sv).capitalize()} first appeared on "
                    f"{ss.conversation.timestamp.strftime('%b %d')} "
                    f"({gap:.0f} days after trigger)."
                )
            temporal = " ".join(temporal_parts)

            months = {s.conversation.timestamp.strftime("%Y-%m") for s in evidence_deduped}
            if len(sorted_symptoms) >= 3 and len(months) >= 2:
                confidence = Confidence.HIGH
            elif len(sorted_symptoms) >= 2:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            gaps_str = ", ".join(
                f"{_label(sv)} at {compute_gap_days(t_time, ss.conversation.timestamp):.0f} days"
                for sv, ss in sorted_symptoms
            )
            justification = (
                f"{len(sorted_symptoms)} distinct symptoms emerged after "
                f"{_label(trigger_val)} began: {gaps_str}"
            )

            values = {trigger_val} | {sv for sv, _ in sorted_symptoms}
            supporting = []
            for s in evidence_deduped:
                supporting.extend(_build_supporting(s, values))

            alternatives = [
                "Multiple independent conditions appearing coincidentally",
                f"Shared underlying cause other than {_label(trigger_val)}",
                "Symptoms may have pre-existing subclinical presence before trigger",
            ]

            self._patterns.append(
                DetectedPattern(
                    pattern_id=self._next_id(),
                    user_id=tl.user.user_id,
                    title=title,
                    evidence_sessions=[s.conversation.session_id for s in evidence_deduped],
                    temporal_reasoning=temporal,
                    confidence=confidence,
                    justification=justification,
                    supporting_signals=supporting,
                    alternative_explanations=alternatives,
                )
            )
            self._log(
                "cascade_found",
                f"{tl.user.user_id}: {title} — {confidence.value}",
            )

    def _strategy_variable_isolation(self, tl: UserTimeline) -> None:
        self._log("strategy_start", f"Running variable isolation analysis for {tl.user.user_id}")

        symptom_sessions: dict[str, list[AnnotatedSession]] = defaultdict(list)
        for s in tl.sessions:
            for sig in s.signals:
                if sig.signal_type == SignalType.SYMPTOM:
                    symptom_sessions[sig.value].append(s)

        for symptom_val, sessions in symptom_sessions.items():
            if len(sessions) < 3:
                continue

            trigger_presence: dict[str, list[bool]] = defaultdict(list)
            all_triggers_for_symptom: set[str] = set()
            for s in sessions:
                for sig in s.signals:
                    if sig.signal_type == SignalType.TRIGGER:
                        all_triggers_for_symptom.add(sig.value)

            if len(all_triggers_for_symptom) < 2:
                continue

            for s in sessions:
                session_triggers = {
                    sig.value for sig in s.signals
                    if sig.signal_type == SignalType.TRIGGER and not _is_negated(s, sig.value)
                }
                for t in all_triggers_for_symptom:
                    trigger_presence[t].append(t in session_triggers)

            trigger_consistency: dict[str, float] = {}
            for t, presences in trigger_presence.items():
                trigger_consistency[t] = sum(presences) / len(presences)

            sorted_triggers = sorted(
                trigger_consistency.items(), key=lambda x: -x[1]
            )

            if len(sorted_triggers) < 2:
                continue

            top_trigger, top_ratio = sorted_triggers[0]
            second_trigger, second_ratio = sorted_triggers[1]

            if top_ratio - second_ratio < 0.2:
                continue

            if top_ratio < 0.6:
                continue

            title = (
                f"{_label(top_trigger).capitalize()} identified as independent driver of "
                f"{_label(symptom_val)} over {_label(second_trigger)}"
            )

            temporal_parts = []
            for s in sessions:
                s_triggers = {
                    sig.value for sig in s.signals
                    if sig.signal_type == SignalType.TRIGGER and not _is_negated(s, sig.value)
                }
                date = s.conversation.timestamp.strftime("%b %d")
                present_labels = [_label(t) for t in sorted(s_triggers)]
                temporal_parts.append(
                    f"Session {s.conversation.session_id} ({date}): "
                    f"{_label(symptom_val)} present, triggers: "
                    f"{', '.join(present_labels) if present_labels else 'none detected'}"
                )
            temporal = (
                f"Variable isolation across {len(sessions)} sessions. "
                + " | ".join(temporal_parts)
                + f". {_label(top_trigger).capitalize()} present in {top_ratio:.0%} of episodes, "
                f"{_label(second_trigger)} in {second_ratio:.0%}."
            )

            if top_ratio >= 0.9 and (top_ratio - second_ratio) >= 0.3:
                confidence = Confidence.HIGH
            elif top_ratio >= 0.7:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            justification = (
                f"{_label(top_trigger).capitalize()} present in {top_ratio:.0%} of "
                f"{_label(symptom_val)} episodes vs {_label(second_trigger)} at "
                f"{second_ratio:.0%}, isolating it as the more consistent driver"
            )

            values = {symptom_val, top_trigger, second_trigger}
            supporting = []
            for s in sessions:
                supporting.extend(_build_supporting(s, values))

            alternatives = [
                f"Both {_label(top_trigger)} and {_label(second_trigger)} may contribute jointly",
                f"An unmeasured variable correlated with {_label(top_trigger)} may be the real cause",
                f"Sample size of {len(sessions)} episodes may be too small for reliable isolation",
            ]

            self._patterns.append(
                DetectedPattern(
                    pattern_id=self._next_id(),
                    user_id=tl.user.user_id,
                    title=title,
                    evidence_sessions=_sessions_to_ids(sessions),
                    temporal_reasoning=temporal,
                    confidence=confidence,
                    justification=justification,
                    supporting_signals=supporting,
                    alternative_explanations=alternatives,
                )
            )
            self._log(
                "variable_isolation_found",
                f"{tl.user.user_id}: {title} — {confidence.value}",
            )

    def _deduplicate(self) -> None:
        self._log("deduplication", f"Checking {len(self._patterns)} patterns for redundancy")

        to_remove: set[int] = set()
        for i in range(len(self._patterns)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(self._patterns)):
                if j in to_remove:
                    continue
                pi = self._patterns[i]
                pj = self._patterns[j]
                if pi.user_id != pj.user_id:
                    continue
                si = set(pi.evidence_sessions)
                sj = set(pj.evidence_sessions)
                overlap = si & sj
                smaller = min(len(si), len(sj))
                if smaller == 0:
                    continue

                pi_syms = {
                    s.extracted_text for s in pi.supporting_signals
                    if s.signal_type == "symptom"
                }
                pj_syms = {
                    s.extracted_text for s in pj.supporting_signals
                    if s.signal_type == "symptom"
                }

                if len(overlap) / smaller >= 0.8 and pi_syms == pj_syms:
                    if len(si) >= len(sj):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)

        if to_remove:
            self._patterns = [
                p for idx, p in enumerate(self._patterns) if idx not in to_remove
            ]
            self._log("deduplication_result", f"Removed {len(to_remove)} redundant patterns")
        else:
            self._log("deduplication_result", "No redundant patterns found")

        for i, p in enumerate(self._patterns):
            p.pattern_id = f"P-{i + 1:03d}"