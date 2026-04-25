from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta

from engine.models import (
    AnnotatedSession,
    SessionWindow,
    User,
    UserTimeline,
)
from engine.extractor import annotate_all


def build_user_timeline(user: User) -> UserTimeline:
    sessions = annotate_all(user.conversations)
    return UserTimeline(user=user, sessions=sessions)


def build_all_timelines(users: list[User]) -> list[UserTimeline]:
    return [build_user_timeline(u) for u in users]


def compute_session_windows(timeline: UserTimeline) -> list[SessionWindow]:
    month_groups: dict[str, list[AnnotatedSession]] = defaultdict(list)
    for s in timeline.sessions:
        key = s.conversation.timestamp.strftime("%Y-%m")
        month_groups[key].append(s)

    windows: list[SessionWindow] = []
    for month_key in sorted(month_groups.keys()):
        sessions_in_month = month_groups[month_key]
        sessions_in_month.sort(key=lambda s: s.conversation.timestamp)

        early: list[AnnotatedSession] = []
        late: list[AnnotatedSession] = []
        for s in sessions_in_month:
            if s.conversation.timestamp.day <= 15:
                early.append(s)
            else:
                late.append(s)

        if early:
            windows.append(
                SessionWindow(
                    label=f"{month_key}-early",
                    session_ids=[s.conversation.session_id for s in early],
                    start=early[0].conversation.timestamp,
                    end=early[-1].conversation.timestamp,
                )
            )
        if late:
            windows.append(
                SessionWindow(
                    label=f"{month_key}-late",
                    session_ids=[s.conversation.session_id for s in late],
                    start=late[0].conversation.timestamp,
                    end=late[-1].conversation.timestamp,
                )
            )

    return windows


def get_sessions_in_range(
    timeline: UserTimeline,
    start: datetime,
    end: datetime,
) -> list[AnnotatedSession]:
    return [
        s for s in timeline.sessions
        if start <= s.conversation.timestamp <= end
    ]


def get_prior_sessions(
    timeline: UserTimeline,
    before: datetime,
    max_count: int | None = None,
) -> list[AnnotatedSession]:
    prior = [
        s for s in timeline.sessions
        if s.conversation.timestamp < before
    ]
    prior.sort(key=lambda s: s.conversation.timestamp)
    if max_count is not None:
        prior = prior[-max_count:]
    return prior


def get_sessions_with_signal_value(
    timeline: UserTimeline,
    value: str,
) -> list[AnnotatedSession]:
    return [
        s for s in timeline.sessions
        if any(sig.value == value for sig in s.signals)
    ]


def compute_gap_days(ts1: datetime, ts2: datetime) -> float:
    delta = abs(ts2 - ts1)
    return delta.total_seconds() / 86400


def find_co_occurring_signals(
    timeline: UserTimeline,
    value_a: str,
    value_b: str,
) -> list[AnnotatedSession]:
    results = []
    for s in timeline.sessions:
        values_in_session = {sig.value for sig in s.signals}
        if value_a in values_in_session and value_b in values_in_session:
            results.append(s)
    return results


def find_signal_sequences(
    timeline: UserTimeline,
    trigger_value: str,
    symptom_value: str,
    max_gap_days: float = 30.0,
) -> list[tuple[AnnotatedSession, AnnotatedSession]]:
    trigger_sessions = get_sessions_with_signal_value(timeline, trigger_value)
    symptom_sessions = get_sessions_with_signal_value(timeline, symptom_value)

    pairs: list[tuple[AnnotatedSession, AnnotatedSession]] = []
    used_symptoms: set[str] = set()

    for t_sess in trigger_sessions:
        t_time = t_sess.conversation.timestamp
        best_match: AnnotatedSession | None = None
        best_gap = max_gap_days + 1

        for s_sess in symptom_sessions:
            s_time = s_sess.conversation.timestamp
            if s_time <= t_time:
                continue
            gap = compute_gap_days(t_time, s_time)
            if gap <= max_gap_days and gap < best_gap:
                if s_sess.conversation.session_id not in used_symptoms:
                    best_match = s_sess
                    best_gap = gap

        if best_match is not None:
            if best_match.conversation.session_id == t_sess.conversation.session_id:
                continue
            pairs.append((t_sess, best_match))
            used_symptoms.add(best_match.conversation.session_id)

    return pairs


def find_intervention_outcomes(
    timeline: UserTimeline,
    intervention_value: str,
    symptom_value: str,
) -> list[dict]:
    intervention_sessions = get_sessions_with_signal_value(timeline, intervention_value)
    results = []

    for i_sess in intervention_sessions:
        i_time = i_sess.conversation.timestamp
        after = [
            s for s in timeline.sessions
            if s.conversation.timestamp > i_time
        ]
        after.sort(key=lambda s: s.conversation.timestamp)

        for a_sess in after[:3]:
            a_values = {sig.value for sig in a_sess.signals}
            improved = any(
                sig.value in ("symptom_improved", "condition_resolved")
                for sig in a_sess.signals
            )
            worsened = symptom_value in a_values and any(
                sig.value in ("symptom_worse", "symptom_recurring")
                for sig in a_sess.signals
            )
            if improved or worsened:
                results.append({
                    "intervention_session": i_sess,
                    "outcome_session": a_sess,
                    "improved": improved,
                    "worsened": worsened,
                    "gap_days": compute_gap_days(
                        i_time, a_sess.conversation.timestamp
                    ),
                })
                break

    return results