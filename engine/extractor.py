from __future__ import annotations
import re
from engine.models import (
    AnnotatedSession,
    Conversation,
    Signal,
    SignalType,
)


_SYMPTOM_PHRASES: dict[str, list[str]] = {
    "stomach_pain": [
        "stomach has been hurting",
        "stomach hurting",
        "burning kind of pain",
        "burning pain",
        "stomach is acting up",
        "stomach pain",
        "stomach again",
        "acidity",
    ],
    "headache": [
        "headaches",
        "headache",
        "splitting headache",
        "pounding",
        "pressure behind my eyes",
        "pressure, behind my eyes",
    ],
    "back_pain": [
        "lower back is bothering",
        "lower back",
        "back pain",
        "back strain",
    ],
    "fatigue": [
        "feeling really tired",
        "really tired",
        "exhausted",
        "feel exhausted",
        "feel tired",
        "tired all day",
        "tired lately",
        "fatigue",
        "cannot focus",
        "energy crash",
        "feel dead",
        "wake up exhausted",
        "wake up tired",
        "waking up tired",
    ],
    "dizziness": [
        "dizzy",
        "dizziness",
        "a little dizzy",
    ],
    "acne": [
        "breaking out",
        "breakout",
        "broke out",
        "acne",
        "skin has been breaking",
        "skin broke out",
    ],
    "hair_fall": [
        "hair has been falling",
        "hair falling",
        "hair fall",
        "hair loss",
        "finding it everywhere",
    ],
    "brain_fog": [
        "brain fog",
    ],
    "menstrual_cramps": [
        "cramps",
        "cramping",
        "period cramps",
        "cramps are really bad",
        "cramps started",
        "cramps have started",
    ],
    "anxiety": [
        "anxious",
        "anxiety",
        "general unease",
        "unease",
    ],
    "mood_low": [
        "mood has been a bit low",
        "mood dip",
        "low mood",
    ],
    "post_lunch_crash": [
        "exhausted after lunch",
        "cannot focus at all between 2 and 4",
        "2pm crash",
        "2pm slump",
        "afternoon crash",
    ],
}

_TRIGGER_PHRASES: dict[str, list[str]] = {
    "late_eating": [
        "dinner pretty late",
        "dinner at midnight",
        "ate around 11",
        "ate at 11",
        "dinner at 11",
        "had dinner at midnight",
        "dinner around 11",
        "11:30pm",
        "midnight",
        "eating late",
        "late dinner",
        "late eating",
    ],
    "dehydration": [
        "barely any water",
        "2 glasses of water",
        "3 glasses",
        "not enough water",
        "low water intake",
        "drink like 2 glasses",
        "low water",
    ],
    "work_stress": [
        "deadline",
        "sprint review",
        "work has been intense",
        "stressful",
        "big project",
        "massive release",
        "work pressure",
        "big deadline",
        "product launch",
        "high pressure",
        "big product launch",
        "another big deadline",
    ],
    "dairy_intake": [
        "greek yogurt",
        "paneer",
        "yogurt",
        "dairy",
        "more dairy",
        "added dairy",
        "added back dairy",
    ],
    "caffeine": [
        "coffee",
        "caffeine",
        "3 or 4 cups",
        "more coffee",
    ],
    "calorie_restriction": [
        "700-800 calories",
        "700 calories",
        "800 calories",
        "cutting down a lot on calories",
        "calorie restriction",
        "severe restriction",
        "low calories",
    ],
    "high_carb_meal": [
        "rice, dal",
        "big meal",
        "biscuits",
        "high sugar",
        "high carb",
    ],
    "screen_use_late": [
        "staying up late watching reels",
        "watching reels",
        "late night screen",
        "screens at night",
        "pick the phone up",
        "reels and series",
    ],
    "sedentary": [
        "sit for like 10 hours",
        "sit for 10 hours",
        "barely moved from my desk",
        "dont exercise",
        "no new exercise",
        "sedentary",
        "sitting for long",
    ],
    "sleep_deprivation": [
        "sleep around 1am",
        "sleep around 2am",
        "1 or 2am",
        "1am or 2am",
        "5-6 hours",
        "sleep has been broken",
        "sleep is bad",
        "sleep is still bad",
        "bad sleep",
        "broken sleep",
        "till 1 or 2am",
    ],
    "intermittent_fasting": [
        "intermittent fasting",
        "16:8",
        "fasting",
    ],
}

_LIFESTYLE_PHRASES: dict[str, list[str]] = {
    "eating_out": [
        "eating out",
        "eats out",
    ],
    "poor_diet": [
        "diet is not great",
        "eating out every day",
    ],
    "work_long_hours": [
        "long work session",
        "10 hours straight",
        "works late",
        "work session",
    ],
    "screen_all_day": [
        "screens all day",
        "staring at a screen",
        "screen time",
    ],
    "irregular_sleep": [
        "inconsistent sleep",
        "irregular sleep",
        "sleep timing",
    ],
}

_INTERVENTION_PHRASES: dict[str, list[str]] = {
    "dairy_reduction": [
        "cut dairy",
        "reducing dairy",
        "dairy reduction",
        "dairy cut",
    ],
    "dairy_reintroduction": [
        "added back dairy",
        "dairy reintroduction",
        "reintroduction",
    ],
    "protein_added": [
        "added chicken",
        "adding protein",
        "added protein",
        "protein to lunch",
        "protein added",
    ],
    "water_increase": [
        "drinking more water",
        "water bottle",
        "reminder to drink water",
    ],
    "calorie_increase": [
        "eating more",
        "increase to at least 1200",
        "increase calories",
        "started eating more",
    ],
    "screen_limit": [
        "hard stop on screens",
        "stopping at 11:30",
        "stop on screens",
    ],
    "posture_break": [
        "getting up every 45 minutes",
        "lumbar cushion",
    ],
}

_IMPROVEMENT_PHRASES: dict[str, list[str]] = {
    "symptom_improved": [
        "mostly gone",
        "so much better",
        "so much clearer",
        "noticeably better",
        "is better",
        "has basically stopped",
        "clearer",
        "improvement",
        "resolved",
        "actually got through a full afternoon",
    ],
    "condition_resolved": [
        "hair fall has basically stopped",
        "skin is clearer",
        "feeling so much better",
        "hair improvement",
        "hair fall stopped",
    ],
}

_WORSENING_PHRASES: dict[str, list[str]] = {
    "symptom_worse": [
        "even worse",
        "worse than usual",
        "really bad",
        "still bad",
        "spreading",
        "so much harder",
        "acne will come back",
        "dreading it",
    ],
    "symptom_recurring": [
        "again",
        "back",
        "is back",
        "still happening",
        "this has been happening",
        "same burning pain",
    ],
}

_TEMPORAL_PHRASES: dict[str, list[str]] = {
    "onset_recent": [
        "since last night",
        "this week",
        "started maybe",
        "started around",
        "2 weeks ago",
        "maybe 3 weeks",
        "3 weeks now",
    ],
    "recurrence_pattern": [
        "every day",
        "every single day",
        "every afternoon",
        "comes and goes",
        "every month",
        "every single time",
    ],
    "duration_marker": [
        "7 weeks",
        "6 weeks",
        "5 weeks",
        "3 weeks",
        "2 weeks",
        "one week",
        "48-72 hours",
    ],
}

_TAG_TO_SIGNAL: dict[str, tuple[SignalType, str]] = {
    "stomach": (SignalType.SYMPTOM, "stomach_pain"),
    "acidity": (SignalType.SYMPTOM, "stomach_pain"),
    "headache": (SignalType.SYMPTOM, "headache"),
    "back pain": (SignalType.SYMPTOM, "back_pain"),
    "fatigue": (SignalType.SYMPTOM, "fatigue"),
    "brain fog": (SignalType.SYMPTOM, "brain_fog"),
    "hair fall": (SignalType.SYMPTOM, "hair_fall"),
    "hair fall improving": (SignalType.IMPROVEMENT, "symptom_improved"),
    "hair fall resolved": (SignalType.IMPROVEMENT, "condition_resolved"),
    "skin": (SignalType.SYMPTOM, "acne"),
    "acne": (SignalType.SYMPTOM, "acne"),
    "dizziness": (SignalType.SYMPTOM, "dizziness"),
    "period": (SignalType.SYMPTOM, "menstrual_cramps"),
    "cramps": (SignalType.SYMPTOM, "menstrual_cramps"),
    "anxiety": (SignalType.SYMPTOM, "anxiety"),
    "mood": (SignalType.SYMPTOM, "mood_low"),
    "late eating": (SignalType.TRIGGER, "late_eating"),
    "dehydration": (SignalType.TRIGGER, "dehydration"),
    "stress": (SignalType.TRIGGER, "work_stress"),
    "work pressure": (SignalType.TRIGGER, "work_stress"),
    "dairy": (SignalType.TRIGGER, "dairy_intake"),
    "dairy increase": (SignalType.TRIGGER, "dairy_intake"),
    "caffeine": (SignalType.TRIGGER, "caffeine"),
    "calorie restriction": (SignalType.TRIGGER, "calorie_restriction"),
    "under-fuelling": (SignalType.TRIGGER, "calorie_restriction"),
    "high carb lunch": (SignalType.TRIGGER, "high_carb_meal"),
    "blood sugar": (SignalType.TRIGGER, "high_carb_meal"),
    "screen time": (SignalType.LIFESTYLE, "screen_all_day"),
    "late night screen use": (SignalType.TRIGGER, "screen_use_late"),
    "screens": (SignalType.TRIGGER, "screen_use_late"),
    "sedentary": (SignalType.TRIGGER, "sedentary"),
    "posture": (SignalType.TRIGGER, "sedentary"),
    "sleep": (SignalType.TRIGGER, "sleep_deprivation"),
    "sleep deprivation": (SignalType.TRIGGER, "sleep_deprivation"),
    "melatonin": (SignalType.TRIGGER, "sleep_deprivation"),
    "intermittent fasting": (SignalType.TRIGGER, "intermittent_fasting"),
    "diet": (SignalType.LIFESTYLE, "poor_diet"),
    "busy work day": (SignalType.LIFESTYLE, "work_long_hours"),
    "dairy reduction": (SignalType.INTERVENTION, "dairy_reduction"),
    "dairy reintroduction": (SignalType.INTERVENTION, "dairy_reintroduction"),
    "lunch protein": (SignalType.INTERVENTION, "protein_added"),
    "protein deficiency": (SignalType.TRIGGER, "high_carb_meal"),
    "snacking": (SignalType.TRIGGER, "high_carb_meal"),
    "post-lunch": (SignalType.SYMPTOM, "post_lunch_crash"),
    "pattern confirmed": (SignalType.TEMPORAL_MARKER, "recurrence_pattern"),
    "pattern very clear": (SignalType.TEMPORAL_MARKER, "recurrence_pattern"),
    "pattern fully confirmed": (SignalType.TEMPORAL_MARKER, "recurrence_pattern"),
    "second month pattern": (SignalType.TEMPORAL_MARKER, "recurrence_pattern"),
    "improvement": (SignalType.IMPROVEMENT, "symptom_improved"),
    "improvement confirmed": (SignalType.IMPROVEMENT, "symptom_improved"),
    "recovery": (SignalType.IMPROVEMENT, "condition_resolved"),
    "skin improved": (SignalType.IMPROVEMENT, "symptom_improved"),
    "nutrition improving": (SignalType.IMPROVEMENT, "symptom_improved"),
    "dairy pattern confirmed": (SignalType.TEMPORAL_MARKER, "recurrence_pattern"),
    "temporal connection": (SignalType.TEMPORAL_MARKER, "duration_marker"),
    "january diet": (SignalType.TEMPORAL_MARKER, "onset_recent"),
    "testing hypothesis": (SignalType.TEMPORAL_MARKER, "recurrence_pattern"),
    "sleep deprivation confirmed driver": (SignalType.TEMPORAL_MARKER, "recurrence_pattern"),
    "hormonal": (SignalType.TEMPORAL_MARKER, "recurrence_pattern"),
    "cortisol": (SignalType.TEMPORAL_MARKER, "duration_marker"),
    "product launch": (SignalType.TRIGGER, "work_stress"),
    "deadline": (SignalType.TRIGGER, "work_stress"),
    "cheeks": (SignalType.SYMPTOM, "acne"),
    "jawline": (SignalType.SYMPTOM, "acne"),
    "telogen effluvium": (SignalType.SYMPTOM, "hair_fall"),
}

_ALL_TAXONOMIES: list[tuple[SignalType, dict[str, list[str]]]] = [
    (SignalType.SYMPTOM, _SYMPTOM_PHRASES),
    (SignalType.TRIGGER, _TRIGGER_PHRASES),
    (SignalType.LIFESTYLE, _LIFESTYLE_PHRASES),
    (SignalType.INTERVENTION, _INTERVENTION_PHRASES),
    (SignalType.IMPROVEMENT, _IMPROVEMENT_PHRASES),
    (SignalType.WORSENING, _WORSENING_PHRASES),
    (SignalType.TEMPORAL_MARKER, _TEMPORAL_PHRASES),
]


def _extract_surrounding(text: str, phrase: str, window: int = 80) -> str:
    lower = text.lower()
    idx = lower.find(phrase.lower())
    if idx == -1:
        return phrase
    start = max(0, idx - window)
    end = min(len(text), idx + len(phrase) + window)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def extract_signals_from_text(
    text: str,
    session_id: str,
    timestamp,
) -> list[Signal]:
    signals: list[Signal] = []
    seen: set[tuple[str, str]] = set()
    lower = text.lower()

    for signal_type, categories in _ALL_TAXONOMIES:
        for category, phrases in categories.items():
            sorted_phrases = sorted(phrases, key=len, reverse=True)
            for phrase in sorted_phrases:
                if phrase.lower() in lower:
                    key = (signal_type.value, category)
                    if key not in seen:
                        seen.add(key)
                        raw = _extract_surrounding(text, phrase)
                        signals.append(
                            Signal(
                                signal_type=signal_type,
                                value=category,
                                raw_text=raw,
                                session_id=session_id,
                                timestamp=timestamp,
                            )
                        )
                    break
    return signals


def extract_signals_from_tags(
    tags: list[str],
    session_id: str,
    timestamp,
) -> list[Signal]:
    signals: list[Signal] = []
    seen: set[tuple[str, str]] = set()

    for tag in tags:
        tag_lower = tag.lower().strip()
        if tag_lower in _TAG_TO_SIGNAL:
            signal_type, value = _TAG_TO_SIGNAL[tag_lower]
            key = (signal_type.value, value)
            if key not in seen:
                seen.add(key)
                signals.append(
                    Signal(
                        signal_type=signal_type,
                        value=value,
                        raw_text=f"[tag] {tag}",
                        session_id=session_id,
                        timestamp=timestamp,
                    )
                )
    return signals


def annotate_session(conversation: Conversation) -> AnnotatedSession:
    full_text = conversation.full_text()
    text_signals = extract_signals_from_text(
        full_text,
        conversation.session_id,
        conversation.timestamp,
    )
    tag_signals = extract_signals_from_tags(
        conversation.tags,
        conversation.session_id,
        conversation.timestamp,
    )

    merged: list[Signal] = []
    seen: set[tuple[str, str]] = set()

    for sig in text_signals:
        key = (sig.signal_type.value, sig.value)
        if key not in seen:
            seen.add(key)
            merged.append(sig)

    for sig in tag_signals:
        key = (sig.signal_type.value, sig.value)
        if key not in seen:
            seen.add(key)
            merged.append(sig)

    return AnnotatedSession(conversation=conversation, signals=merged)


def annotate_all(conversations: list[Conversation]) -> list[AnnotatedSession]:
    sorted_convos = sorted(conversations, key=lambda c: c.timestamp)
    return [annotate_session(c) for c in sorted_convos]