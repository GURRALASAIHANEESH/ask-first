from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SignalType(Enum):
    SYMPTOM = "symptom"
    TRIGGER = "trigger"
    LIFESTYLE = "lifestyle"
    INTERVENTION = "intervention"
    IMPROVEMENT = "improvement"
    WORSENING = "worsening"
    TEMPORAL_MARKER = "temporal_marker"


class Confidence(Enum):
    VERY_LOW = "very low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very high"


@dataclass
class Conversation:
    session_id: str
    timestamp: datetime
    user_message: str
    clary_response: str
    severity: str
    tags: list[str]
    clary_questions: list[str] = field(default_factory=list)
    user_followup: Optional[str] = None

    def full_text(self) -> str:
        parts = [self.user_message]
        if self.user_followup:
            parts.append(self.user_followup)
        parts.append(self.clary_response)
        return " ".join(parts)


@dataclass
class User:
    user_id: str
    name: str
    age: int
    gender: str
    location: str
    occupation: str
    onboarding_notes: str
    conversations: list[Conversation]

    def sessions_sorted(self) -> list[Conversation]:
        return sorted(self.conversations, key=lambda c: c.timestamp)


@dataclass
class Dataset:
    version: str
    total_users: int
    total_conversations: int
    date_range: str
    users: list[User]


@dataclass
class Signal:
    signal_type: SignalType
    value: str
    raw_text: str
    session_id: str
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "signal_type": self.signal_type.value,
            "extracted_text": self.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AnnotatedSession:
    conversation: Conversation
    signals: list[Signal]


@dataclass
class UserTimeline:
    user: User
    sessions: list[AnnotatedSession]

    def get_signals_by_type(self, signal_type: SignalType) -> list[Signal]:
        result = []
        for s in self.sessions:
            for sig in s.signals:
                if sig.signal_type == signal_type:
                    result.append(sig)
        return result

    def get_session_by_id(self, session_id: str) -> Optional[AnnotatedSession]:
        for s in self.sessions:
            if s.conversation.session_id == session_id:
                return s
        return None


@dataclass
class SessionWindow:
    label: str
    session_ids: list[str]
    start: datetime
    end: datetime


@dataclass
class SupportingSignal:
    session_id: str
    signal_type: str
    extracted_text: str
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "signal_type": self.signal_type,
            "extracted_text": self.extracted_text,
            "timestamp": self.timestamp,
        }


@dataclass
class DetectedPattern:
    pattern_id: str
    user_id: str
    title: str
    evidence_sessions: list[str]
    temporal_reasoning: str
    confidence: Confidence
    justification: str
    supporting_signals: list[SupportingSignal]
    alternative_explanations: list[str]

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "user_id": self.user_id,
            "title": self.title,
            "evidence_sessions": self.evidence_sessions,
            "temporal_reasoning": self.temporal_reasoning,
            "confidence": self.confidence.value,
            "justification": self.justification,
            "supporting_signals": [s.to_dict() for s in self.supporting_signals],
            "alternative_explanations": self.alternative_explanations,
        }


@dataclass
class ReasoningStep:
    step: int
    action: str
    detail: str

    def to_dict(self) -> dict:
        return {"step": self.step, "action": self.action, "detail": self.detail}


@dataclass
class UserWindowInfo:
    window: str
    sessions: list[str]

    def to_dict(self) -> dict:
        return {"window": self.window, "sessions": self.sessions}


@dataclass
class ChunkingStrategy:
    method: str
    description: str
    per_user_windows: dict[str, list[UserWindowInfo]]

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "description": self.description,
            "per_user_windows": {
                uid: [w.to_dict() for w in windows]
                for uid, windows in self.per_user_windows.items()
            },
        }


@dataclass
class EngineOutput:
    detected_patterns: list[DetectedPattern]
    reasoning_trace: list[ReasoningStep]
    chunking_strategy: ChunkingStrategy
    failure_notes: list[str]

    def to_dict(self) -> dict:
        return {
            "detected_patterns": [p.to_dict() for p in self.detected_patterns],
            "reasoning_trace": [r.to_dict() for r in self.reasoning_trace],
            "chunking_strategy": self.chunking_strategy.to_dict(),
            "failure_notes": self.failure_notes,
        }