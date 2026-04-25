from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path

from engine.models import Conversation, Dataset, User


def load_dataset(path: str) -> Dataset:
    file_path = Path(path)
    if not file_path.exists():
        print(f"[ERROR] Dataset file not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    if not file_path.suffix == ".json":
        print(f"[ERROR] Dataset must be a JSON file, got: {file_path.suffix}", file=sys.stderr)
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    _validate_top_level(raw)
    info = raw["dataset_info"]
    users = [_parse_user(u) for u in raw["users"]]

    actual_convos = sum(len(u.conversations) for u in users)
    if actual_convos != info["total_conversations"]:
        print(
            f"[WARN] dataset_info says {info['total_conversations']} conversations "
            f"but found {actual_convos}",
            file=sys.stderr,
        )

    return Dataset(
        version=info["version"],
        total_users=info["total_users"],
        total_conversations=info["total_conversations"],
        date_range=info["date_range"],
        users=users,
    )


def _validate_top_level(raw: dict) -> None:
    required_keys = {"dataset_info", "users"}
    missing = required_keys - set(raw.keys())
    if missing:
        print(f"[ERROR] Dataset missing top-level keys: {missing}", file=sys.stderr)
        sys.exit(1)

    info_keys = {"version", "total_users", "total_conversations", "date_range"}
    missing_info = info_keys - set(raw["dataset_info"].keys())
    if missing_info:
        print(f"[ERROR] dataset_info missing keys: {missing_info}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(raw["users"], list) or len(raw["users"]) == 0:
        print("[ERROR] 'users' must be a non-empty list", file=sys.stderr)
        sys.exit(1)


def _parse_user(raw_user: dict) -> User:
    required = {"user_id", "name", "age", "gender", "location", "occupation", "conversations"}
    missing = required - set(raw_user.keys())
    if missing:
        print(f"[ERROR] User object missing keys: {missing}", file=sys.stderr)
        sys.exit(1)

    conversations = [_parse_conversation(c) for c in raw_user["conversations"]]

    return User(
        user_id=raw_user["user_id"],
        name=raw_user["name"],
        age=raw_user["age"],
        gender=raw_user["gender"],
        location=raw_user["location"],
        occupation=raw_user["occupation"],
        onboarding_notes=raw_user.get("onboarding_notes", ""),
        conversations=conversations,
    )


def _parse_conversation(raw_conv: dict) -> Conversation:
    required = {"session_id", "timestamp", "user_message", "clary_response", "severity", "tags"}
    missing = required - set(raw_conv.keys())
    if missing:
        print(
            f"[ERROR] Conversation {raw_conv.get('session_id', '?')} missing keys: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        ts = datetime.fromisoformat(raw_conv["timestamp"])
    except ValueError:
        print(
            f"[ERROR] Invalid timestamp in {raw_conv['session_id']}: {raw_conv['timestamp']}",
            file=sys.stderr,
        )
        sys.exit(1)

    return Conversation(
        session_id=raw_conv["session_id"],
        timestamp=ts,
        user_message=raw_conv["user_message"],
        clary_response=raw_conv["clary_response"],
        severity=raw_conv["severity"],
        tags=raw_conv["tags"],
        clary_questions=raw_conv.get("clary_questions", []),
        user_followup=raw_conv.get("user_followup"),
    )