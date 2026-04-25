from __future__ import annotations
import time
from typing import Generator

import streamlit as st

from engine.loader import load_dataset
from engine.timeline import build_all_timelines, compute_session_windows, compute_gap_days
from engine.detector import PatternDetector
from engine.scorer import recalibrate
from engine.output import build_output
from engine.models import (
    Confidence,
    DetectedPattern,
    EngineOutput,
    UserTimeline,
)


DATASET_PATH = "dataset.json"

USER_DISPLAY = {
    "USR001": "Arjun",
    "USR002": "Meera",
    "USR003": "Priya",
}


@st.cache_data(show_spinner=False)
def run_engine() -> dict:
    ds = load_dataset(DATASET_PATH)
    timelines = build_all_timelines(ds.users)
    detector = PatternDetector()
    patterns, reasoning = detector.detect(timelines)
    patterns, reasoning = recalibrate(patterns, timelines, reasoning)
    output = build_output(patterns, reasoning, timelines)

    tl_map = {tl.user.user_id: tl for tl in timelines}

    result = {
        "output": output.to_dict(),
        "patterns_by_user": {},
        "timelines": {},
        "user_info": {},
    }

    for tl in timelines:
        uid = tl.user.user_id
        result["patterns_by_user"][uid] = [
            p for p in patterns if p.user_id == uid
        ]
        result["timelines"][uid] = tl
        result["user_info"][uid] = {
            "name": tl.user.name,
            "age": tl.user.age,
            "gender": tl.user.gender,
            "location": tl.user.location,
            "occupation": tl.user.occupation,
            "onboarding_notes": tl.user.onboarding_notes,
            "session_count": len(tl.sessions),
        }

    return result


def stream_text(text: str, delay: float = 0.02) -> Generator[str, None, None]:
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)


def format_confidence_badge(confidence: Confidence) -> str:
    colors = {
        Confidence.VERY_HIGH: "🟢",
        Confidence.HIGH: "🔵",
        Confidence.MEDIUM: "🟡",
        Confidence.LOW: "🟠",
        Confidence.VERY_LOW: "🔴",
    }
    return f"{colors.get(confidence, '⚪')} {confidence.value.upper()}"


def build_timeline_text(tl: UserTimeline) -> str:
    lines = [f"**Timeline for {tl.user.name}** — {len(tl.sessions)} sessions\n"]
    for i, s in enumerate(tl.sessions):
        conv = s.conversation
        date_str = conv.timestamp.strftime("%b %d, %Y %I:%M %p")
        symptoms = [sig.value.replace("_", " ") for sig in s.signals if sig.signal_type.value == "symptom"]
        triggers = [sig.value.replace("_", " ") for sig in s.signals if sig.signal_type.value == "trigger"]
        interventions = [sig.value.replace("_", " ") for sig in s.signals if sig.signal_type.value == "intervention"]

        gap_str = ""
        if i > 0:
            prev_ts = tl.sessions[i - 1].conversation.timestamp
            gap = compute_gap_days(prev_ts, conv.timestamp)
            gap_str = f" *(+{gap:.0f} days)*"

        line = f"**{conv.session_id}** — {date_str}{gap_str}  \n"
        line += f"  Severity: {conv.severity}"
        if symptoms:
            line += f" | Symptoms: {', '.join(symptoms)}"
        if triggers:
            line += f" | Triggers: {', '.join(triggers)}"
        if interventions:
            line += f" | Interventions: {', '.join(interventions)}"
        lines.append(line)

    return "\n\n".join(lines)


def build_patterns_overview(patterns: list[DetectedPattern]) -> str:
    if not patterns:
        return "No patterns detected for this user."

    lines = [f"I found **{len(patterns)} pattern(s)** in the conversation history:\n"]
    for p in patterns:
        badge = format_confidence_badge(p.confidence)
        lines.append(f"**{p.pattern_id}: {p.title}**  \n"
                      f"Confidence: {badge}  \n"
                      f"Evidence: {', '.join(p.evidence_sessions)}  \n"
                      f"{p.temporal_reasoning}\n")
    return "\n---\n".join(lines)


def build_pattern_detail(pattern: DetectedPattern) -> str:
    badge = format_confidence_badge(pattern.confidence)
    lines = [
        f"**{pattern.pattern_id}: {pattern.title}**\n",
        f"**Confidence:** {badge}\n",
        f"**Evidence sessions:** {', '.join(pattern.evidence_sessions)}\n",
        f"**Temporal reasoning:** {pattern.temporal_reasoning}\n",
        f"**Justification:** {pattern.justification}\n",
    ]
    if pattern.supporting_signals:
        sig_lines = []
        for sig in pattern.supporting_signals[:10]:
            sig_lines.append(f"- [{sig.signal_type}] {sig.extracted_text} ({sig.session_id}, {sig.timestamp[:10]})")
        lines.append("**Supporting signals:**\n" + "\n".join(sig_lines) + "\n")

    if pattern.alternative_explanations:
        alt_lines = [f"- {a}" for a in pattern.alternative_explanations]
        lines.append("**Alternative explanations:**\n" + "\n".join(alt_lines) + "\n")

    return "\n".join(lines)


def build_timing_explanation(patterns: list[DetectedPattern]) -> str:
    if not patterns:
        return "No patterns to explain timing for."

    lines = ["Here's how timing matters in each detected pattern:\n"]
    for p in patterns:
        lines.append(f"**{p.pattern_id} — {p.title}**  \n"
                      f"{p.temporal_reasoning}\n")
    return "\n---\n".join(lines)


def build_changes_over_time(tl: UserTimeline, patterns: list[DetectedPattern]) -> str:
    lines = [f"**How {tl.user.name}'s health picture changed over time:**\n"]

    windows = compute_session_windows(tl)
    for w in windows:
        w_sessions = [s for s in tl.sessions if s.conversation.session_id in w.session_ids]
        symptoms = set()
        triggers = set()
        improvements = False
        for s in w_sessions:
            for sig in s.signals:
                if sig.signal_type.value == "symptom":
                    symptoms.add(sig.value.replace("_", " "))
                elif sig.signal_type.value == "trigger":
                    triggers.add(sig.value.replace("_", " "))
                elif sig.signal_type.value == "improvement":
                    improvements = True

        status = "📈 improvement noted" if improvements else "📋 ongoing"
        lines.append(f"**{w.label}** ({', '.join(w.session_ids)}): {status}  \n"
                      f"  Symptoms: {', '.join(symptoms) if symptoms else 'none'}  \n"
                      f"  Triggers: {', '.join(triggers) if triggers else 'none'}")

    if patterns:
        lines.append("\n**Key shifts detected:**")
        for p in patterns:
            if "improvement" in p.temporal_reasoning.lower() or "resolved" in p.temporal_reasoning.lower():
                lines.append(f"- {p.title}: evidence of improvement")
            elif "worsening" in p.temporal_reasoning.lower() or "worse" in p.temporal_reasoning.lower():
                lines.append(f"- {p.title}: evidence of worsening")

    return "\n\n".join(lines)


def build_json_view(patterns: list[DetectedPattern]) -> str:
    import json
    data = [p.to_dict() for p in patterns]
    return json.dumps(data, indent=2, ensure_ascii=False)


def match_intent(user_input: str, patterns: list[DetectedPattern], tl: UserTimeline) -> str:
    lower = user_input.lower().strip()

    if any(kw in lower for kw in ["pattern", "what do you see", "what did you find", "findings", "summary", "overview"]):
        return build_patterns_overview(patterns)

    if any(kw in lower for kw in ["timeline", "history", "sessions", "conversations"]):
        return build_timeline_text(tl)

    if any(kw in lower for kw in ["change", "changed", "over time", "progress", "trajectory"]):
        return build_changes_over_time(tl, patterns)

    if any(kw in lower for kw in ["timing", "explain the timing", "temporal", "when"]):
        return build_timing_explanation(patterns)

    if any(kw in lower for kw in ["json", "structured", "raw", "export"]):
        return "```json\n" + build_json_view(patterns) + "\n```"

    if any(kw in lower for kw in ["high confidence", "why high", "strongest", "most confident", "best evidence"]):
        high = [p for p in patterns if p.confidence in (Confidence.VERY_HIGH, Confidence.HIGH)]
        if high:
            lines = ["These are the highest-confidence patterns:\n"]
            for p in high:
                lines.append(build_pattern_detail(p))
            return "\n---\n".join(lines)
        return "No high-confidence patterns found for this user."

    if any(kw in lower for kw in ["low confidence", "weak", "uncertain", "least confident"]):
        low = [p for p in patterns if p.confidence in (Confidence.LOW, Confidence.VERY_LOW)]
        if low:
            lines = ["These patterns have lower confidence:\n"]
            for p in low:
                lines.append(build_pattern_detail(p))
            return "\n---\n".join(lines)
        return "All patterns have medium or higher confidence."

    if any(kw in lower for kw in ["alternative", "other explanation", "what else", "could it be"]):
        lines = ["Here are the alternative explanations considered for each pattern:\n"]
        for p in patterns:
            if p.alternative_explanations:
                alts = "\n".join(f"  - {a}" for a in p.alternative_explanations)
                lines.append(f"**{p.pattern_id} — {p.title}:**\n{alts}\n")
        return "\n".join(lines)

    if any(kw in lower for kw in ["reasoning trace", "reasoning", "how did you decide", "show trace", "steps", "how did you reach"]):
        output_data = run_engine()["output"]
        trace = output_data.get("reasoning_trace", [])
        uid = tl.user.user_id
        user_trace = [
            r for r in trace
            if uid in r.get("detail", "")
            or r.get("action") in ("scoring_start", "scoring_complete", "deduplication", "deduplication_result")
        ]
        if not user_trace:
            user_trace = trace
        lines = [f"**Reasoning trace** ({len(user_trace)} steps for {tl.user.name}):\n"]
        for r in user_trace[:30]:
            lines.append(f"**Step {r['step']}** [{r['action']}]: {r['detail']}")
        if len(user_trace) > 30:
            lines.append(f"\n*...and {len(user_trace) - 30} more steps. Showing first 30.*")
        return "\n\n".join(lines)

    if any(kw in lower for kw in ["limitation", "fail", "miss", "wrong", "gap", "weakness"]):
        output = run_engine()["output"]
        notes = output.get("failure_notes", [])
        if notes:
            lines = ["Here's where the system may fall short:\n"]
            for i, note in enumerate(notes, 1):
                lines.append(f"{i}. {note}")
            return "\n\n".join(lines)
        return "No failure notes available."

    # Try to match a specific pattern ID
    for p in patterns:
        pid_lower = p.pattern_id.lower().replace("-", "")
        if pid_lower in lower.replace("-", "").replace(" ", ""):
            return build_pattern_detail(p)

    # Try to match a pattern by keyword in title
    for p in patterns:
        title_words = p.title.lower().split()
        if any(tw in lower for tw in title_words if len(tw) > 3):
            return build_pattern_detail(p)

    return (
        "I can help you explore this user's health patterns. Try asking:\n\n"
        "- **\"What patterns do you see?\"** — overview of all detected patterns\n"
        "- **\"Show the timeline\"** — full session history\n"
        "- **\"Explain the timing\"** — temporal reasoning for each pattern\n"
        "- **\"What changed over time?\"** — progress and trajectory\n"
        "- **\"Why is this pattern high confidence?\"** — strongest findings\n"
        "- **\"What are the alternative explanations?\"** — what else it could be\n"
        "- **\"Show JSON\"** — structured output\n"
        "- **\"What could the system miss?\"** — honest limitations\n"
        "- **\"P-001\"** — details on a specific pattern\n"
    )


def main():
    st.set_page_config(
        page_title="Ask First",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Ask First")
    st.caption("Health clarity through temporal reasoning — no diagnosis, just patterns from data.")

    with st.spinner("Running analysis engine..."):
        data = run_engine()

    # Sidebar
    with st.sidebar:
        st.header("Select User")
        user_options = {f"{v} ({k})": k for k, v in USER_DISPLAY.items()}
        selected_label = st.selectbox("User", list(user_options.keys()))
        selected_uid = user_options[selected_label]
        user_info = data["user_info"][selected_uid]

        st.markdown("---")
        st.subheader(f"About {user_info['name']}")
        st.markdown(f"**Age:** {user_info['age']}  \n"
                     f"**Gender:** {user_info['gender']}  \n"
                     f"**Location:** {user_info['location']}  \n"
                     f"**Occupation:** {user_info['occupation']}  \n"
                     f"**Sessions:** {user_info['session_count']}")
        if user_info["onboarding_notes"]:
            st.markdown(f"**Notes:** {user_info['onboarding_notes']}")

        st.markdown("---")
        patterns = data["patterns_by_user"][selected_uid]
        st.subheader(f"Patterns: {len(patterns)}")
        for p in patterns:
            badge = format_confidence_badge(p.confidence)
            st.markdown(f"- {badge} {p.title}")

    # Chat state
    chat_key = f"messages_{selected_uid}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # Clear chat when switching users
    if "current_uid" not in st.session_state:
        st.session_state.current_uid = selected_uid
    elif st.session_state.current_uid != selected_uid:
        st.session_state[chat_key] = []
        st.session_state.current_uid = selected_uid

    tl = data["timelines"][selected_uid]
    patterns = data["patterns_by_user"][selected_uid]

    # Display chat history
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Welcome message
    if not st.session_state[chat_key]:
        welcome = (
            f"I've analyzed **{user_info['name']}'s** {user_info['session_count']} "
            f"conversations and found **{len(patterns)} pattern(s)**. "
            f"Ask me anything about their health history — patterns, timing, "
            f"what changed, or why I'm confident in a finding."
        )
        with st.chat_message("assistant"):
            st.write_stream(stream_text(welcome))
        st.session_state[chat_key].append({"role": "assistant", "content": welcome})

    # User input
    if prompt := st.chat_input(f"Ask about {user_info['name']}'s patterns..."):
        st.session_state[chat_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = match_intent(prompt, patterns, tl)

        with st.chat_message("assistant"):
            st.write_stream(stream_text(response, delay=0.015))
        st.session_state[chat_key].append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()