from __future__ import annotations
import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.models import (
    AnnotatedSession,
    Confidence,
    Conversation,
    Dataset,
    DetectedPattern,
    EngineOutput,
    Signal,
    SignalType,
    User,
    UserTimeline,
)
from engine.loader import load_dataset
from engine.extractor import annotate_session, annotate_all, extract_signals_from_text
from engine.timeline import (
    build_user_timeline,
    build_all_timelines,
    compute_session_windows,
    compute_gap_days,
    find_co_occurring_signals,
    get_sessions_with_signal_value,
)
from engine.detector import PatternDetector
from engine.scorer import recalibrate
from engine.output import build_output


DATASET_PATH = os.environ.get("ASKFIRST_DATASET", "dataset.json")


def _dataset_available() -> bool:
    return Path(DATASET_PATH).exists()


class TestLoader(unittest.TestCase):

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_load_returns_dataset(self):
        ds = load_dataset(DATASET_PATH)
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(ds.total_users, 3)
        self.assertEqual(ds.total_conversations, 27)
        self.assertEqual(len(ds.users), 3)

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_user_ids(self):
        ds = load_dataset(DATASET_PATH)
        ids = {u.user_id for u in ds.users}
        self.assertEqual(ids, {"USR001", "USR002", "USR003"})

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_conversation_count_per_user(self):
        ds = load_dataset(DATASET_PATH)
        for u in ds.users:
            self.assertEqual(len(u.conversations), 9, f"{u.user_id} should have 9 conversations")

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_timestamps_are_datetime(self):
        ds = load_dataset(DATASET_PATH)
        for u in ds.users:
            for c in u.conversations:
                self.assertIsInstance(c.timestamp, datetime)


class TestExtractor(unittest.TestCase):

    def _make_conversation(self, text: str, followup: str = None, tags: list[str] = None) -> Conversation:
        return Conversation(
            session_id="TEST_S01",
            timestamp=datetime(2026, 1, 10, 12, 0, 0),
            user_message=text,
            clary_response="noted",
            severity="mild",
            tags=tags or [],
            user_followup=followup,
        )

    def test_stomach_pain_detected(self):
        conv = self._make_conversation("my stomach has been hurting since last night")
        ann = annotate_session(conv)
        values = {s.value for s in ann.signals}
        self.assertIn("stomach_pain", values)

    def test_headache_detected(self):
        conv = self._make_conversation("I have been getting headaches in the afternoon")
        ann = annotate_session(conv)
        values = {s.value for s in ann.signals}
        self.assertIn("headache", values)

    def test_late_eating_trigger(self):
        conv = self._make_conversation(
            "stomach hurts",
            followup="ate around 11pm, had a deadline",
            tags=["stomach", "late eating"],
        )
        ann = annotate_session(conv)
        values = {s.value for s in ann.signals}
        self.assertIn("late_eating", values)

    def test_dehydration_trigger(self):
        conv = self._make_conversation(
            "headache again",
            followup="barely any water today",
        )
        ann = annotate_session(conv)
        values = {s.value for s in ann.signals}
        self.assertIn("dehydration", values)

    def test_dairy_trigger(self):
        conv = self._make_conversation(
            "skin breaking out",
            followup="having greek yogurt and paneer a lot",
            tags=["skin", "dairy"],
        )
        ann = annotate_session(conv)
        values = {s.value for s in ann.signals}
        self.assertIn("dairy_intake", values)
        self.assertIn("acne", values)

    def test_calorie_restriction_trigger(self):
        conv = self._make_conversation(
            "started intermittent fasting, eating maybe 700-800 calories a day"
        )
        ann = annotate_session(conv)
        values = {s.value for s in ann.signals}
        self.assertIn("calorie_restriction", values)

    def test_hair_fall_detected(self):
        conv = self._make_conversation("my hair has been falling out so much")
        ann = annotate_session(conv)
        values = {s.value for s in ann.signals}
        self.assertIn("hair_fall", values)

    def test_menstrual_cramps_detected(self):
        conv = self._make_conversation("period cramps are really bad this month")
        ann = annotate_session(conv)
        values = {s.value for s in ann.signals}
        self.assertIn("menstrual_cramps", values)

    def test_screen_use_trigger(self):
        conv = self._make_conversation("staying up late watching reels and series")
        ann = annotate_session(conv)
        values = {s.value for s in ann.signals}
        self.assertIn("screen_use_late", values)

    def test_improvement_detected(self):
        conv = self._make_conversation(
            "headaches are mostly gone",
            tags=["improvement"],
        )
        ann = annotate_session(conv)
        types = {s.signal_type for s in ann.signals}
        self.assertIn(SignalType.IMPROVEMENT, types)

    def test_no_duplicate_signals(self):
        conv = self._make_conversation(
            "stomach has been hurting, burning pain in stomach",
            followup="stomach is acting up again",
            tags=["stomach", "acidity"],
        )
        ann = annotate_session(conv)
        symptom_signals = [s for s in ann.signals if s.value == "stomach_pain"]
        self.assertEqual(len(symptom_signals), 1)

    def test_empty_text(self):
        conv = self._make_conversation("")
        ann = annotate_session(conv)
        text_signals = [s for s in ann.signals if not s.raw_text.startswith("[tag]")]
        self.assertEqual(len(text_signals), 0)


class TestTimeline(unittest.TestCase):

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_build_timelines(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        self.assertEqual(len(timelines), 3)
        for tl in timelines:
            self.assertEqual(len(tl.sessions), 9)

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_sessions_chronological(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        for tl in timelines:
            timestamps = [s.conversation.timestamp for s in tl.sessions]
            self.assertEqual(timestamps, sorted(timestamps))

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_session_windows(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        for tl in timelines:
            windows = compute_session_windows(tl)
            self.assertGreater(len(windows), 0)
            all_ids = []
            for w in windows:
                all_ids.extend(w.session_ids)
            session_ids = {s.conversation.session_id for s in tl.sessions}
            self.assertEqual(set(all_ids), session_ids)

    def test_gap_days(self):
        t1 = datetime(2026, 1, 5)
        t2 = datetime(2026, 1, 12)
        self.assertAlmostEqual(compute_gap_days(t1, t2), 7.0)

    def test_gap_days_symmetric(self):
        t1 = datetime(2026, 1, 5)
        t2 = datetime(2026, 2, 5)
        self.assertEqual(compute_gap_days(t1, t2), compute_gap_days(t2, t1))

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_co_occurring_signals_arjun(self):
        ds = load_dataset(DATASET_PATH)
        arjun = [u for u in ds.users if u.user_id == "USR001"][0]
        tl = build_user_timeline(arjun)
        cooc = find_co_occurring_signals(tl, "stomach_pain", "late_eating")
        self.assertGreaterEqual(len(cooc), 3)

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_get_sessions_with_signal(self):
        ds = load_dataset(DATASET_PATH)
        meera = [u for u in ds.users if u.user_id == "USR002"][0]
        tl = build_user_timeline(meera)
        dairy_sessions = get_sessions_with_signal_value(tl, "dairy_intake")
        self.assertGreaterEqual(len(dairy_sessions), 3)


class TestDetector(unittest.TestCase):

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_detector_runs(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        detector = PatternDetector()
        patterns, reasoning = detector.detect(timelines)
        self.assertGreater(len(patterns), 0)
        self.assertGreater(len(reasoning), 0)

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_patterns_have_required_fields(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        detector = PatternDetector()
        patterns, _ = detector.detect(timelines)
        for p in patterns:
            self.assertIsInstance(p.pattern_id, str)
            self.assertIsInstance(p.user_id, str)
            self.assertIsInstance(p.title, str)
            self.assertIsInstance(p.evidence_sessions, list)
            self.assertGreater(len(p.evidence_sessions), 0)
            self.assertIsInstance(p.temporal_reasoning, str)
            self.assertIsInstance(p.confidence, Confidence)
            self.assertIsInstance(p.justification, str)
            self.assertIsInstance(p.supporting_signals, list)
            self.assertIsInstance(p.alternative_explanations, list)

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_arjun_stomach_pattern_detected(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        detector = PatternDetector()
        patterns, _ = detector.detect(timelines)
        arjun_patterns = [p for p in patterns if p.user_id == "USR001"]
        stomach_patterns = [
            p for p in arjun_patterns
            if "stomach" in p.title.lower()
        ]
        self.assertGreater(len(stomach_patterns), 0)

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_meera_dairy_pattern_detected(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        detector = PatternDetector()
        patterns, _ = detector.detect(timelines)
        meera_patterns = [p for p in patterns if p.user_id == "USR002"]
        dairy_patterns = [
            p for p in meera_patterns
            if "dairy" in p.title.lower() or "acne" in p.title.lower()
        ]
        self.assertGreater(len(dairy_patterns), 0)

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_priya_cramps_pattern_detected(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        detector = PatternDetector()
        patterns, _ = detector.detect(timelines)
        priya_patterns = [p for p in patterns if p.user_id == "USR003"]
        cramp_patterns = [
            p for p in priya_patterns
            if "cramp" in p.title.lower() or "menstrual" in p.title.lower()
        ]
        self.assertGreater(len(cramp_patterns), 0)


class TestScorer(unittest.TestCase):

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_recalibrate_runs(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        detector = PatternDetector()
        patterns, reasoning = detector.detect(timelines)
        reasoning_len_before = len(reasoning)
        scored_patterns, scored_reasoning = recalibrate(patterns, timelines, reasoning)
        self.assertEqual(len(scored_patterns), len(patterns))
        self.assertGreater(len(scored_reasoning), reasoning_len_before)

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_confidence_values_valid(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        detector = PatternDetector()
        patterns, reasoning = detector.detect(timelines)
        patterns, reasoning = recalibrate(patterns, timelines, reasoning)
        valid = {c for c in Confidence}
        for p in patterns:
            self.assertIn(p.confidence, valid)


class TestOutput(unittest.TestCase):

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_output_is_valid_json(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        detector = PatternDetector()
        patterns, reasoning = detector.detect(timelines)
        patterns, reasoning = recalibrate(patterns, timelines, reasoning)
        output = build_output(patterns, reasoning, timelines)
        output_dict = output.to_dict()
        json_str = json.dumps(output_dict)
        parsed = json.loads(json_str)
        self.assertIn("detected_patterns", parsed)
        self.assertIn("reasoning_trace", parsed)
        self.assertIn("chunking_strategy", parsed)
        self.assertIn("failure_notes", parsed)

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_output_schema_structure(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        detector = PatternDetector()
        patterns, reasoning = detector.detect(timelines)
        patterns, reasoning = recalibrate(patterns, timelines, reasoning)
        output = build_output(patterns, reasoning, timelines)
        d = output.to_dict()
        self.assertIsInstance(d["detected_patterns"], list)
        self.assertIsInstance(d["reasoning_trace"], list)
        self.assertIsInstance(d["chunking_strategy"], dict)
        self.assertIsInstance(d["failure_notes"], list)
        for p in d["detected_patterns"]:
            required_keys = {
                "pattern_id", "user_id", "title", "evidence_sessions",
                "temporal_reasoning", "confidence", "justification",
                "supporting_signals", "alternative_explanations",
            }
            self.assertEqual(required_keys, required_keys & set(p.keys()))
        for r in d["reasoning_trace"]:
            self.assertIn("step", r)
            self.assertIn("action", r)
            self.assertIn("detail", r)


class TestEndToEnd(unittest.TestCase):

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_minimum_pattern_count(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        detector = PatternDetector()
        patterns, reasoning = detector.detect(timelines)
        patterns, reasoning = recalibrate(patterns, timelines, reasoning)
        self.assertGreaterEqual(len(patterns), 6, "Expected at least 6 of 8 hidden patterns")

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_all_users_have_patterns(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        detector = PatternDetector()
        patterns, _ = detector.detect(timelines)
        users_with_patterns = {p.user_id for p in patterns}
        self.assertEqual(users_with_patterns, {"USR001", "USR002", "USR003"})

    @unittest.skipUnless(_dataset_available(), "dataset.json not found")
    def test_no_empty_evidence(self):
        ds = load_dataset(DATASET_PATH)
        timelines = build_all_timelines(ds.users)
        detector = PatternDetector()
        patterns, _ = detector.detect(timelines)
        for p in patterns:
            self.assertGreater(len(p.evidence_sessions), 0, f"{p.pattern_id} has no evidence")
            self.assertGreater(len(p.supporting_signals), 0, f"{p.pattern_id} has no signals")


if __name__ == "__main__":
    unittest.main()