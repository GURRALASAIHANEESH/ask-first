# Ask First — Writeup

## What It Does

Ask First takes 27 health conversations across 3 users spanning January–March 2026 and finds
patterns that no single conversation reveals on its own. It looks at what symptoms repeat, what
triggers co-occur with them, whether interventions helped, and whether stopping those interventions
brought symptoms back. Everything runs deterministically — no LLM decides what counts as a pattern.

## How the Engine Works

The pipeline has five stages:

1. **Loading** (`loader.py`) — Parses the dataset, validates structure, builds typed objects.

2. **Extraction** (`extractor.py`) — Scans user messages, followups, assistant responses, and tags
   for symptoms, triggers, interventions, improvements, and worsenings. Uses phrase matching against
   a fixed taxonomy. Each signal is tagged with its session ID and timestamp.

3. **Timeline construction** (`timeline.py`) — Sorts each user's sessions chronologically, annotates
   them with extracted signals, and computes gap days between sessions. Groups sessions into
   half-month windows for structured reporting.

4. **Detection** (`detector.py`) — Runs five strategies per user, independently:
   - **Recurring co-occurrence**: Finds symptoms appearing in 3+ sessions with a trigger present
     in most of them. Computes overlap ratios and checks month span.
   - **Delayed sequence**: Finds triggers that precede a new symptom by 14–90 days, catching
     effects that don't show up immediately (e.g., calorie restriction → hair fall after 6 weeks).
   - **Intervention outcome**: Checks whether applying an intervention correlates with improvement
     in a following session, and enriches existing patterns with that evidence.
   - **Compounding cascade**: Identifies a single trigger that produces multiple distinct symptoms
     appearing in sequence over time, none of which existed before the trigger started.
   - **Variable isolation**: When a symptom has multiple candidate triggers, compares their
     consistency rates across sessions to identify which one is the independent driver.

   After all strategies run, deduplication removes patterns with 80%+ session overlap and identical
   symptom sets.

5. **Scoring** (`scorer.py`) — Recalibrates confidence using evidence count, month span,
   intervention confirmation, worsening confirmation, and alternative explanation count. Applies
   ceiling validation: VERY HIGH requires 3+ sessions across 2+ months. Demotes patterns with
   thin evidence.

## Temporal Reasoning — Specifically

The system does not match keywords. It reasons about time in these ways:

- **Recurrence timing**: Arjun's stomach pain appears Jan 5, Jan 28, Feb 23, Mar 19 — all after
  late dinners during deadlines. The detector finds 100% co-occurrence with late eating and 75%
  with work stress across 4 sessions spanning 3 months.

- **Delayed onset**: Meera starts severe calorie restriction on Jan 8. Hair fall first appears
  Feb 19 — a 42-day gap. The delayed sequence strategy catches this because the trigger precedes
  the symptom by more than 14 days, and co-occurrence validates the link.

- **Intervention response**: Priya adds protein to lunch around Feb 20. The following session shows
  improvement. The system enriches the existing lunch crash pattern with this evidence rather than
  creating a duplicate.

- **Variable isolation**: Priya's menstrual cramps occur in January (high stress, normal sleep),
  February (high stress, bad sleep), and March (low stress, bad sleep). The system compares trigger
  consistency and identifies sleep deprivation as present in more cramp episodes than work stress.

## Chunking Strategy

Sessions are grouped into half-month windows (days 1–15, days 16–31) within each calendar month.
This provides temporal structure for the "changes over time" view in the UI.

The detector itself operates on the full user timeline rather than windowed subsets. This is a
deliberate choice: with only 9 sessions per user spanning 3 months, windowed processing would
fragment the evidence. A pattern like Arjun's stomach pain needs all 4 episodes visible
simultaneously to compute the co-occurrence ratio. Windowing the detector would require cross-window
merging logic that adds complexity without improving detection at this data scale.

At larger scale (hundreds of sessions over years), windowed detection with cross-window pattern
linking would become necessary to manage memory and latency. The architecture supports this — each
strategy already processes sessions in chronological order and could be adapted to operate on
rolling windows.

## Where the System Fails

I want to be specific about limitations rather than listing generic caveats.

**Over-detection on USR003 (Priya).** The system produces 9 patterns for Priya against a likely
target of 3. The cascade and intervention strategies generate patterns from thinner evidence when
a user has many interacting symptoms. Tighter thresholds (e.g., requiring 3+ downstream symptoms
for a cascade instead of 2) would reduce this, but risk missing real patterns for users with fewer
sessions.

**Signal extraction is brittle.** The extractor uses phrase matching against a fixed taxonomy. If a
user describes "can't keep my eyes open after lunch" instead of "fatigue," the system misses it.
Metaphorical language, regional phrasing, and indirect descriptions all fall through. This is the
single biggest source of false negatives.

**Negation handling is limited.** The system checks for specific negation phrases (e.g., "stress is
actually low") per trigger. But complex negations like "I wouldn't say it's stress exactly" or
conditional statements like "only when I eat dairy, not otherwise" are not parsed correctly.

**Correlation, not causation.** Every co-occurrence could be coincidental. The system provides
alternative explanations for each pattern, but it cannot determine ground truth. The variable
isolation strategy is the closest it gets to causal reasoning, and even that requires at least 3
sessions to be meaningful.

**No numeric confidence.** Confidence uses an enum (VERY LOW through VERY HIGH) rather than a
continuous score. This was a design choice for interpretability, but it loses granularity. Two
HIGH patterns with different evidence quality are indistinguishable.

**Intervention strategy can over-enrich.** When an intervention session overlaps with an existing
pattern's evidence, the system enriches that pattern regardless of whether the intervention is
relevant to the symptom. For Arjun, a "water increase" intervention gets attached to the stomach
pain pattern even though the water was for headaches.

## What I Would Improve With More Time

1. **Tighten Priya's pattern count.** Add minimum evidence thresholds per strategy type. Cascade
   should require 3+ downstream symptoms. Variable isolation should require 4+ sessions. This
   reduces noise for users with complex, interacting symptoms.

2. **Add fuzzy signal extraction.** Use embedding similarity instead of exact phrase matching to
   catch paraphrased symptoms. This would require a small model (not an LLM for pattern detection,
   just for signal extraction).

3. **Numeric confidence scores.** Replace the enum with a 0.0–1.0 score computed from co-occurrence
   ratio, evidence count, month span, and intervention confirmation as weighted factors. Keep the
   categorical label as a derived property.

4. **Smarter intervention linking.** Before enriching a pattern with an intervention outcome, verify
   that the intervention is semantically related to the pattern's symptom. "Water increase" should
   enrich the headache pattern, not the stomach pattern.

5. **User-facing reasoning trace.** The reasoning trace exists in the JSON output but is not
   surfaced in the Streamlit chat. Adding a "show me your reasoning" intent that walks through
   the detection steps would make the system more transparent.

6. **Feedback loop.** Let users confirm or reject detected patterns. Over time, this builds a
   validation set that the confidence scorer can learn from.

## Tech Stack

- Python 3.10+
- Streamlit for the conversational UI
- No external LLM — all detection is algorithmic
- No database — dataset loaded from JSON at startup
- No ML models — pattern detection is rule-based with temporal heuristics