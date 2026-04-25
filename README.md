# Ask First

Health clarity through temporal reasoning. Detects cross-conversation health patterns
from user chat histories without using an LLM for pattern detection.

## What It Does

Takes 27 synthetic health conversations across 3 users (January–March 2026) and finds
patterns that span multiple sessions: recurring symptoms with consistent triggers,
delayed-onset effects, intervention outcomes, compounding cascades, and isolated
variables. Every pattern is traced to specific sessions and timestamps.

## Project Structure

```
askfirst/
├── engine/
│   ├── __init__.py
│   ├── models.py          # Data models (User, Conversation, Pattern, etc.)
│   ├── loader.py           # Dataset parsing and validation
│   ├── extractor.py        # Signal extraction from conversation text
│   ├── timeline.py         # Timeline construction and temporal utilities
│   ├── detector.py         # Pattern detection (5 strategies)
│   ├── scorer.py           # Confidence recalibration
│   └── output.py           # JSON output assembly
├── llm/
│   ├── __init__.py
│   └── summarizer.py       # Optional LLM summary layer (disabled by default)
├── tests/
│   ├── __init__.py
│   └── test_engine.py      # 35 tests covering all engine stages
├── app.py                  # Streamlit conversational interface
├── main.py                 # CLI entry point
├── dataset.json            # Synthetic dataset
├── results.json            # Engine output
├── requirements.txt
├── writeup.md              # Approach, gaps, and improvements
└── README.md
```

## Setup

Requires Python 3.10 or later.

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/askfirst.git
cd askfirst

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Running the Streamlit App

```bash
streamlit run app.py
```

This opens the conversational interface in your browser. Select a user from the sidebar
and ask questions about their health patterns.

Example queries:
- "What patterns do you see?"
- "Explain the timing"
- "What changed over time?"
- "Why is this pattern high confidence?"
- "Show JSON"
- "P-001"
- "What could the system miss?"

## Running the CLI

```bash
python main.py dataset.json
```

Write output to a file:

```bash
python main.py dataset.json -o results.json
```

Suppress progress messages:

```bash
python main.py dataset.json -o results.json --quiet
```

## Running Tests

```bash
python -m unittest tests.test_engine -v
```

All 35 tests should pass. Tests cover loading, extraction, timeline construction,
detection, scoring, output schema, and end-to-end pattern validation.

## How Detection Works

The engine runs five strategies per user, independently:

1. **Recurring co-occurrence** — Finds symptoms appearing in 3+ sessions with a trigger
   present in most of them. Computes overlap ratios and checks month span.

2. **Delayed sequence** — Finds triggers that precede a new symptom by 14–90 days,
   catching effects like calorie restriction causing hair fall after 6 weeks.

3. **Intervention outcome** — Checks whether applying an intervention correlates with
   improvement in following sessions. Enriches existing patterns with this evidence.

4. **Compounding cascade** — Identifies a single trigger producing multiple distinct
   symptoms appearing in sequence, none of which existed before the trigger.

5. **Variable isolation** — When a symptom has multiple candidate triggers, compares
   consistency rates to isolate which trigger is the independent driver.

After detection, deduplication removes patterns with 80%+ session overlap and identical
symptom sets. Confidence recalibration adjusts scores based on evidence breadth, month
span, and intervention confirmation.

## Confidence Levels

| Level | Meaning |
|-------|---------|
| VERY HIGH | 3+ evidence sessions, 2+ months, high co-occurrence ratio |
| HIGH | 3+ sessions or strong intervention confirmation |
| MEDIUM | 2+ sessions with moderate co-occurrence |
| LOW | Thin evidence or single-month span |
| VERY LOW | Weak signal, flagged for transparency |

## LLM Layer

The optional LLM layer in `llm/summarizer.py` is **disabled by default**. It is never
used for pattern detection. If enabled via environment variables, it only rewrites
engine-generated explanations into more conversational language.

To enable (optional):

```bash
export ASKFIRST_LLM_ENABLED=true
export ASKFIRST_LLM_PROVIDER=openai
export ASKFIRST_LLM_API_KEY=your-key-here
```

## Output Schema

Each detected pattern follows this structure:

```json
{
  "pattern_id": "P-001",
  "user_id": "USR001",
  "short_title": "Recurring stomach pain correlated with late night eating",
  "evidence_sessions": ["USR001_S01", "USR001_S04", "USR001_S07", "USR001_S09"],
  "temporal_reasoning": "Stomach pain appeared in 4 sessions...",
  "confidence": "very high",
  "justification": "4 evidence sessions across 3 months...",
  "supporting_signals": [],
  "possible_alternative_explanations": []
}
```

## Deployment

The app is deployed on Streamlit Community Cloud.

**Live link:** https://ask-first-lkkxjszhlnqhv5fb9got47.streamlit.app/

To deploy your own instance:

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository, branch `main`, and file `app.py`
5. Deploy

Ensure `dataset.json` is committed to the repository. No environment variables are
required for the core engine. The LLM layer environment variables are optional.

## Limitations

- Signal extraction uses phrase matching against a fixed taxonomy. Novel phrasing is missed.
- The system finds correlations, not causation. Alternative explanations are provided.
- Negation handling is limited to known phrases per trigger.
- With 9 sessions per user, some patterns have thin evidence.
- The detector may over-detect for users with many interacting symptoms.

See `writeup.md` for detailed failure analysis and improvement suggestions.
