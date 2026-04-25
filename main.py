from __future__ import annotations
import argparse
import sys
import time

from engine.loader import load_dataset
from engine.timeline import build_all_timelines
from engine.detector import PatternDetector
from engine.scorer import recalibrate
from engine.output import (
    build_output,
    emit_json,
    stream_pattern_found,
    stream_progress,
    write_json_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="askfirst",
        description="Ask First — Cross-conversation health pattern detection with temporal reasoning",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the dataset JSON file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Write JSON output to this file instead of stdout",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (default: 2)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages on stderr",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    quiet = args.quiet

    def progress(msg: str) -> None:
        if not quiet:
            stream_progress(msg)

    start_time = time.monotonic()

    progress("Loading dataset...")
    dataset = load_dataset(args.dataset)
    progress(
        f"Loaded {len(dataset.users)} users, "
        f"{sum(len(u.conversations) for u in dataset.users)} conversations"
    )

    progress("Building user timelines and extracting signals...")
    timelines = build_all_timelines(dataset.users)
    for tl in timelines:
        signal_count = sum(len(s.signals) for s in tl.sessions)
        progress(
            f"  {tl.user.name} ({tl.user.user_id}): "
            f"{len(tl.sessions)} sessions, {signal_count} signals extracted"
        )

    progress("Running pattern detection...")
    detector = PatternDetector()
    patterns, reasoning = detector.detect(timelines)
    progress(f"Detection complete: {len(patterns)} candidate patterns found")

    if not quiet:
        for p in patterns:
            stream_pattern_found(p)

    progress("Recalibrating confidence scores...")
    patterns, reasoning = recalibrate(patterns, timelines, reasoning)
    progress("Confidence recalibration complete")

    if not quiet:
        progress("Final patterns after scoring:")
        for p in patterns:
            stream_pattern_found(p)

    progress("Assembling output...")
    output = build_output(patterns, reasoning, timelines)

    elapsed = time.monotonic() - start_time
    progress(f"Done in {elapsed:.2f}s — {len(output.detected_patterns)} patterns, "
             f"{len(output.reasoning_trace)} reasoning steps")

    if args.output:
        write_json_file(output, args.output, indent=args.indent)
        progress(f"JSON written to {args.output}")
    else:
        emit_json(output, stream=sys.stdout, indent=args.indent)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()