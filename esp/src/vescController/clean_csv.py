#!/usr/bin/env python3
"""Clean ESP log CSV lines into a pure delimiter-separated file.

Usage:
  ./clean_csv.py output.csv output_clean.csv
"""

import argparse
import re
from pathlib import Path


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
CSV_PREFIX_RE = re.compile(r"CSV:\s*(.*)")
TIME_CSV_RE = re.compile(r"E\s*\((\d+)\)\s*CSV:\s*(.*)")


def clean_line(line: str) -> str | None:
    """Extract the timeline and payload after 'CSV:' and strip ANSI escapes."""
    line = ANSI_ESCAPE_RE.sub("", line).strip()
    match = TIME_CSV_RE.search(line)
    if match:
        time_value = match.group(1).strip()
        payload = match.group(2).strip()
        return f"{time_value};{payload}" if payload else None

    match = CSV_PREFIX_RE.search(line)
    if not match:
        return None
    payload = match.group(1).strip()
    return payload if payload else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean CSV log output.")
    parser.add_argument("input", type=Path, help="Input log file (output.csv)")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output cleaned CSV file (default: <input>_clean.csv)",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output or input_path.with_name(
        f"{input_path.stem}_clean{input_path.suffix}"
    )

    cleaned_lines = []
    with input_path.open("r", encoding="utf-8", errors="ignore") as infile:
        for raw_line in infile:
            cleaned = clean_line(raw_line)
            if cleaned is not None:
                cleaned_lines.append(cleaned)

    header = (
        "time;car_heading;bearing_to_goal_north;bearing_to_goal_front_of_car;"
        "latitude;longitude;distance"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        outfile.write(f"{header}\n")
        outfile.write("\n".join(cleaned_lines))
        if cleaned_lines:
            outfile.write("\n")

    print(f"Wrote {len(cleaned_lines)} lines to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
