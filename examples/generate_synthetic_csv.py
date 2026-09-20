"""Generate a small, non-scientific CSV matching the BTEM input schema.

The signal is intended only to exercise loading, segmentation and the user
interface without distributing flight data. It is not an aircraft model and
must not be used to evaluate prediction accuracy.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path


FIELDNAMES = [
    "Timestamp",
    "GROUND SPEED",
    "GROSS WEIGHT",
    "Static Air Temperature [BNR]",
    "Temperature Brake n°1",
    "Temperature Brake n°2",
    "Temperature Brake n°3",
    "Temperature Brake n°4",
    "LH L/G COMPRESSED",
    "Thrust_Reverser_Cowl_Position",
]


def flight_state(minutes: float) -> tuple[float, int, float]:
    """Return ground speed [kt], gear-compressed flag, and brake heat rise."""
    if minutes < 25:  # initial parking
        return 0.0, 1, 0.0
    if minutes < 35:  # taxi out
        phase = minutes - 25
        return 12.0 + 3.0 * math.sin(phase), 1, 8.0 * phase / 10.0
    if minutes < 40:  # take-off roll
        phase = (minutes - 35) / 5
        return 145.0 * phase, 1 if phase < 0.35 else 0, 8.0
    if minutes < 75:  # airborne
        return 155.0, 0, 5.0
    if minutes < 80:  # landing roll
        phase = (minutes - 75) / 5
        return 145.0 * (1.0 - phase), 1, 5.0 + 105.0 * phase
    if minutes < 92:  # taxi in
        phase = minutes - 80
        return max(0.0, 14.0 - phase), 1, 110.0 - 2.5 * phase
    return 0.0, 1, max(0.0, 80.0 * math.exp(-(minutes - 92) / 28.0))


def generate(path: Path, sample_seconds: int = 30) -> None:
    start = datetime(2026, 1, 15, 6, 0, 0)
    duration_minutes = 135
    rows = []
    for elapsed_s in range(0, duration_minutes * 60 + 1, sample_seconds):
        minutes = elapsed_s / 60.0
        speed, compressed, heat_rise = flight_state(minutes)
        ambient = 12.0 + 1.5 * math.sin(minutes / 30.0)
        base_temperature = ambient + 8.0 + heat_rise
        timestamp = (start + timedelta(seconds=elapsed_s)).strftime(
            "%Y/%m/%d %H:%M:%S.%f"
        )[:-3] + "T"
        rows.append(
            {
                "Timestamp": timestamp,
                "GROUND SPEED": f"{speed:.3f}",
                "GROSS WEIGHT": "64000",
                "Static Air Temperature [BNR]": f"{ambient:.3f}",
                "Temperature Brake n°1": f"{base_temperature:.3f}",
                "Temperature Brake n°2": f"{base_temperature + 1.2:.3f}",
                "Temperature Brake n°3": f"{base_temperature - 0.8:.3f}",
                "Temperature Brake n°4": f"{base_temperature + 0.4:.3f}",
                "LH L/G COMPRESSED": str(compressed),
                "Thrust_Reverser_Cowl_Position": (
                    "1" if 75 <= minutes < 77 else "0"
                ),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("synthetic_day.csv"),
    )
    parser.add_argument("--sample-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.sample_seconds <= 0:
        parser.error("--sample-seconds must be positive")
    generate(args.output, args.sample_seconds)


if __name__ == "__main__":
    main()
