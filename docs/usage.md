# BTEM usage guide

The Brake Temperature Estimation Model predicts the BTMS sensor temperature
across one or more flights, reconstructing braking energy from recorded speed
and weight before running a 14-node thermal model.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Input CSV

Provide one whole-day CSV or several per-flight CSVs. The separator is detected
automatically. Timestamps determine chronology and real parking gaps.

Required columns:

| Column | Meaning |
|---|---|
| `Timestamp` | `YYYY/MM/DD HH:MM:SS.fffT` |
| `GROUND SPEED` | Ground speed in knots |
| `GROSS WEIGHT` | Aircraft mass in kg or lb |
| `Static Air Temperature [BNR]` | Outside-air temperature in °C |
| `Temperature Brake n°1` … `n°4` | Measured BTMS temperature for each braked wheel |
| `LH L/G COMPRESSED` or `RH L/G COMPRESSED` | Non-zero when the landing gear is compressed |

Optional: `Thrust_Reverser_Cowl_Position`. If absent, reverse thrust contributes
nothing to the landing-roll force balance.

Column names are matched exactly after surrounding whitespace is removed.
Sampling does not need to be regular; missing values are filtered channel by
channel.

## Non-confidential format example

```bash
python examples/generate_synthetic_csv.py
```

This writes `examples/synthetic_day.csv`. The generated signal is an ingestion
and interface demonstration only. It is not representative aircraft data and
must not be used to judge model accuracy.

## Streamlit application

```bash
streamlit run app.py
```

1. Upload one or more CSV files.
2. Inspect the segmentation plot before calibration.
3. Select flights, wheel and calibration source.
4. Run the prediction and export the resulting series or calibration bundle.

## Command line

```bash
python predict_day_from_csv.py --csv day.csv --calibration fit-first
python predict_day_from_csv.py --csv flight_a.csv flight_b.csv \
    --flights 1-2 --gap-model exp_gap1 --envelope --out-csv result.csv
```

Important options:

- `--flights`: `all`, `2`, `2-4`, or `1,3,4`;
- `--calibration`: fit the first flight or load an existing bundle;
- `--min-park-gap-min`: minimum parked duration defining a flight boundary;
- `--gap-model`: natural-convection simulation or fitted exponential cooling;
- `--check-segmentation`: display the detected flight split;
- `--no-cache`: ignore cached calibration results;
- `--out-csv`: export measured and predicted temperatures.

Run `python predict_day_from_csv.py --help` for the complete option list.
The original full operating reference, including detailed model behaviour and
troubleshooting notes, is preserved in [usage-reference.txt](usage-reference.txt).

## Operating assumptions and limitations

- Segmentation is a heuristic based on ground speed. Always inspect it before
  calibration.
- The landing-gear discrete controls airborne/on-ground thermal regimes and is
  not interchangeable with ground speed.
- Without radio altitude, the approach window uses a fixed-duration fallback.
- Public parameter values and uncertainty settings are generic placeholders.
  Calibrate on your own data before quantitative use.
- No Airbus flight data, fitted parameter set, timestamp or flight identifier
  is included in this release.
