# BTEM: lumped-parameter brake temperature model (sanitized public release)

A 14-node lumped-parameter thermal model of an aircraft wheel and brake assembly, with a full calibration and prediction pipeline: automatic flight segmentation from CSV, robust touchdown detection, kinematic braking-energy reconstruction, multi-start Nelder-Mead calibration, full-day prediction with a sensitivity envelope, as a command line tool and a Streamlit app.

Developed within my Master's thesis with Airbus and Cranfield University (*Physics-Based Lumped-Parameter Thermal Modelling of Aircraft Wheel Assembly to Estimate Brake Temperature on an In-Service A320*, supervised by Dr Fakhre Ali). **This is a sanitized release**: every calibrated parameter value has been replaced by a generic placeholder, and every flight identifier, timestamp and path by an example. No flight data, no fitted values, no derived quantities from confidential data are included. The pipeline is designed to calibrate itself on whatever data you feed it.

## Layout

| File | Role |
|---|---|
| `btem_v5_full_flight.py` | The 14-node thermal model and full-flight simulation |
| `btem_pipeline_with_eps_T.py` | Calibration pipeline (single source of truth) |
| `predict_day_from_csv.py` | Generalist CLI: segmentation, calibration, full-day prediction |
| `app.py` | Streamlit application on top of the pipeline |
| `calibrate_all_with_eps_T.py` | Batch calibration over flights and wheels |
| `compare_landing_taxi_energy.py` | Kinematic energy reconstruction vs torque comparison |

## Run it

```bash
pip install numpy scipy matplotlib pandas streamlit
streamlit run app.py
```

Feed it a flight CSV (ground speed, radio altitude, brake temperature, gross weight); the pipeline segments, calibrates and predicts.

## Companion project

The physics-informed neural network counterpart, on synthetic data: [pinn-brake-stack](https://github.com/ugo-roccamatisi/pinn-brake-stack). More on my [portfolio](https://ugo-roccamatisi.github.io).
