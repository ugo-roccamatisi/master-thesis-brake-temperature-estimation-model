# =============================================================================
# SANITIZED PUBLIC VERSION
# All calibrated parameter values have been replaced with generic round
# placeholders, and all flight identifiers, timestamps and paths with
# examples. Calibrate on your own data before any quantitative use.
# =============================================================================
# -*- coding: utf-8 -*-
"""
Illustrates the first calibrated flight's own temperature curve,
extended past its own recording to show how far the SAME cooling
physics (governed by the calibrated core, eps_T included) reaches into
the second flight's own pre-touchdown approach -- specifically, up to
and past the second anchor: the second flight's own first local
minimum, the same point used to constrain the core fit in
calibrate_all_with_eps_T.py / btem_pipeline_with_eps_T.py.

Loads a cached core if one exists (same _epsT-suffixed cache filename
both of those scripts write, so either one's output works here
unchanged); otherwise stops with a clear message rather than
re-implementing the fit a third time -- run one of those two scripts
first if no cache is found.
"""

import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

import btem_v5_full_flight as m
import predict_day_from_csv as core_mod

# --- configuration -----------------------------------------------------------
DATA_DIR   = r"data"
CALIB_DIR  = r"data"
FLIGHT_CAL = "flight_0001"   # <-- the calibrated flight to illustrate
WHEEL_ID   = 1               # <-- double-check against the wheel you want
MARGIN_S   = 600.0           # how far past the 2nd anchor to keep plotting

NAMES = ["C_tube", "G_ST", "tau", "eps_DT", "G_RW", "h_nat", "eta_disc", "eps_T"]


def find_csv(flight):
    hits = glob.glob(os.path.join(DATA_DIR, f"{flight}_*.csv"))
    if not hits:
        raise SystemExit(f"{flight}: no CSV found in {DATA_DIR}")
    clean = [h for h in hits if not any(
        t in os.path.basename(h).lower()
        for t in ("copie", "copy", " - ", "(1)", "(2)"))]
    return (clean if clean else hits)[0]


def _next_flight_name(flight):
    prefix, num = flight.rsplit("_", 1)
    return f"{prefix}_{int(num)+1:04d}"


NEXT_FLIGHT = _next_flight_name(FLIGHT_CAL)
CORE_FILE = os.path.join(CALIB_DIR, f"{FLIGHT_CAL}_core_w{WHEEL_ID}_epsT.json")

print("=" * 74)
print(f"Illustrating {FLIGHT_CAL} -> {NEXT_FLIGHT}, wheel {WHEEL_ID}")
print("=" * 74)

if not os.path.exists(CORE_FILE):
    raise SystemExit(
        f"No cached core found at {CORE_FILE}.\n"
        f"Run calibrate_all_with_eps_T.py or btem_pipeline_with_eps_T.py "
        f"for {FLIGHT_CAL} first -- this script only visualises an "
        f"existing calibration, it doesn't refit one.")
with open(CORE_FILE) as f:
    _raw = json.load(f)
CORE = {k: float(_raw["core"][k]) for k in NAMES}
print(f"  Using cached core: {os.path.basename(CORE_FILE)} "
     f"({_raw.get('written_utc', '?')})")
print("  " + "  ".join(f"{k}={CORE[k]:.4g}" for k in NAMES))

paths = [find_csv(FLIGHT_CAL), find_csv(NEXT_FLIGHT)]
segments, _, _ = core_mod.segment_csv_inputs(paths, wheel=WHEEL_ID)
if len(segments) < 2:
    raise SystemExit(f"Only found {len(segments)} segment(s) from these "
                     f"two files -- need both {FLIGHT_CAL} and {NEXT_FLIGHT}.")
seg, seg1 = segments[0], segments[1]


def set_core(d):
    m.C_tube, m.G_ST, m.G_RW, m.eps_DT = d["C_tube"], d["G_ST"], d["G_RW"], d["eps_DT"]
    m.G_sens = m.C_sens / d["tau"]
    m.h_nat = d["h_nat"]
    m.ETA_DISC = d["eta_disc"]
    m.eps_T = d["eps_T"]


seg["apply_module_state"]()
set_core(CORE)
T_INIT_TD = m.btms_at(0.0)
tm_b, Tm_b = m._btms
fit_t, fit_T = tm_b[tm_b >= 0], Tm_b[tm_b >= 0]
anchor_t, anchor_T = float(fit_t[-1]), float(fit_T[-1])

t_nd, P_nd = seg["build_power_trace"](1., None, 1.)
m._P_interp = interp1d(t_nd, P_nd, bounds_error=False, fill_value=0.)

# --- second anchor: seg1's own first local minimum --------------------------
pre1 = seg1["tm_b"] < 0
if pre1.sum() == 0:
    raise SystemExit(f"{NEXT_FLIGHT} has no pre-touchdown points -- "
                     f"can't locate a second anchor.")
t_min_abs, T_min = core_mod._first_local_minimum(
    seg1["tm_b"][pre1] + seg1["touchdown_time_s"], seg1["Tm_b"][pre1])
anchor2_t = float(t_min_abs - seg["touchdown_time_s"])
anchor2_T = float(T_min)
print(f"  Second anchor: {anchor2_T:.1f} C at {anchor2_t/3600:.2f} h "
     f"({NEXT_FLIGHT}'s own first local minimum)")

# --- extend the parking-phase integration past this flight's own -----------
# recording, all the way past the second anchor (same method as the
# calibration itself -- see calibrate_all_with_eps_T.py's _run_to()).
y0 = np.full(14, T_INIT_TD + 273.15)
kw = dict(method="LSODA", dense_output=True, rtol=1e-6, atol=1e-3)
t_land = m.t_stop()
t_taxi_end = t_land + m.T_TAXI
t_end = max(m.t_record() or 0.0, t_taxi_end + 60.0, anchor2_t + MARGIN_S)
s1 = solve_ivp(m.rhs, (0, t_land), y0, args=("landing",), max_step=0.5, **kw)
s2 = solve_ivp(m.rhs, (t_land, t_taxi_end), s1.y[:, -1], args=("taxi",), **kw)
s3 = solve_ivp(m.rhs, (t_taxi_end, t_end), s2.y[:, -1], args=("parking",), **kw)
pf = m._PiecewiseFlight([s1, s2, s3],
                        [(0, t_land), (t_land, t_taxi_end), (t_taxi_end, t_end)])

t_plot = np.linspace(0, t_end, 3000)
T_plot = pf.sol(t_plot)[m.I_SENS] - 273.15

rmse_td = float(np.sqrt(np.mean((np.interp(fit_t, t_plot, T_plot) - fit_T)**2)))
model_at_anchor2 = float(pf.sol([anchor2_t])[m.I_SENS][0]) - 273.15
print(f"  Touchdown-window RMSE : {rmse_td:.2f} C")
print(f"  Model at 2nd anchor   : {model_at_anchor2:.1f} C "
     f"(measured {anchor2_T:.1f} C, error {model_at_anchor2 - anchor2_T:+.1f} C)")

# =============================================================================
# FIGURE
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(tm_b / 3600, Tm_b, "r.", ms=3, alpha=0.6,
       label=f"{FLIGHT_CAL} measured")
ax.plot(t_plot / 3600, T_plot, "g-", lw=1.6,
       label=f"{FLIGHT_CAL} model (extended)")

t1_abs_plot = (seg1["tm_b"][pre1] + seg1["touchdown_time_s"]
              - seg["touchdown_time_s"])
ax.plot(t1_abs_plot / 3600, seg1["Tm_b"][pre1], "b.", ms=3, alpha=0.6,
       label=f"{NEXT_FLIGHT} measured (pre-touchdown)")

ax.axvline(0, color="firebrick", ls="--", lw=1.2, label=f"{FLIGHT_CAL} touchdown")
ax.plot(anchor_t / 3600, anchor_T, "k*", ms=13, zorder=5,
       label=f"anchor 1 ({FLIGHT_CAL}'s own last point)")
ax.plot(anchor2_t / 3600, anchor2_T, "k^", ms=10, zorder=5,
       label=f"anchor 2 ({NEXT_FLIGHT}'s first local min.)")

ax.set(xlabel=f"Time [h] (0 = {FLIGHT_CAL} touchdown)", ylabel="BTMS [deg C]",
      title=f"{FLIGHT_CAL} wheel {WHEEL_ID} -- core calibration extended "
            f"into {NEXT_FLIGHT}\nRMSE (touchdown window) = {rmse_td:.2f} C, "
            f"2nd-anchor error = {model_at_anchor2 - anchor2_T:+.1f} C")
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
