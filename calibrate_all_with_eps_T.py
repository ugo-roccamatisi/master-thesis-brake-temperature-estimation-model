# =============================================================================
# SANITIZED PUBLIC VERSION
# All calibrated parameter values have been replaced with generic round
# placeholders, and all flight identifiers, timestamps and paths with
# examples. Calibrate on your own data before any quantitative use.
# =============================================================================
# -*- coding: utf-8 -*-
"""
calibrate_all.py -- run core + pre-touchdown calibration for all
combinations of calibration flight × wheel (4 × 4 = 16 cases).

VARIANT (per Ugo, 2026-08-20): two changes to the calibrated core set,
still 7 parameters but a different 7 than the original script's:
  + eps_T added. Previously fixed at the module's own default (0.80,
    per the paper's own sensitivity table -- eps_T was the only
    high-sensitivity parameter never calibrated, with no fitted value
    or independent source).
  - eta_disc REMOVED. No longer fitted -- m.ETA_DISC is simply left
    untouched by this script's own set_core(), at whatever value the
    module itself already carries (still reported in the output JSON
    for reference, just no longer part of the search).
predict_day_from_csv.py's own apply_core()/set_core() only know about
the ORIGINAL 7 (its own NAMES list, C_tube/G_ST/tau/eps_DT/G_RW/h_nat/
eta_disc) -- calling apply_core() with this script's own 7-key core dict
would now raise a KeyError on 'eta_disc' outright (not silently drop
anything, since apply_core() actively reads core['eta_disc']). Every
call site in this script that used to go through core_mod.apply_core()
now calls this script's own set_core() directly instead -- so the
shared module stays untouched and every OTHER script still calibrating
the original 7 keeps working exactly as before.
OTHER script still calibrating the original 7 keeps working exactly as
before.

Each case is cached independently:
  CALIB_DIR/<flight>_core_w<W>.json
  CALIB_DIR/btem_v5_calibration_<flight>_w<W>.json

Existing cache files are reused (skipped) automatically. Note: switching
to this variant on a day/wheel that already has a core cache file from
the ORIGINAL (7-parameter) script will NOT reuse it -- the cache check
compares the cached "names" list to this script's own NAMES, and an
8-element list never matches a 7-element one, so it correctly refits
rather than silently loading a core with no eps_T in it.
A summary table and heatmap figure are shown at the end.

Run: python calibrate_all_with_eps_T.py
"""

import json, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

import btem_v5_full_flight as m
import predict_day_from_csv as core_mod

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR  = r"data"
CALIB_DIR = r"data"

# All flights to calibrate (one per day, first flight of each day)
CAL_FLIGHTS = [
    "flight_0001",   # DAY1
    "flight_0005",   # DAY2
    "flight_0009",   # DAY3
    "flight_0013",   # DAY4
]

WHEELS = [1, 2, 3, 4]

# Calibration weights
ANCHOR_WEIGHT_CORE = 50.0
ANCHOR_WEIGHT_PRE  = 55.0

# Stage-2 bounds
P3_LO = np.array([0.1,  1.0,  10.0])
P3_HI = np.array([10.0, 25.0, 1800.0])   # pulse bound synced to btem_pipeline.py (was 240.0)

# 5 seeds, each a fully self-contained task on ONE shared pool created
# once for the whole run (see main() and _worker_seed_chain below).
N_WORKERS_STAGE2 = min(os.cpu_count() or 4, 5)

# Core parameter names and bounds. eps_T added here (per Ugo, 2026-08-20)
# -- base value 0.80 matches the module's own former fixed default;
# range +-10%, the same relative width already used for eps_DT, capped
# at 1.0 the same way (an emissivity can't physically exceed 1).
# eta_disc REMOVED from calibration (per Ugo, 2026-08-20): no longer
# fitted -- m.ETA_DISC is simply left untouched by set_core() below, at
# whatever value the module itself already carries, the same convention
# eps_T followed before it was added to this variant's own calibrated set.
NAMES = ["C_tube","G_ST","tau","eps_DT","G_RW","h_nat","eps_T"]
BASE  = dict(C_tube=10000.,G_ST=2.5,tau=500/3,eps_DT=.90,
             G_RW=2.,h_nat=8.,eps_T=.80)
FRAC  = dict(C_tube=(.5,1.5),G_ST=(.3/2.5,1.8),tau=(.012,2.5),
             G_RW=(.5,4.),eps_DT=(.9,1.1),h_nat=(1/8,25/8),
             eps_T=(.9,1.1))
LO = np.array([BASE[k]*FRAC[k][0] for k in NAMES])
HI = np.array([min(BASE[k]*FRAC[k][1],1.) if k in ("eps_DT","eps_T")
               else BASE[k]*FRAC[k][1] for k in NAMES])

# =============================================================================
# HELPERS
# =============================================================================
def core_file(flight, wheel):
    # _epsT suffix (per Ugo, 2026-08-20): keeps this variant's cache
    # separate from the original 7-parameter calibrate_all.py's own --
    # same base filename would otherwise flip-flop between a 7- and
    # 8-key core every time you switched between the two scripts (each
    # correctly detects the OTHER's cache as stale and overwrites it).
    return os.path.join(CALIB_DIR, f"{flight}_core_w{wheel}_epsT.json")

def stage2_file(flight, wheel):
    return os.path.join(CALIB_DIR,
                        f"btem_v5_calibration_{flight}_w{wheel}_epsT.json")

def set_core(xp):
    d = dict(zip(NAMES, xp))
    m.C_tube,m.G_ST,m.G_RW,m.eps_DT = (
        d["C_tube"],d["G_ST"],d["G_RW"],d["eps_DT"])
    m.G_sens   = m.C_sens / d["tau"]
    m.h_nat    = d["h_nat"]
    m.eps_T    = d["eps_T"]


def _next_flight_name(flight):
    """flight_0001 -> flight_0002, etc. -- used to find the second anchor
    (per Ugo, 2026-08-20): the flight immediately after the one being
    calibrated."""
    prefix, num = flight.rsplit("_", 1)
    return f"{prefix}_{int(num)+1:04d}"

# =============================================================================
# STAGE 1: CORE CALIBRATION  (one flight, one wheel)
# =============================================================================
def calibrate_core(seg, seg1, flight, wheel):
    """seg1: the NEXT flight's own segment (or None if unavailable), used
    for a SECOND anchor at its first local minimum -- per Ugo,
    2026-08-20. seg and seg1 must come from the SAME segment_csv_inputs()
    call (both files loaded together) so their touchdown_time_s values
    share one absolute time origin -- loading them separately would give
    each its own, unrelated origin, and this anchor's timing math
    silently wrong."""
    cf = core_file(flight, wheel)
    has_anchor2 = seg1 is not None
    if os.path.exists(cf):
        with open(cf) as f: _raw = json.load(f)
        # second_flight_anchor must match this run's OWN availability of
        # seg1 -- a cache written without it (or with it, if seg1 is now
        # missing) is stale for THIS run, not just a different NAMES list.
        if (_raw.get("names") == NAMES
                and _raw.get("second_flight_anchor") == has_anchor2):
            print(f"    core: cached ({_raw.get('written_utc','?')})")
            return {k: float(_raw["core"][k]) for k in NAMES}

    seg["apply_module_state"]()
    T_INIT_TD  = m.btms_at(0.0)
    tm_b, Tm_b = m._btms
    fit_t = tm_b[tm_b >= 0]
    fit_T = Tm_b[tm_b >= 0]
    anchor_t, anchor_T = float(fit_t[-1]), float(fit_T[-1])

    t_nd, P_nd = seg["build_power_trace"](1., None, 1.)
    m._P_interp = interp1d(t_nd, P_nd, bounds_error=False, fill_value=0.)

    # Second anchor: seg1's own first local minimum, converted into
    # seg's own touchdown-relative frame via the shared absolute origin
    # both segments carry (touchdown_time_s). Same "first local minimum"
    # concept already used by fit_exp_gap1()/_fit_exp() for the
    # exponential gap fit -- reused here to also constrain the core fit
    # over a much longer horizon than flight 0's own recording covers.
    anchor2_t = anchor2_T = None
    if has_anchor2:
        pre1 = seg1["tm_b"] < 0
        if pre1.sum() > 0:
            t_min_abs, T_min = core_mod._first_local_minimum(
                seg1["tm_b"][pre1] + seg1["touchdown_time_s"],
                seg1["Tm_b"][pre1])
            anchor2_t = float(t_min_abs - seg["touchdown_time_s"])
            anchor2_T = float(T_min)
            print(f"    second anchor: {anchor2_T:.1f} C at "
                 f"{anchor2_t/3600:.2f} h past this flight's own touchdown "
                 f"(next flight's first local minimum)")
        else:
            has_anchor2 = False
            print("    second anchor: skipped -- next flight has no "
                 "pre-touchdown points")

    def _run_to(t_end_min):
        """Same physics as m.run(), but the parking leg is integrated out
        to at least t_end_min instead of stopping at this flight's own
        t_record() -- needed to reach anchor2_t, which lies hours into
        the NEXT flight's own timeline. Built from m.rhs/m.t_stop/
        m.T_TAXI/m.t_record directly (all public) rather than modifying
        the shared module, since this extended-reach fit is specific to
        this variant."""
        y0 = np.full(14, T_INIT_TD + 273.15)
        kw = dict(method="LSODA", dense_output=True, rtol=1e-6, atol=1e-3)
        t_land = m.t_stop()
        t_taxi_end = t_land + m.T_TAXI
        t_end = max(m.t_record() or 0.0, t_taxi_end + 60.0, t_end_min)
        s1 = solve_ivp(m.rhs, (0, t_land), y0, args=("landing",), max_step=0.5, **kw)
        s2 = solve_ivp(m.rhs, (t_land, t_taxi_end), s1.y[:, -1], args=("taxi",), **kw)
        s3 = solve_ivp(m.rhs, (t_taxi_end, t_end), s2.y[:, -1], args=("parking",), **kw)
        pf = m._PiecewiseFlight([s1, s2, s3],
                                [(0, t_land), (t_land, t_taxi_end), (t_taxi_end, t_end)])
        return pf

    def obj(logx):
        xr = np.exp(logx); xp = np.clip(xr, LO, HI)
        pen = 50. * np.sum(np.abs(np.log(xr) - np.log(xp)))
        set_core(xp)
        try:
            pf = _run_to(anchor2_t if has_anchor2 else 0.0)
            mf = pf.sol(fit_t)[m.I_SENS] - 273.15
            se = (mf - fit_T)**2
            sa = (float(pf.sol([anchor_t])[m.I_SENS][0]) - 273.15 - anchor_T)**2
            anchor_term = ANCHOR_WEIGHT_CORE * sa
            n_anchor = ANCHOR_WEIGHT_CORE
            if has_anchor2:
                sa2 = (float(pf.sol([anchor2_t])[m.I_SENS][0]) - 273.15 - anchor2_T)**2
                anchor_term += ANCHOR_WEIGHT_CORE * sa2
                n_anchor += ANCHOR_WEIGHT_CORE
            return float(np.sqrt(
                (se.sum() + anchor_term) / (len(se) + n_anchor))) + pen
        except Exception:
            return 1e6 + pen

    bx, bv = np.log(np.array([BASE[k] for k in NAMES])), 1e9
    bv = obj(bx)
    for _ in range(4):
        res = minimize(obj, bx, method="Nelder-Mead",
                       options=dict(maxfev=900, xatol=1e-4,
                                    fatol=1e-3, adaptive=True))
        if res.fun < bv - 1e-4:
            bv, bx = res.fun, res.x
        else:
            break

    X1   = np.clip(np.exp(bx), LO, HI)
    CORE = {k: float(v) for k, v in zip(NAMES, X1)}
    set_core(X1)
    t1, Tc1, _ = m.run(T_INIT_TD)
    RMSE = float(np.sqrt(np.mean(
        (np.interp(fit_t, t1, Tc1[m.I_SENS]) - fit_T)**2)))
    print(f"    core: RMSE={RMSE:.2f} C  h_nat={CORE['h_nat']:.2f}")

    with open(cf, "w") as f:
        json.dump(dict(names=NAMES, core=CORE, own_flight_only=not has_anchor2,
                       second_flight_anchor=has_anchor2,
                       wheel=wheel, fans_off=True,
                       alpha_landing_removed=True,
                       written_utc=time.strftime(
                           "%Y-%m-%d %H:%M:%S", time.gmtime())),
                  f, indent=2)
    return CORE

# =============================================================================
# STAGE 2: PRE-TOUCHDOWN CALIBRATION  (one flight, one wheel)
# =============================================================================
# =============================================================================
# STAGE 2 PARALLELISATION: 5 independent seeds, each its own worker process.
# Same pattern as btem_pipeline.py's own run_pretouchdown_calibration --
# see that file's comment for the full reasoning. In short: the 5 seeds
# ([.5, 1., 2., 4., 7.]) don't depend on each other, only the up-to-4
# restarts WITHIN one seed's chain do. seg['build_power_trace'] is a
# CLOSURE over build_segment()'s own locals, so it can't be pickled to a
# worker -- each worker rebuilds its own seg from the CSV path instead.
#
# FULLY SELF-CONTAINED WORKER, NO INITIALIZER (per Ugo, 2026-08-20): an
# earlier version used a ProcessPoolExecutor initializer to load the
# segment ONCE per worker, then scored many seeds against that shared
# state. That meant creating and tearing down a NEW pool once per
# (flight, wheel) combination -- 16 times in this script's own loop, 80
# worker-process spawns total. On Windows this produced sporadic
# FileNotFoundError inside the initializer for a CSV file that had just
# been read successfully moments earlier in the main process -- almost
# certainly a transient file-handle/timing issue from spawning many
# process pools back-to-back in a tight loop, not a real missing file.
# The fix here is structural rather than guessing at the exact Windows
# mechanism: ONE pool is created for the entire run (see main()) instead
# of 16, and each task below is fully self-contained -- it loads its own
# segment fresh rather than relying on a separate initializer call. This
# costs a little redundant CSV re-reading (once per seed instead of once
# per worker), but a single CSV load is cheap next to the ODE solves
# that dominate this fit, and removing the initializer removes the
# specific failure point above entirely.
# =============================================================================
def _worker_seed_chain(args):
    """One full seed for one (flight, wheel): loads its own segment,
    then runs up to 4 sequential Nelder-Mead restarts from that seed.
    Returns (cv, cx, a0) so the caller can pick the best seed, same as
    before."""
    csv_path, wheel_id, core, a0 = args
    segments, _, _ = core_mod.segment_csv_inputs(
        [csv_path], log=lambda *a, **k: None, wheel=wheel_id)
    seg = segments[0]
    seg["apply_module_state"]()
    set_core(np.array([core[k] for k in NAMES]))   # not apply_core(): its own
                                                     # NAMES still expects eta_disc

    T_INIT_FULL = seg["T_INIT_FULL"]
    tm_b, Tm_b = seg["tm_b"], seg["Tm_b"]
    T_START = max(seg["t_move_start"] - 60., float(tm_b[0]))
    T_END = min(seg["t_taxi_accel"] + 1800., -60.)
    w = (tm_b >= T_START) & (tm_b <= T_END)
    tm_w, Tm_w = tm_b[w], Tm_b[w]
    _pre = np.where(tm_b < 0.)[0]
    _ANCHOR_TARGET = -8 * 60.
    _ai = _pre[np.argmin(np.abs(tm_b[_pre] - _ANCHOR_TARGET))]
    anchor_t, anchor_T = float(tm_b[_ai]), float(Tm_b[_ai])
    T_END_ANCHOR = min(anchor_t + 1., -1e-3)

    def score3(xp):
        alpha, hnf, pdur = xp
        m.P_FLIGHT_SCALE = alpha
        m.C_TUBE_FLIGHT_MULT = 1.
        m.H_NAT_FLIGHT = hnf
        tn_p, pn_p = seg["build_power_trace"](1., pdur, 1.)
        saved = m._P_interp
        m._P_interp = interp1d(tn_p, pn_p, bounds_error=False, fill_value=0.)
        try:
            t, Tc = m.run_flight_window(T_INIT_FULL, T_START, T_END_ANCHOR)
            se = (np.interp(tm_w, t, Tc[m.I_SENS]) - Tm_w)**2
            sa = (float(np.interp(anchor_t, t, Tc[m.I_SENS])) - anchor_T)**2
            return float(np.sqrt((se.sum() + ANCHOR_WEIGHT_PRE*sa)
                                 / (len(se) + ANCHOR_WEIGHT_PRE)))
        except Exception:
            return 1e6
        finally:
            m.P_FLIGHT_SCALE = 1.
            m.C_TUBE_FLIGHT_MULT = 1.
            m.H_NAT_FLIGHT = m.h_nat
            m._P_interp = saved

    def obj3(logx):
        return score3(np.clip(_safe_exp_local(logx), P3_LO, P3_HI))

    x0 = np.log(np.array([a0, m.h_nat, 30.]))
    cx, cv = x0, obj3(x0)
    for _ in range(4):
        res = minimize(obj3, cx, method="Nelder-Mead",
                       options=dict(maxfev=300, xatol=1e-3, fatol=1e-2,
                                    adaptive=True))
        if res.fun < cv - 1e-3:
            cv, cx = res.fun, res.x
        else:
            break
    return cv, cx, a0


def _safe_exp_local(x):
    return np.exp(np.clip(x, -50, 50))


def calibrate_pretouchdown(seg, flight, wheel, CORE, path, executor):
    s2f = stage2_file(flight, wheel)
    if os.path.exists(s2f):
        with open(s2f) as f: _raw = json.load(f)
        if _raw.get("core") == CORE:
            print(f"    pre-TD: cached ({_raw.get('written_utc','?')})")
            return _raw

    seg["apply_module_state"]()
    set_core(np.array([CORE[k] for k in NAMES]))   # not apply_core(): its own
                                                     # NAMES still expects eta_disc
    T_INIT_FULL = seg["T_INIT_FULL"]
    tm_b, Tm_b  = seg["tm_b"], seg["Tm_b"]
    T_START = max(seg["t_move_start"] - 60., float(tm_b[0]))
    T_END   = min(seg["t_taxi_accel"] + 1800., -60.)
    w_mask  = (tm_b >= T_START) & (tm_b <= T_END)
    tm_w, Tm_w = tm_b[w_mask], Tm_b[w_mask]
    _pre = np.where(tm_b < 0.)[0]
    if len(_pre) == 0:
        raise RuntimeError(f"{flight} w{wheel}: no pre-TD point for anchor")
    # Anchor at -8 min before touchdown (synced to btem_pipeline.py --
    # was the LAST pre-TD point here before, see the note flagged to Ugo).
    _ANCHOR_TARGET = -8 * 60.
    _ai = _pre[np.argmin(np.abs(tm_b[_pre] - _ANCHOR_TARGET))]
    anchor_t, anchor_T = float(tm_b[_ai]), float(Tm_b[_ai])
    T_END_ANCHOR = min(anchor_t + 1., -1e-3)

    # 5 seeds, each a fully self-contained task on the SHARED pool (see
    # main() -- one pool for the whole run, not one per flight/wheel).
    tasks = [(path, wheel, CORE, a0) for a0 in [0.5, 1.0, 2.0, 4.0, 7.0]]
    seed_results = list(executor.map(_worker_seed_chain, tasks))
    bx, bv = None, np.inf
    for cv, cx, a0 in seed_results:
        if cv < bv - 1e-3:
            bv, bx = cv, cx

    ALPHA, H_NAT, PULSE = tuple(
        float(v) for v in np.clip(np.exp(bx), P3_LO, P3_HI))
    print(f"    pre-TD: alpha={ALPHA:.2f}  H_nat={H_NAT:.2f}  "
          f"pulse={PULSE:.0f}s  RMSE={bv:.2f} C")

    # Full-flight RMSE
    m.P_FLIGHT_SCALE = ALPHA
    m.C_TUBE_FLIGHT_MULT = 1.
    m.H_NAT_FLIGHT = H_NAT
    tn_f, pn_f = seg["build_power_trace"](1., PULSE, 1.)
    m._P_interp = interp1d(tn_f, pn_f, bounds_error=False, fill_value=0.)
    tf, Tcf, _  = m.run_full(T_INIT_FULL)
    RMSE_F = float(np.sqrt(np.mean(
        (np.interp(tm_b, tf, Tcf[m.I_SENS]) - Tm_b)**2)))

    # Touchdown RMSE
    T_INIT_TD = seg["T_INIT_TD"]
    t_td_r, Tc_td_r, _ = m.run(T_INIT_TD)
    e_td = m.btms_error(t_td_r, Tc_td_r[m.I_SENS])

    out = dict(
        flight=flight, wheel=wheel,
        core=CORE, core_source=os.path.basename(core_file(flight, wheel)),
        touchdown_rmse=float(e_td[5]),
        full_flight_rmse=RMSE_F,
        pretouchdown_rmse=float(bv),
        taxi_mode="pulse",
        alpha=ALPHA, c_tube_flight_mult=1., h_nat_flight=H_NAT,
        taxi_pulse_dur_s=float(PULSE),
        eps_T=float(m.eps_T), h_nat=float(m.h_nat),
        ETA_DISC=float(m.ETA_DISC), T_TAXI=float(m.T_TAXI),
        E_landing_MJ_per_brake=float(seg["E_landing_MJ"]),
        source="calibrate_all_with_eps_T.py",
        written_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    with open(s2f, "w") as f:
        json.dump(out, f, indent=2)
    return out

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*74)
    print(f"BATCH CALIBRATION  |  {len(CAL_FLIGHTS)} flights × {len(WHEELS)} wheels "
         f"= {len(CAL_FLIGHTS)*len(WHEELS)} cases")
    print(f"  Stage 2 pool: {N_WORKERS_STAGE2} worker(s), created ONCE for "
         f"the whole run (per Ugo, 2026-08-20 -- creating one per case "
         f"caused sporadic Windows file-access errors under repeated "
         f"pool churn)")
    print("="*74)

    # Summary storage: results[flight][wheel] = dict with RMSE / alpha / pulse
    results = {f: {} for f in CAL_FLIGHTS}

    # Resolve every flight's CSV path FIRST, before the pool exists (per
    # Ugo, 2026-08-20: with the pool created ahead of the loop, spawning
    # 5 fresh Python interpreters -- each re-importing pandas/scipy/numpy
    # -- creates a burst of disk activity right at start-up; every glob()
    # call came back empty when it landed inside that window. Resolving
    # paths first means file discovery never overlaps with pool start-up
    # at all, regardless of the exact Windows-level mechanism).
    print("\nResolving CSV files...")
    print(f"  DATA_DIR = {DATA_DIR}")
    print(f"  DATA_DIR exists: {os.path.isdir(DATA_DIR)}")
    import glob as _glob

    def _resolve(name):
        hits = _glob.glob(os.path.join(DATA_DIR, f"{name}_*.csv"))
        if not hits:
            return None
        clean = [h for h in hits if not any(
            t in os.path.basename(h).lower()
            for t in ("copie", "copy", " - ", "(1)", "(2)"))]
        return (clean if clean else hits)[0]

    flight_paths = {}
    next_flight_paths = {}   # per Ugo, 2026-08-20: for the core fit's 2nd anchor
    for flight in CAL_FLIGHTS:
        p = _resolve(flight)
        if p is None:
            print(f"  {flight}: no CSV found")
            continue
        flight_paths[flight] = p
        print(f"  {flight}: {os.path.basename(p)}")
        next_flight = _next_flight_name(flight)
        p2 = _resolve(next_flight)
        if p2 is not None:
            next_flight_paths[flight] = p2
            print(f"    (+ {next_flight} for the core fit's second anchor: "
                 f"{os.path.basename(p2)})")
        else:
            print(f"    ({next_flight} not found -- core fit for {flight} "
                 f"will use only its own anchor)")

    with ProcessPoolExecutor(max_workers=N_WORKERS_STAGE2) as executor:
        for flight in CAL_FLIGHTS:
            print(f"\n{'─'*74}")
            print(f"  Flight: {flight}")
            print(f"{'─'*74}")

            path = flight_paths.get(flight)
            if path is None:
                for wheel in WHEELS:
                    print(f"\n  wheel {wheel}:\n    SKIP -- no CSV found")
                continue

            for wheel in WHEELS:
                print(f"\n  wheel {wheel}:")

                next_path = next_flight_paths.get(flight)
                paths_for_core = [path] + ([next_path] if next_path else [])
                segs, _, _ = core_mod.segment_csv_inputs(
                    paths_for_core, log=lambda *a: None, wheel=wheel)
                if not segs:
                    print(f"    SKIP -- segmentation failed")
                    continue
                seg = segs[0]
                seg1 = segs[1] if len(segs) > 1 else None

                try:
                    CORE = calibrate_core(seg, seg1, flight, wheel)
                    out = calibrate_pretouchdown(seg, flight, wheel, CORE,
                                                 path, executor)
                    results[flight][wheel] = out
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results[flight][wheel] = None

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print(f"\n{'='*74}")
    print("SUMMARY")
    print(f"{'='*74}")
    hdr = f"  {'flight':<14}" + "".join(
        f"{'w'+str(w)+' α':>8}{'pulse':>7}{'RMSE':>7}" for w in WHEELS)
    print(hdr)
    for flight in CAL_FLIGHTS:
        row = f"  {flight:<14}"
        for wheel in WHEELS:
            r = results[flight].get(wheel)
            if r:
                row += (f"{r['alpha']:>8.2f}"
                        f"{r['taxi_pulse_dur_s']:>7.0f}"
                        f"{r['touchdown_rmse']:>7.1f}")
            else:
                row += f"{'--':>8}{'--':>7}{'--':>7}"
        print(row)

    # =========================================================================
    # FIGURES
    # =========================================================================
    metrics = {
        "alpha":            ("Alpha (P_FLIGHT_SCALE)", "─"),
        "taxi_pulse_dur_s": ("Pulse duration [s]", "─"),
        "touchdown_rmse":   ("Touchdown RMSE [°C]", "─"),
        "full_flight_rmse": ("Full-flight RMSE [°C]", "─"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (key, (title, _)) in zip(axes, metrics.items()):
        mat = np.full((len(CAL_FLIGHTS), len(WHEELS)), np.nan)
        for i, flight in enumerate(CAL_FLIGHTS):
            for j, wheel in enumerate(WHEELS):
                r = results[flight].get(wheel)
                if r:
                    mat[i, j] = r.get(key, np.nan)

        cmap = "RdYlGn_r" if "rmse" in key else "viridis"
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(WHEELS)))
        ax.set_xticklabels([f"Wheel {w}" for w in WHEELS])
        ax.set_yticks(range(len(CAL_FLIGHTS)))
        ax.set_yticklabels([f[-4:] for f in CAL_FLIGHTS])
        ax.set_title(title, fontsize=10)

        vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 1.
        for i in range(len(CAL_FLIGHTS)):
            for j in range(len(WHEELS)):
                if not np.isnan(mat[i, j]):
                    txt_col = ("white" if mat[i,j] > 0.6*vmax
                               else "black")
                    ax.text(j, i, f"{mat[i,j]:.2f}" if key=="alpha"
                            else f"{mat[i,j]:.0f}",
                            ha="center", va="center",
                            fontsize=9, color=txt_col)

    fig.suptitle("Calibration results -- all flights × all wheels",
                 fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()