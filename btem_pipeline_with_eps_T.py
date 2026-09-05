# =============================================================================
# SANITIZED PUBLIC VERSION
# All calibrated parameter values have been replaced with generic round
# placeholders, and all flight identifiers, timestamps and paths with
# examples. Calibrate on your own data before any quantitative use.
# =============================================================================
# -*- coding: utf-8 -*-
"""
btem_pipeline.py -- full BTEM v5 pipeline, one day / one wheel.

VARIANT (per Ugo, sessions of 2026-08-20): accumulated changes on top of
the original 7-parameter, 3-run script --

STAGE 1 (core calibration):
  - Calibrated set is still 7 parameters, but a DIFFERENT 7 than the
    original: eps_T added (previously fixed at the module's own default,
    0.80 -- the paper's own sensitivity table flags it as the only
    high-sensitivity parameter never calibrated); eta_disc REMOVED (no
    longer fitted -- m.eta_disc [sic] is simply left untouched, at
    whatever value the module itself already carries).
  - A SECOND anchor: DAY_FLIGHTS[1]'s own first local minimum (same
    _first_local_minimum() concept fit_exp_gap1()/_fit_exp() already use
    for the exponential gap fit), reached by extending the parking-phase
    integration past its usual m.t_record() cutoff (m.rhs/m.t_stop/
    m.T_TAXI/m._PiecewiseFlight directly, not by modifying run()).

STAGE 2 (pre-touchdown): parallelised across 5 worker processes (one
  per Nelder-Mead seed), otherwise unchanged.

STAGE 3 (day extension), now THREE runs side by side instead of the
  original two:
  - Exponential, LOO-averaged: previously a SEPARATE LOO fit per gap
    (each itself averaged over the other 3 wheels); now ONE (k, T_amb),
    averaged across every gap of the day, applied everywhere -- same
    k_override/T_amb_override mechanism the gap-1 run below already
    used, just fed a day-wide average instead of the first gap alone.
  - Continuous physics: no empirical gap-fill model at all. From
    touchdown, landing -> taxi -> parking run normally; parking carries
    the peak AND the whole gap-cooling tail (the m._t_record override,
    which must be applied AFTER seg['apply_module_state'](), not before
    -- that call resets m._t_record to this flight's own natural length
    on every invocation, silently undoing an override applied earlier).
    T_gap is read straight off that continuous trajectory, right at the
    next flight's own recording start, where that flight's own run_full()
    naturally continues in 'flight' mode from there.
  - Exponential, gap-1 fixed: k/T_amb from the first gap only (_fit_exp
    on WHEEL_ID directly, not LOO), applied everywhere -- unchanged from
    the original script.
  The original standalone "linear" run has been removed entirely.

Every Stage-2/3 call site that used to go through predict_day_from_csv's
own apply_core() now sets the core parameters directly instead: that
shared function reads its own NAMES list (the ORIGINAL 7, eta_disc
included, eps_T not) to build the array it hands to its own set_core(),
so calling it with this script's own 7-key (eps_T, no eta_disc) core
dict would raise a KeyError on 'eta_disc' outright. Direct assignment
sidesteps this without touching the shared module, so every OTHER
script still calibrating the original 7 keeps working unchanged.

Cache filenames get an _epsT suffix so this variant never collides with
the original 7-parameter script's own cache.

SINGLE SOURCE OF TRUTH: all energy reconstruction and module state setup
go through predict_day_from_csv's segment_csv_inputs()/build_segment().
Stages 1, 2, and 3 all use the same seg['apply_module_state']() and
seg['build_power_trace']() -- no parallel reimplementation, no drift.

STAGES:
  1. Core calibration (7 parameters, this variant's own set) on the
     first flight, post-TD window, plus a second anchor from the
     second flight.
     JSON -> CALIB_DIR/<FLIGHT_CAL>_core_w<WHEEL>_epsT.json
  2. Pre-touchdown calibration (alpha, H_nat_flight, pulse_dur).
     JSON -> CALIB_DIR/btem_v5_calibration_<FLIGHT_CAL>_w<WHEEL>_epsT.json
  3. Day extension: exponential LOO-averaged, continuous physics, and
     exponential gap-1 fixed, side by side.

CACHE: if JSONs exist and are consistent, calibration stages are skipped.
CONFIGURATION: edit only the block below.
"""

import glob, json, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy import stats as _stats
from concurrent.futures import ProcessPoolExecutor
import btem_v5_full_flight as m
import predict_day_from_csv as core_mod

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR  = r"data"
CALIB_DIR = r"data"
DAY_FLIGHTS = ["flight_0001", "flight_0002", "flight_0003", "flight_0004"]
WHEEL_ID    = 1

RISE_THRESHOLD_C   = 0.5
CONFIRM_N          = 3
ANCHOR_WEIGHT_CORE = 50.0
ANCHOR_WEIGHT_PRE  = 55.0

FLIGHT_CAL  = DAY_FLIGHTS[0]
W           = WHEEL_ID
# _epsT suffix (per Ugo, 2026-08-20): keeps this variant's cache separate
# from the original 7-parameter script's own -- see the docstring above.
CORE_FILE   = os.path.join(CALIB_DIR, f"{FLIGHT_CAL}_core_w{W}_epsT.json")
STAGE2_FILE = os.path.join(CALIB_DIR,
                           f"btem_v5_calibration_{FLIGHT_CAL}_w{W}_epsT.json")

# Module-level (per Ugo, 2026-08-20): was previously local to
# run_core_calibration() only, but _load_calibration_bundle_epsT() below
# needs it too -- one shared definition instead of a second copy.
# eta_disc REMOVED from calibration (same date): no longer fitted --
# m.ETA_DISC is simply left untouched by this script's own set_core(),
# at whatever value the module itself already carries.
NAMES = ["C_tube","G_ST","tau","eps_DT","G_RW","h_nat","eps_T"]

# STAGE 2 (pre-touchdown) runs 5 independent seeds -- see the note by
# _worker_init_stage2 below for why this needs its own worker processes
# rather than a simple loop-body parallel-for. Capped at 5: there is
# nothing to gain past one worker per seed.
N_WORKERS_STAGE2 = min(os.cpu_count() or 4, 5)

# =============================================================================
# UTILITIES
# =============================================================================
def find_csv(flight):
    hits = glob.glob(os.path.join(DATA_DIR, f"{flight}_*.csv"))
    if not hits:
        raise SystemExit(f"No CSV for {flight} in {DATA_DIR}")
    if len(hits) > 1:
        clean = [h for h in hits if not any(
            t in os.path.basename(h).lower()
            for t in ("copie","copy"," - ","(1)","(2)"))]
        pool = clean if clean else hits
        pool.sort(key=lambda q: os.path.getsize(q), reverse=True)
        return pool[0]
    return hits[0]


def _compute_liftoff_marker(flight, lg_debounce_s=5.0):
    """Own take-off lift-off instant (touchdown-relative, negative) for
    `flight`, computed independently of DISABLE_RETRACTION_WINDOW --
    predict_day_from_csv.py's own m._t_liftoff is unconditionally set to
    None when that flag is on (the project's own default), which would
    otherwise silently hide the marker for every flight. This is purely
    for the pre-touchdown calibration plot: it re-derives the same
    falling-edge-of-LG-COMPRESSED event predict_day_from_csv.py computes
    internally, without touching the module's own physics state. Returns
    None if no landing-gear column is found or no falling edge exists in
    the pre-touchdown window (e.g. recording starts after take-off)."""
    path = find_csv(flight)
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    lg_col = ("LH L/G COMPRESSED" if "LH L/G COMPRESSED" in df.columns
              else "RH L/G COMPRESSED" if "RH L/G COMPRESSED" in df.columns
              else None)
    if lg_col is None:
        return None
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], format="%Y/%m/%d %H:%M:%S.%fT", errors="coerce")
    df = (df.dropna(subset=["Timestamp"])
            .sort_values("Timestamp").reset_index(drop=True))
    t_full = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds().values
    gs     = df["GROUND SPEED"].fillna(0).values
    above  = np.where(gs > 110.0)[0]
    if len(above) == 0:
        return None
    i_td = int(above[-1])
    tm   = t_full - t_full[i_td]

    lg_raw  = df[lg_col].fillna(0).astype(float).values > 0.5
    dt_med  = float(np.nanmedian(np.diff(t_full))) if len(t_full) > 1 else 1.0
    min_n   = max(1, int(round(lg_debounce_s / max(dt_med, 1e-6))))
    lg_deb  = lg_raw.copy()
    i = 0
    while i < len(lg_deb):
        if not lg_deb[i]:
            jj = i
            while jj < len(lg_deb) and not lg_deb[jj]:
                jj += 1
            if (jj - i) < min_n and i > 0 and jj < len(lg_deb):
                lg_deb[i:jj] = True
            i = jj
        else:
            i += 1

    pre   = np.where(tm < 0)[0]
    if len(pre) < 2:
        return None
    og    = lg_deb[pre]
    falls = np.where(og[:-1] & ~og[1:])[0]
    if len(falls) == 0:
        return None
    return float(tm[pre[falls[-1] + 1]])

# =============================================================================
# STAGE 1: CORE CALIBRATION
# =============================================================================
def run_core_calibration(seg, seg1):
    """seg1: DAY_FLIGHTS[1]'s own segment (or None if the day has only one
    flight), used for a SECOND anchor at its first local minimum -- per
    Ugo, 2026-08-20. seg and seg1 come from the SAME segment_csv_inputs()
    call in main() (all of DAY_FLIGHTS loaded together), so their
    touchdown_time_s already share one absolute origin."""
    print("\n" + "="*74)
    print(f"STAGE 1 -- Core calibration  |  {FLIGHT_CAL}  |  wheel {WHEEL_ID}")
    print("="*74)
    has_anchor2 = seg1 is not None
    if os.path.exists(CORE_FILE):
        with open(CORE_FILE) as f: _raw = json.load(f)
        if (_raw.get("names") == NAMES
                and _raw.get("second_flight_anchor") == has_anchor2):
            print(f"  Cached ({_raw.get('written_utc','?')}) -- skipping.")
            CORE = {k: float(_raw["core"][k]) for k in NAMES}
            # Re-display calibration plot from cached result
            seg["apply_module_state"]()
            d=CORE; m.C_tube,m.G_ST,m.G_RW,m.eps_DT=d["C_tube"],d["G_ST"],d["G_RW"],d["eps_DT"]
            m.G_sens=m.C_sens/d["tau"]; m.h_nat=d["h_nat"]
            m.eps_T=d["eps_T"]
            T_INIT_TD=m.btms_at(0.0); tm_b,Tm_b=m._btms
            fit_t=tm_b[tm_b>=0]; fit_T=Tm_b[tm_b>=0]
            t_nd,P_nd=seg["build_power_trace"](1.,None,1.)
            m._P_interp=interp1d(t_nd,P_nd,bounds_error=False,fill_value=0.)
            t1,Tc1,_=m.run(T_INIT_TD)
            RMSE=float(np.sqrt(np.mean((np.interp(fit_t,t1,Tc1[m.I_SENS])-fit_T)**2)))
            fig,ax=plt.subplots(figsize=(11,5))
            ax.plot(tm_b[tm_b>=0]/60,Tm_b[tm_b>=0],"r.",ms=4,
                    label=f"measured -- {FLIGHT_CAL} wheel {WHEEL_ID}")
            ax.plot(t1/60,Tc1[m.I_SENS],"g-",lw=1.8,
                    label=f"core model (RMSE={RMSE:.1f} C)")
            ax.axvline(0,color="firebrick",ls="--",lw=1.4,label="touchdown")
            ax.set(xlabel="Time [min] (0=touchdown)",ylabel="BTMS [degC]",
                   title=f"{FLIGHT_CAL} wheel {WHEEL_ID} -- core calibration (cached)\nRMSE={RMSE:.1f} C")
            ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout()
            plt.show(block=False)
            return CORE, NAMES
    seg["apply_module_state"]()
    T_INIT_TD = m.btms_at(0.0)
    tm_b, Tm_b = m._btms
    fit_t = tm_b[tm_b >= 0]; fit_T = Tm_b[tm_b >= 0]
    anchor_t, anchor_T = float(fit_t[-1]), float(fit_T[-1])
    print(f"  Touchdown : {seg['v_td_kt']:.0f} kt -- {len(fit_t)} post-TD points")
    print(f"  E_landing : {seg['E_landing_MJ']:.4f} MJ/brake (build_segment)")
    print(f"  Anchor    : {anchor_T:.1f} C at {anchor_t/60:.1f} min")

    # Second anchor: seg1's own first local minimum, converted into seg's
    # own touchdown-relative frame via the shared absolute origin both
    # segments carry (touchdown_time_s). Same concept fit_exp_gap1()/
    # _fit_exp() already use for the exponential gap fit.
    anchor2_t = anchor2_T = None
    if has_anchor2:
        pre1 = seg1["tm_b"] < 0
        if pre1.sum() > 0:
            t_min_abs, T_min = core_mod._first_local_minimum(
                seg1["tm_b"][pre1] + seg1["touchdown_time_s"],
                seg1["Tm_b"][pre1])
            anchor2_t = float(t_min_abs - seg["touchdown_time_s"])
            anchor2_T = float(T_min)
            print(f"  Anchor 2  : {anchor2_T:.1f} C at {anchor2_t/3600:.2f} h "
                 f"({DAY_FLIGHTS[1]}'s own first local minimum)")
        else:
            has_anchor2 = False
            print(f"  Anchor 2  : skipped -- {DAY_FLIGHTS[1]} has no "
                 f"pre-touchdown points")

    BASE = dict(C_tube=10000.,G_ST=2.5,tau=500/3,eps_DT=.90,G_RW=2.,
               h_nat=8.,eps_T=.80)
    FRAC = dict(C_tube=(.5,1.5),G_ST=(.3/2.5,1.8),tau=(.012,2.5),G_RW=(.5,4.),
                eps_DT=(.9,1.1),h_nat=(1/8,25/8),eps_T=(.9,1.1))
    lo = np.array([BASE[k]*FRAC[k][0] for k in NAMES])
    hi = np.array([min(BASE[k]*FRAC[k][1],1.) if k in ("eps_DT","eps_T")
                   else BASE[k]*FRAC[k][1] for k in NAMES])
    def set_core(xp):
        d = dict(zip(NAMES,xp))
        m.C_tube,m.G_ST,m.G_RW,m.eps_DT = d["C_tube"],d["G_ST"],d["G_RW"],d["eps_DT"]
        m.G_sens=m.C_sens/d["tau"]; m.h_nat=d["h_nat"]
        m.eps_T=d["eps_T"]
    t_nd,P_nd = seg["build_power_trace"](1.,None,1.)
    m._P_interp = interp1d(t_nd,P_nd,bounds_error=False,fill_value=0.)

    def _run_to(t_end_min):
        """Same physics as m.run(), but the parking leg integrates out to
        at least t_end_min instead of stopping at this flight's own
        t_record() -- needed to reach anchor2_t, which lies hours into
        DAY_FLIGHTS[1]'s own timeline. Built from m.rhs/m.t_stop/
        m.T_TAXI/m.t_record directly (all public) rather than modifying
        the shared module."""
        y0 = np.full(14, T_INIT_TD + 273.15)
        kw = dict(method="LSODA", dense_output=True, rtol=1e-6, atol=1e-3)
        t_land = m.t_stop(); t_taxi_end = t_land + m.T_TAXI
        t_end = max(m.t_record() or 0.0, t_taxi_end + 60.0, t_end_min)
        s1 = solve_ivp(m.rhs, (0, t_land), y0, args=("landing",), max_step=0.5, **kw)
        s2 = solve_ivp(m.rhs, (t_land, t_taxi_end), s1.y[:, -1], args=("taxi",), **kw)
        s3 = solve_ivp(m.rhs, (t_taxi_end, t_end), s2.y[:, -1], args=("parking",), **kw)
        return m._PiecewiseFlight([s1, s2, s3],
                                  [(0, t_land), (t_land, t_taxi_end), (t_taxi_end, t_end)])

    def obj(logx):
        xr=np.exp(logx); xp=np.clip(xr,lo,hi)
        pen=50.*np.sum(np.abs(np.log(xr)-np.log(xp))); set_core(xp)
        try:
            pf = _run_to(anchor2_t if has_anchor2 else 0.0)
            mf = pf.sol(fit_t)[m.I_SENS] - 273.15
            se = (mf-fit_T)**2
            sa = (float(pf.sol([anchor_t])[m.I_SENS][0]) - 273.15 - anchor_T)**2
            anchor_term = ANCHOR_WEIGHT_CORE * sa
            n_anchor = ANCHOR_WEIGHT_CORE
            if has_anchor2:
                sa2 = (float(pf.sol([anchor2_t])[m.I_SENS][0]) - 273.15 - anchor2_T)**2
                anchor_term += ANCHOR_WEIGHT_CORE * sa2
                n_anchor += ANCHOR_WEIGHT_CORE
            return float(np.sqrt((se.sum()+anchor_term)/(len(se)+n_anchor)))+pen
        except Exception: return 1e6+pen
    x0=np.array([BASE[k] for k in NAMES]); bx,bv=np.log(x0),obj(np.log(x0))
    for _ in range(4):
        res=minimize(obj,bx,method="Nelder-Mead",
                     options=dict(maxfev=900,xatol=1e-4,fatol=1e-3,adaptive=True))
        if res.fun<bv-1e-4: bv,bx=res.fun,res.x
        else: break
    X1=np.clip(np.exp(bx),lo,hi); CORE={k:float(v) for k,v in zip(NAMES,X1)}
    set_core(X1); t1,Tc1,_=m.run(T_INIT_TD)
    RMSE=float(np.sqrt(np.mean((np.interp(fit_t,t1,Tc1[m.I_SENS])-fit_T)**2)))
    print(f"  Core RMSE : {RMSE:.2f} C (touchdown window only -- the fit "
         f"itself also scored the second anchor above)")
    for k,v in CORE.items():
        blo,bhi=lo[NAMES.index(k)],hi[NAMES.index(k)]
        flag=" <- AT BOUND" if (v<=blo*1.01 or v>=bhi*.99) else ""
        print(f"    {k:<10} {v:>10.4g}   [{blo:.4g},{bhi:.4g}]{flag}")
    with open(CORE_FILE,"w") as f:
        json.dump(dict(names=NAMES,core=CORE,own_flight_only=not has_anchor2,
                       second_flight_anchor=has_anchor2,wheel=WHEEL_ID,
                       fans_off=True,alpha_landing_removed=True,
                       written_utc=time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime())),f,indent=2)
    print(f"  Written: {os.path.basename(CORE_FILE)}")
    fig,ax=plt.subplots(figsize=(11,5))
    ax.plot(tm_b[tm_b>=0]/60,Tm_b[tm_b>=0],"r.",ms=4,
            label=f"measured -- {FLIGHT_CAL} wheel {WHEEL_ID}")
    ax.plot(t1/60,Tc1[m.I_SENS],"g-",lw=1.8,label=f"core model (RMSE={RMSE:.1f} C)")
    ax.axvline(0,color="firebrick",ls="--",lw=1.4,label="touchdown")
    ax.plot(anchor_t/60,anchor_T,"k*",ms=10,zorder=5,label="anchor")
    ax.set(xlabel="Time [min] (0=touchdown)",ylabel="BTMS [degC]",
           title=f"{FLIGHT_CAL} wheel {WHEEL_ID} -- core calibration\nRMSE={RMSE:.1f} C"
                + (f"  (+ 2nd anchor {anchor2_T:.1f} C @ {anchor2_t/3600:.2f} h, "
                   f"off-plot)" if has_anchor2 else ""))
    ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout()
    plt.show(block=False)
    return CORE, NAMES

# =============================================================================
# STAGE 2: PRE-TOUCHDOWN CALIBRATION
# =============================================================================

def _draw_pretouchdown_figure(tm_b, Tm_b, tf, Tcf, title, events,
                              normalize=True):
    """events: dict with keys 'liftoff','lg_retract','lg_deploy',
    'touchdown','full_stop' -- values in seconds (touchdown-relative),
    None entries are skipped (e.g. liftoff/lg_retract when
    DISABLE_RETRACTION_WINDOW disabled the signal for this flight).

    normalize=True: t/t_total and T/T_max (measured peak), the convention
    used elsewhere in this project. False: minutes from touchdown and degC,
    which is the only view that says how many degrees the error is worth.
    Called twice per figure, once each way."""
    T_MIN_ALL, T_MAX_ALL = float(tm_b.min()), float(tm_b.max())
    T_SPAN = T_MAX_ALL - T_MIN_ALL if T_MAX_ALL > T_MIN_ALL else 1.0
    T_PEAK = float(Tm_b.max())

    if normalize:
        def xn(t):
            return (t - T_MIN_ALL) / T_SPAN

        def yn(T):
            return np.asarray(T, dtype=float) / T_PEAK
        xlab = "t / t_total (0=window start, 1=window end)"
        ylab = "T / T_max"
    else:
        def xn(t):
            return np.asarray(t, dtype=float) / 60.0

        def yn(T):
            return np.asarray(T, dtype=float)
        xlab = "Time [min] (0 = touchdown, negative = before)"
        ylab = "BTMS [\u00b0C]"

    fig, (ax, axe) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1.5]))
    ax.plot(xn(tm_b), yn(Tm_b), "r.", ms=3, alpha=.6,
            label="measured")
    ax.plot(xn(tf), yn(Tcf[m.I_SENS]), "g-", lw=1.6,
            label="model full flight")

    marker_style = dict(
        liftoff    =("firebrick",  "--", "lift-off"),
        lg_retract =("darkorange", "--", "LG retract"),
        lg_deploy  =("goldenrod",  "--", "LG deploy"),
        touchdown  =("navy",       "-",  "touchdown"),
        full_stop  =("purple",     "--", "full stop"),
    )
    for key, (color, ls, label) in marker_style.items():
        t_ev = events.get(key)
        if t_ev is None:
            continue
        ax.axvline(xn(t_ev), color=color, ls=ls, lw=1.4, label=label)
        axe.axvline(xn(t_ev), color=color, ls=ls, lw=1.0)

    # Relative error panel, same convention as stage 3's draw()
    Tp = np.interp(tm_b, tf, Tcf[m.I_SENS])
    vl = np.abs(Tm_b) > 5.
    _metric = ""
    if vl.sum() > 0:
        err_pct = (Tp[vl] - Tm_b[vl]) / Tm_b[vl] * 100.
        axe.plot(xn(tm_b[vl]), err_pct, ".", ms=2.5, color="tab:green", alpha=.7)
        # headline number in the unit this view can be read in: MAPE when the
        # axes are dimensionless, RMSE in degC when they are not
        _metric = (f"  --  MAPE = {np.mean(np.abs(err_pct)):.1f} %" if normalize
                   else f"  --  RMSE = "
                        f"{np.sqrt(np.mean((Tp[vl]-Tm_b[vl])**2)):.1f} \u00b0C")
    ax.set(ylabel=ylab, title=title + _metric)
    ax.grid(alpha=.3)
    axe.axhline(0,   color="k",  lw=.8, ls="--")
    axe.axhline(20,  color=".7", lw=.6, ls=":")
    axe.axhline(-20, color=".7", lw=.6, ls=":")
    axe.set(xlabel=xlab,
            ylabel="Rel. error [%]\n(model-meas)/meas")
    axe.grid(alpha=.3)

    fig.legend(*ax.get_legend_handles_labels(),
              fontsize=8, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 0.04), borderaxespad=0.)
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    plt.show(block=False)


# =============================================================================
# STAGE 2 PARALLELISATION: 5 independent seeds, each its own worker process.
#
# The 5 seeds in run_pretouchdown_calibration's Nelder-Mead search
# ([.5, 1., 2., 4., 7.]) don't depend on each other -- only the up-to-4
# restarts WITHIN one seed's own chain are sequential (each restart
# begins from the previous one's result). That makes the 5 seeds a clean
# unit of parallel work, same idea as Stage 2's own coarse-grid cousin.
#
# WHY THIS NEEDS ITS OWN WORKER-PROCESS SETUP, NOT JUST A PARALLEL LOOP.
# seg['build_power_trace'] -- used inside every scored point -- is a
# CLOSURE returned by build_segment(), capturing that call's own local
# variables (iv, E_iv, t_land_sub, ...). Closures like this cannot be
# pickled to send to a worker process: multiprocessing can only send a
# function BY REFERENCE (module + name), not by value with its captured
# state. So instead of trying to send seg itself to each worker, every
# worker rebuilds its OWN seg from scratch, by calling
# segment_csv_inputs() on just the calibration flight's own CSV --
# _worker_init_stage2() below, run once per worker before it scores
# anything, mirroring exactly what run_pretouchdown_calibration itself
# already does for the main process's own seg.
# =============================================================================
_w2 = {}   # worker-process-local state, set once by _worker_init_stage2

_P3_LO = np.array([.1, 1., 10.])
_P3_HI = np.array([10., 25., 1800.])


def _worker_init_stage2(csv_path, wheel_id, core):
    """Runs once per worker process, before it scores any seed."""
    segments, _, _ = core_mod.segment_csv_inputs(
        [csv_path], log=lambda *a, **k: None, wheel=wheel_id)
    seg = segments[0]
    seg["apply_module_state"]()
    m.C_tube,m.G_ST,m.G_RW,m.eps_DT = core["C_tube"],core["G_ST"],core["G_RW"],core["eps_DT"]
    m.G_sens=m.C_sens/core["tau"]; m.h_nat=core["h_nat"]; m.eps_T=core["eps_T"]
    # not apply_core(): its own NAMES still expects eta_disc, which this
    # script's own 7-key core no longer has (removed, per Ugo, 2026-08-20)

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

    _w2['seg'] = seg
    _w2['T_INIT_FULL'] = T_INIT_FULL
    _w2['T_START'] = T_START
    _w2['T_END_ANCHOR'] = T_END_ANCHOR
    _w2['tm_w'] = tm_w
    _w2['Tm_w'] = Tm_w
    _w2['anchor_t'] = anchor_t
    _w2['anchor_T'] = anchor_T


def _worker_score3(xp):
    """Identical logic to run_pretouchdown_calibration's own score3(),
    reading its fixed inputs from _w2 (set once by _worker_init_stage2)
    instead of from that function's local closure."""
    seg = _w2['seg']
    alpha, hnf, pdur = xp
    m.P_FLIGHT_SCALE = alpha
    m.C_TUBE_FLIGHT_MULT = 1.
    m.H_NAT_FLIGHT = hnf
    tn_p, pn_p = seg["build_power_trace"](1., pdur, 1.)
    saved = m._P_interp
    m._P_interp = interp1d(tn_p, pn_p, bounds_error=False, fill_value=0.)
    try:
        t, Tc = m.run_flight_window(_w2['T_INIT_FULL'], _w2['T_START'],
                                    _w2['T_END_ANCHOR'])
        se = (np.interp(_w2['tm_w'], t, Tc[m.I_SENS]) - _w2['Tm_w'])**2
        sa = (float(np.interp(_w2['anchor_t'], t, Tc[m.I_SENS]))
             - _w2['anchor_T'])**2
        return float(np.sqrt((se.sum() + ANCHOR_WEIGHT_PRE*sa)
                             / (len(se) + ANCHOR_WEIGHT_PRE)))
    except Exception:
        return 1e6
    finally:
        m.P_FLIGHT_SCALE = 1.
        m.C_TUBE_FLIGHT_MULT = 1.
        m.H_NAT_FLIGHT = m.h_nat
        m._P_interp = saved


def _worker_obj3(logx):
    return _worker_score3(np.clip(np.exp(logx), _P3_LO, _P3_HI))


def _worker_seed_chain(a0):
    """One full seed: up to 4 sequential Nelder-Mead restarts -- exactly
    the original inner loop's own logic, just run inside one worker
    process instead of inline in the main process. Returns (best_score,
    best_logx, seed) so the main process can pick the winning seed
    across all 5, same as the original sequential version did."""
    x0 = np.log(np.array([a0, m.h_nat, 30.]))
    cx, cv = x0, _worker_obj3(x0)
    for _ in range(4):
        res = minimize(_worker_obj3, cx, method="Nelder-Mead",
                       options=dict(maxfev=300, xatol=1e-3, fatol=1e-2,
                                    adaptive=True))
        if res.fun < cv - 1e-3:
            cv, cx = res.fun, res.x
        else:
            break
    return cv, cx, a0


def run_pretouchdown_calibration(seg, CORE, NAMES):
    print("\n"+"="*74)
    print(f"STAGE 2 -- Pre-touchdown calibration  |  {FLIGHT_CAL}  |  wheel {WHEEL_ID}")
    print("="*74)
    if os.path.exists(STAGE2_FILE):
        with open(STAGE2_FILE) as f: _raw=json.load(f)
        if _raw.get("core")==CORE:
            print(f"  Cached ({_raw.get('written_utc','?')}) -- skipping.")
            # Re-display pre-touchdown plot from cached result
            seg["apply_module_state"]()
            m.C_tube,m.G_ST,m.G_RW,m.eps_DT = CORE["C_tube"],CORE["G_ST"],CORE["G_RW"],CORE["eps_DT"]
            m.G_sens=m.C_sens/CORE["tau"]; m.h_nat=CORE["h_nat"]; m.eps_T=CORE["eps_T"]
            m.P_FLIGHT_SCALE=_raw["alpha"]; m.C_TUBE_FLIGHT_MULT=1.
            m.H_NAT_FLIGHT=_raw["h_nat_flight"]
            tm_b,Tm_b=seg["tm_b"],seg["Tm_b"]
            _pulse_cached = _raw["taxi_pulse_dur_s"]
            tn_f,pn_f=seg["build_power_trace"](1.,_pulse_cached,1.)
            m._P_interp=interp1d(tn_f,pn_f,bounds_error=False,fill_value=0.)
            tf,Tcf,_=m.run_full(seg["T_INIT_FULL"])
            RMSE_F=float(np.sqrt(np.mean((np.interp(tm_b,tf,Tcf[m.I_SENS])-Tm_b)**2)))
            _liftoff = _compute_liftoff_marker(FLIGHT_CAL)
            _events = dict(
                liftoff=_liftoff,
                lg_retract=(_liftoff + m.T_RETRACT_S) if _liftoff is not None else None,
                lg_deploy=-m.APPROACH_FALLBACK_S,
                touchdown=0.0,
                full_stop=seg["t_full_stop"])
            for _norm in (True, False):     # normalised, then physical units
                _draw_pretouchdown_figure(
                    tm_b, Tm_b, tf, Tcf, normalize=_norm,
                    title=(f"{FLIGHT_CAL} wheel {WHEEL_ID} -- pre-touchdown "
                           f"calibration (cached)\n"
                           f"alpha={_raw['alpha']:.2f}  "
                           f"H_nat={_raw['h_nat_flight']:.2f} W/m2K  "
                           f"pulse={_pulse_cached:.0f}s"),
                    events=_events)
            return _raw
    seg["apply_module_state"]()
    m.C_tube,m.G_ST,m.G_RW,m.eps_DT = CORE["C_tube"],CORE["G_ST"],CORE["G_RW"],CORE["eps_DT"]
    m.G_sens=m.C_sens/CORE["tau"]; m.h_nat=CORE["h_nat"]; m.eps_T=CORE["eps_T"]
    T_INIT_FULL=seg["T_INIT_FULL"]; T_INIT_TD=seg["T_INIT_TD"]
    tm_b,Tm_b=seg["tm_b"],seg["Tm_b"]
    T_START=max(seg["t_move_start"]-60.,float(tm_b[0]))
    T_END=min(seg["t_taxi_accel"]+1800.,-60.)
    w=(tm_b>=T_START)&(tm_b<=T_END); tm_w,Tm_w=tm_b[w],Tm_b[w]
    print(f"  Window    : [{T_START/60:+.1f},{T_END/60:+.1f}] min, {int(w.sum())} points")
    print(f"  E_landing : {seg['E_landing_MJ']:.4f} MJ/brake (build_segment)")
    _pre=np.where(tm_b<0.)[0]
    if len(_pre)==0: raise SystemExit("No pre-TD measured point for anchor.")
    # Anchor at -8 min before touchdown (closest measured point to -480s)
    _ANCHOR_TARGET = -8*60.  # -480 s
    _ai=_pre[np.argmin(np.abs(tm_b[_pre]-_ANCHOR_TARGET))]
    anchor_t,anchor_T=float(tm_b[_ai]),float(Tm_b[_ai])
    T_END_ANCHOR=min(anchor_t+1.,-1e-3)
    print(f"  Anchor    : {anchor_T:.1f} C at {anchor_t/60:+.2f} min (target -8 min)")
    p3_lo=np.array([.1,1.,10.]); p3_hi=np.array([10.,25.,1800.])
    # 5 seeds, each its own worker process -- see the note above
    # _worker_init_stage2 for why each worker rebuilds its own seg
    # rather than receiving this one (build_power_trace is a closure,
    # not picklable). Same scoring, same restarts, same result as the
    # sequential version -- only how the 5 seeds are computed changes.
    with ProcessPoolExecutor(max_workers=N_WORKERS_STAGE2,
                             initializer=_worker_init_stage2,
                             initargs=(find_csv(FLIGHT_CAL), WHEEL_ID, CORE)) as ex:
        seed_results = list(ex.map(_worker_seed_chain, [.5,1.,2.,4.,7.]))
    bx,bv,bseed=None,np.inf,None
    for cv,cx,a0 in seed_results:
        if cv<bv-1e-3: bv,bx,bseed=cv,cx,a0
    ALPHA,H_NAT,PULSE=tuple(float(v) for v in np.clip(np.exp(bx),p3_lo,p3_hi))
    print(f"  alpha={ALPHA:.2f}  H_nat_flight={H_NAT:.2f} W/m2K  pulse_dur={PULSE:.0f}s  RMSE={bv:.2f} C")
    m.P_FLIGHT_SCALE=ALPHA; m.C_TUBE_FLIGHT_MULT=1.; m.H_NAT_FLIGHT=H_NAT
    tn_f,pn_f=seg["build_power_trace"](1.,PULSE,1.)
    m._P_interp=interp1d(tn_f,pn_f,bounds_error=False,fill_value=0.)
    tf,Tcf,_=m.run_full(T_INIT_FULL)
    RMSE_F=float(np.sqrt(np.mean((np.interp(tm_b,tf,Tcf[m.I_SENS])-Tm_b)**2)))
    print(f"  Full-flight RMSE: {RMSE_F:.2f} C")
    t_td_r,Tc_td_r,_=m.run(T_INIT_TD); e_td=m.btms_error(t_td_r,Tc_td_r[m.I_SENS])
    out=dict(flight=FLIGHT_CAL,wheel=WHEEL_ID,core=CORE,
             core_source=os.path.basename(CORE_FILE),
             touchdown_rmse=float(e_td[5]),taxi_mode="pulse",
             alpha=ALPHA,c_tube_flight_mult=1.,h_nat_flight=H_NAT,
             taxi_pulse_dur_s=float(PULSE),
             eps_T=float(m.eps_T),h_nat=float(m.h_nat),ETA_DISC=float(m.ETA_DISC),
             T_TAXI=float(m.T_TAXI),E_landing_MJ_per_brake=float(seg["E_landing_MJ"]),
             source=f"btem_pipeline.py wheel={WHEEL_ID}",
             written_utc=time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()))
    with open(STAGE2_FILE,"w") as f: json.dump(out,f,indent=2)
    print(f"  Written: {os.path.basename(STAGE2_FILE)}")
    _liftoff = _compute_liftoff_marker(FLIGHT_CAL)
    _events = dict(
        liftoff=_liftoff,
        lg_retract=(_liftoff + m.T_RETRACT_S) if _liftoff is not None else None,
        lg_deploy=-m.APPROACH_FALLBACK_S,
        touchdown=0.0,
        full_stop=seg["t_full_stop"])
    for _norm in (True, False):     # normalised, then physical units
        _draw_pretouchdown_figure(
            tm_b, Tm_b, tf, Tcf, normalize=_norm,
            title=(f"{FLIGHT_CAL} wheel {WHEEL_ID} -- pre-touchdown calibration\n"
                   f"alpha={ALPHA:.2f}  H_nat={H_NAT:.2f} W/m2K  "
                   f"pulse={PULSE:.0f}s"),
            events=_events)
    return out

# =============================================================================
# STAGE 3: DAY EXTENSION
# =============================================================================
def _first_local_minimum(t_arr,T_arr):
    o=np.argsort(t_arr); t_s,T_s=t_arr[o],T_arr[o]
    rm,im=T_s[0],0; i=1
    while i<len(T_s):
        if T_s[i]<rm: rm,im=T_s[i],i
        elif T_s[i]>rm+RISE_THRESHOLD_C:
            if i+CONFIRM_N<=len(T_s) and np.all(T_s[i:i+CONFIRM_N]>rm+RISE_THRESHOLD_C*.5):
                return t_s[im],T_s[im]
        i+=1
    return t_s[im],T_s[im]

def _load_raw(flight):
    path=find_csv(flight)
    df=pd.read_csv(path,sep=None,engine="python",encoding="utf-8-sig")
    df.columns=[str(c).strip() for c in df.columns]
    df["Timestamp"]=pd.to_datetime(df["Timestamp"],format="%Y/%m/%d %H:%M:%S.%fT",errors="coerce")
    df=df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    t_full=(df["Timestamp"]-df["Timestamp"].iloc[0]).dt.total_seconds().values
    gs=df["GROUND SPEED"].fillna(0).values
    above=np.where(gs>110.)[0]
    if len(above)==0: raise SystemExit(f"{flight}: no touchdown")
    i_td=int(above[-1]); tm=t_full-t_full[i_td]
    ts=df["Timestamp"].iloc[i_td]
    wheels={}
    for w in (1,2,3,4):
        col=f"Temperature Brake n\u00b0{w}"
        if col not in df.columns: continue
        b=df[col]; ok=b.notna().values
        wheels[w]=dict(tm_b=tm[ok],Tm_b=b[ok].values)
    amb_col="Static Air Temperature [BNR]"
    amb_c=(df[amb_col].ffill().bfill().values if amb_col in df.columns else np.full(len(tm),np.nan))
    return dict(flight=flight,touchdown_ts=ts,wheels=wheels,tm=tm,gspd_kt=gs,amb_c=amb_c,tm_min=float(tm[0]))

def _fit_exp(fN,fNp1,w):
    if w not in fN["wheels"] or w not in fNp1["wheels"]: return None
    dN,dNp1=fN["wheels"][w],fNp1["wheels"][w]
    post=dN["tm_b"]>=0
    if post.sum()==0: return None
    tp,Tp=dN["tm_b"][post],dN["Tm_b"][post]
    ipk=np.argmax(Tp); t_pk,T_pk=float(tp[ipk]),float(Tp[ipk])
    OFF=(fNp1["touchdown_ts"]-fN["touchdown_ts"]).total_seconds()
    pre=dNp1["tm_b"]<0
    if pre.sum()==0: return None
    tin=dNp1["tm_b"][pre]+OFF; Tpre=dNp1["Tm_b"][pre]
    t_min,T_min=_first_local_minimum(tin,Tpre)
    if t_min<=t_pk: return None
    mN=tp>=t_pk
    t_pts=np.concatenate([tp[mN],tin[(tin>t_pk)&(tin<=t_min)]])
    T_pts=np.concatenate([Tp[mN],Tpre[(tin>t_pk)&(tin<=t_min)]])
    o=np.argsort(t_pts); t_pts,T_pts=t_pts[o],T_pts[o]
    aN=fN["amb_c"][fN["tm"]>=t_pk]
    aNp1=fNp1["amb_c"][(fNp1["tm"]+OFF>t_pk)&(fNp1["tm"]+OFF<=t_min)]
    aall=np.concatenate([aN,aNp1]); aall=aall[~np.isnan(aall)]
    if len(aall)==0: return None
    T_amb=float(np.mean(aall)); exc=T_pts-T_amb; vl=exc>0.5
    if vl.sum()<3: return None
    slope,_,r,_,_=_stats.linregress(t_pts[vl]-t_pk,np.log(exc[vl])); k=-slope
    if k<=0: return None
    return dict(t_peak=t_pk,T_peak=T_pk,k=k,T_amb=T_amb,r2=float(r**2))

def _measured_rate(fN,fNp1,w):
    if w not in fN["wheels"] or w not in fNp1["wheels"]: return None
    dN,dNp1=fN["wheels"][w],fNp1["wheels"][w]
    post=dN["tm_b"]>=0
    if post.sum()==0: return None
    tp,Tp=dN["tm_b"][post],dN["Tm_b"][post]
    t_pk,T_pk=float(tp[np.argmax(Tp)]),float(Tp[np.argmax(Tp)])
    OFF=(fNp1["touchdown_ts"]-fN["touchdown_ts"]).total_seconds()
    pre=dNp1["tm_b"]<0
    if pre.sum()==0: return None
    t_min,T_min=_first_local_minimum(dNp1["tm_b"][pre]+OFF,dNp1["Tm_b"][pre])
    dt=(t_min-t_pk)/60.; return (T_pk-T_min)/dt if dt>0 else None

def _loo_rate(raw_flights):
    other=[w for w in (1,2,3,4) if w!=WHEEL_ID]; rates=[]
    for i in range(len(raw_flights)-1):
        for w in other:
            r=_measured_rate(raw_flights[i],raw_flights[i+1],w)
            if r is not None: rates.append(r)
    return float(np.mean(rates)) if len(rates)>=2 else None

def _loo_exp(fN,fNp1,return_fits=False):
    other=[w for w in (1,2,3,4) if w!=WHEEL_ID]
    wheel_fits={w: _fit_exp(fN,fNp1,w) for w in other}
    fits=[f for f in wheel_fits.values() if f is not None]
    if len(fits)<2: return (None,None,{}) if return_fits else None
    k_mean=float(np.mean([f["k"] for f in fits]))
    T_amb_mean=float(np.mean([f["T_amb"] for f in fits]))
    if return_fits: return k_mean, T_amb_mean, wheel_fits
    return k_mean, T_amb_mean

def _takeoff_time(rf):
    gs,tm=rf["gspd_kt"],rf["tm"]
    for i in np.where((gs[:-1]<=100)&(gs[1:]>100))[0]:
        if tm[i]>=0: continue
        return float(tm[i]+(100-gs[i])/(gs[i+1]-gs[i])*(tm[i+1]-tm[i]))
    raise SystemExit(f"{rf['flight']}: take-off not found")

def _first_moving(rf):
    idx=np.where(rf["gspd_kt"]>0.5)[0]
    if len(idx)==0: raise SystemExit(f"{rf['flight']}: never moves")
    return float(rf["tm"][idx[0]])

def run_day_prediction(segments, calibration, raw_flights):
    print("\n"+"="*74)
    print(f"STAGE 3 -- Day extension  |  wheel {WHEEL_ID}")
    print("="*74)
    loo_rate=_loo_rate(raw_flights)
    if loo_rate is None: raise SystemExit("Not enough data for LOO rate.")
    print(f"  LOO rate (linear): {loo_rate:.4f} C/min")

    def predict_one(gap_model, k_override=None, T_amb_override=None):
        t_all,T_all=[],[]
        seg_curves,gap_curves=[],[]
        T_init=t_start=None; _nxtP=None
        for idx,seg in enumerate(segments):
            seg["apply_module_state"]()
            _cc=calibration["core"]
            m.C_tube,m.G_ST,m.G_RW,m.eps_DT = _cc["C_tube"],_cc["G_ST"],_cc["G_RW"],_cc["eps_DT"]
            m.G_sens=m.C_sens/_cc["tau"]; m.h_nat=_cc["h_nat"]; m.eps_T=_cc["eps_T"]
            # _t_record override MUST come after apply_module_state() above,
            # not before (per Ugo, 2026-08-20 -- caught because the gap
            # cooling wasn't showing up at all): apply_module_state() sets
            # m._t_record = tm[-1] (this flight's own natural recording
            # length) internally on every call, via build_segment()'s own
            # closure -- applying the override BEFORE it just got silently
            # wiped the moment apply_module_state() ran, so parking always
            # stopped at the flight's own short recording regardless.
            if idx<len(segments)-1:
                _tnm=_first_moving(raw_flights[idx+1])
                _tgt=_tnm+segments[idx+1]["touchdown_time_s"]-seg["touchdown_time_s"]
                _str=m._t_record; m._t_record=max(m._t_record or 0.,_tgt)
            m.P_FLIGHT_SCALE=calibration["alpha"]; m.C_TUBE_FLIGHT_MULT=1.
            m.H_NAT_FLIGHT=calibration["h_nat_flight"]
            tnd,Pnd=seg["build_power_trace"](1.,calibration["pulse_dur"],1.)
            if _nxtP is not None: m._P_interp=_nxtP; _nxtP=None
            else: m._P_interp=interp1d(tnd,Pnd,bounds_error=False,fill_value=0.)
            if idx==0:
                T_init=seg["T_INIT_FULL"]
                print(f"  [{gap_model}] {seg['label']}: T_INIT_FULL={T_init:.2f} C  E_landing={seg['E_landing_MJ']:.4f} MJ/brake")
            else:
                print(f"  [{gap_model}] {seg['label']}: T_init={T_init:.1f} C")
            _sb=m._btms if t_start is not None else None
            if t_start is not None: m._btms=None
            try: t_loc,Tc,_=m.run_full(T_init,t_start=t_start)
            finally:
                if _sb is not None: m._btms=_sb
            if idx<len(segments)-1: m._t_record=_str
            t_abs=t_loc+seg["touchdown_time_s"]
            t_all.append(t_abs); T_all.append(Tc[m.I_SENS])
            post=t_loc>=0; ipk=int(np.argmax(Tc[m.I_SENS][post]))
            peak_T=float(Tc[m.I_SENS][post][ipk]); peak_abs=float(t_abs[post][ipk])
            if idx<len(segments)-1:
                nxt=segments[idx+1]
                if gap_model=="physics":
                    # Continuous physics (per Ugo, 2026-08-20): no separate
                    # empirical gap-fill model at all. Starting from
                    # touchdown, landing -> taxi -> parking run normally;
                    # parking carries the peak AND the whole gap-cooling
                    # tail (m.run_full() above already reaches this point,
                    # via the m._t_record override extending it all the
                    # way to the next flight's own first-moving instant).
                    # T_gap is read straight off that continuous
                    # trajectory, right at the next flight's own recording
                    # start -- at which point THAT flight's own run_full()
                    # naturally begins in 'flight' mode from there, same
                    # as it always does. No k/T_amb fit, and no separate
                    # gap curve to draw: the segment's own model line
                    # (kept untrimmed below) already IS the gap.
                    rec_start_own = float(raw_flights[idx+1]["tm"][0])
                    rec_start_abs = rec_start_own + nxt["touchdown_time_s"]
                    T_gap = float(np.interp(rec_start_abs, t_abs, Tc[m.I_SENS]))
                    gap_curves.append(dict(t0=peak_abs, T0=peak_T,
                                           t1=rec_start_abs, T1=T_gap,
                                           model="physics", k=None, T_amb=None,
                                           rec_start_abs=rec_start_abs))
                    T_init=T_gap; t_start=rec_start_own
                else:
                    _rf_nxt = raw_flights[idx+1]
                    _rf_cur = raw_flights[idx]
                    _td_offset = (_rf_nxt["touchdown_ts"] - _rf_cur["touchdown_ts"]).total_seconds()
                    rec_start_own_in_N = _rf_nxt["tm"][0] + _td_offset
                    rec_start_abs = rec_start_own_in_N + seg["touchdown_time_s"]
                    rec_start_own = float(_rf_nxt["tm"][0])
                    er=_loo_exp(raw_flights[idx],raw_flights[idx+1],return_fits=True)
                    _ke3,_Ta3,_wfits3=er if er[0] is not None else (None,None,{})
                    er = (_ke3,_Ta3) if _ke3 is not None else None
                    if k_override is not None and T_amb_override is not None:
                        er = (k_override, T_amb_override); _wfits3={}
                    if er:
                        ke,Ta=er
                        _fN   = raw_flights[idx]
                        _fNp1 = raw_flights[idx+1]
                        OFF   = (_fNp1["touchdown_ts"]-_fN["touchdown_ts"]).total_seconds()
                        _w    = WHEEL_ID if WHEEL_ID in _fNp1["wheels"] else next(iter(_fNp1["wheels"]))
                        _pre  = _fNp1["wheels"][_w]["tm_b"] < 0
                        if _pre.sum() > 0:
                            t_min_in_N, _ = _first_local_minimum(
                                _fNp1["wheels"][_w]["tm_b"][_pre] + OFF,
                                _fNp1["wheels"][_w]["Tm_b"][_pre])
                            t1_fig = t_min_in_N + nxt["touchdown_time_s"] - OFF
                        else:
                            t1_fig = rec_start_abs
                        T1_fig = Ta + (peak_T-Ta)*np.exp(-ke*(t1_fig - peak_abs))
                        T_gap  = Ta + (peak_T-Ta)*np.exp(-ke*(rec_start_abs - peak_abs))
                        gap_curves.append(dict(t0=peak_abs, T0=peak_T,
                                               t1=t1_fig,   T1=T1_fig,
                                               model="exponential", k=ke, T_amb=Ta,
                                               rec_start_abs=rec_start_abs,
                                               wheel_fits=_wfits3,
                                               raw_N=raw_flights[idx],
                                               raw_Np1=raw_flights[idx+1],
                                               flight_N=seg["label"],
                                               flight_Np1=nxt["label"]))
                    else:
                        T_gap = peak_T - loo_rate*((rec_start_abs-peak_abs)/60.)
                        gap_curves.append(dict(t0=peak_abs, T0=peak_T,
                                               t1=rec_start_abs, T1=T_gap,
                                               model="linear_fallback", k=None, T_amb=None,
                                               rec_start_abs=rec_start_abs,
                                               flight_N=seg["label"],
                                               flight_Np1=nxt["label"]))
                    T_init=T_gap; t_start=rec_start_own
                if gap_model=="physics":
                    seg_curves.append((t_abs, Tc[m.I_SENS]))   # untrimmed -- IS the gap
                else:
                    _trim = t_abs <= peak_abs
                    seg_curves.append((t_abs[_trim], Tc[m.I_SENS][_trim]))
            else:
                seg_curves.append((t_abs,Tc[m.I_SENS]))
        return np.concatenate(t_all),np.concatenate(T_all),gap_curves,seg_curves

    print("\n  Exponential run (LOO, averaged across all gaps of the day)...")
    # Per Ugo, 2026-08-20: previously a SEPARATE LOO fit per gap (each
    # itself averaged over the other 3 wheels) -- now ONE (k, T_amb),
    # averaged across every gap of the day (still each gap's own LOO-
    # over-wheels fit as the input to that average), applied everywhere.
    # Same override mechanism the gap-1 run already used below, just fed
    # a day-wide average instead of the first gap's own fit alone.
    _gap_fits = []
    for _i in range(len(DAY_FLIGHTS) - 1):
        _ke, _Ta = _loo_exp(raw_flights[_i], raw_flights[_i + 1])
        if _ke is not None:
            _gap_fits.append((_ke, _Ta))
    if len(_gap_fits) == 0:
        raise SystemExit("No valid LOO exponential fit found on any gap of the day.")
    _k_avg = float(np.mean([f[0] for f in _gap_fits]))
    _Ta_avg = float(np.mean([f[1] for f in _gap_fits]))
    print(f"    {len(_gap_fits)}/{len(DAY_FLIGHTS)-1} gaps fitted -- "
         f"k={_k_avg*1000:.4f}e-3/s, T_amb={_Ta_avg:.1f} C")
    tE,TE,gE,sE=predict_one("exponential", k_override=_k_avg, T_amb_override=_Ta_avg)
    print("\n  Continuous physics run (no gap-fill model at all -- the "
         "calibrated core physics carries straight through every gap)...")
    tC,TC,gC,sC=predict_one("physics")
    _fit_gap1 = _fit_exp(raw_flights[0], raw_flights[1], WHEEL_ID)
    if _fit_gap1 is not None:
        _k_gap1, _Ta_gap1 = _fit_gap1["k"], _fit_gap1["T_amb"]
        print(f"\n  Exponential run (gap-1 k={_k_gap1*1000:.4f}e-3/s, "
              f"T_amb={_Ta_gap1:.1f} C)...")
        tP,TP,gP,sP=predict_one("exponential",
                                 k_override=_k_gap1, T_amb_override=_Ta_gap1)
    else:
        print("  Gap-1 fit failed -- skipping.")
        tP=TP=gP=sP=None


    tm_all=np.concatenate([s["tm_b"]+s["touchdown_time_s"] for s in segments])
    Tm_all=np.concatenate([s["Tm_b"] for s in segments])
    t0,t1=float(tE.min()),float(tE.max())
    # NORMALIZED_TIME is switched by the caller loop below: this figure's
    # temperatures are already in degC, so only the TIME axis changes between
    # the two versions.
    NORMALIZED_TIME = True
    def xn(t):
        a=np.asarray(t,float)
        return (a-t0)/(t1-t0) if NORMALIZED_TIME else a/3600.0
    def draw(ax,axe,t_all,T_all,gaps,segs,title):
        ax.plot(xn(tm_all),Tm_all,"r.",ms=2,alpha=.5,label="measured")
        for k,(ts,Ts) in enumerate(segs):
            if k > 0 and k-1 < len(gaps) and "rec_start_abs" in gaps[k-1]:
                _rs = gaps[k-1]["rec_start_abs"]
                _vis = ts >= _rs
                ts_plot, Ts_plot = ts[_vis], Ts[_vis]
            else:
                ts_plot, Ts_plot = ts, Ts
            ax.plot(xn(ts_plot),Ts_plot,"g-",lw=1.3,
                    label="model" if k==0 else None)
        for j,g in enumerate(gaps):
            fi=j==0
            if g["model"]=="exponential" and g["k"] is not None:
                tg=np.linspace(g["t0"],g["t1"],300)
                Tg=g["T_amb"]+(g["T0"]-g["T_amb"])*np.exp(-g["k"]*(tg-g["t0"]))
                ax.plot(xn(tg),Tg,"b--",lw=1.8,zorder=4,label="gap (exp LOO)" if fi else None)
            elif g["model"]=="physics":
                pass   # the segment's own untrimmed model line already shows this
            else:
                ax.plot([xn(g["t0"]),xn(g["t1"])],[g["T0"],g["T1"]],"k--",
                        lw=1.8,zorder=4,label="gap (linear LOO)" if fi else None)
        for s in segments:
            ax.axvline(xn(s["touchdown_time_s"]),color=".5",ls=":",lw=.8)
            ax.text(xn(s["touchdown_time_s"]),1.,s["label"],rotation=90,
                    va="top",ha="right",fontsize=7,color=".4",
                    transform=ax.get_xaxis_transform())
        ax.set(ylabel="BTMS [degC]",title=title); ax.legend(fontsize=8); ax.grid(alpha=.3)
        et,ep=[],[]
        for s in segments:
            tb=s["tm_b"]+s["touchdown_time_s"]; Tb=s["Tm_b"]
            ok=(tb>=t_all.min())&(tb<=t_all.max())
            if ok.sum()==0: continue
            Tp=np.interp(tb[ok],t_all,T_all); vl=np.abs(Tb[ok])>5.
            if vl.sum()==0: continue
            et.append(tb[ok][vl]); ep.append((Tp[vl]-Tb[ok][vl])/Tb[ok][vl]*100)
        if et: axe.plot(xn(np.concatenate(et)),np.concatenate(ep),".",ms=2.5,color="tab:green",alpha=.7)
        axe.axhline(0,color="k",lw=.8,ls="--")
        axe.axhline(20,color=".7",lw=.6,ls=":"); axe.axhline(-20,color=".7",lw=.6,ls=":")
        for s in segments: axe.axvline(xn(s["touchdown_time_s"]),color=".5",ls=":",lw=.8)
        axe.set(ylabel="Rel. error [%]\n(model-meas)/meas"); axe.grid(alpha=.3)
    astr=f"alpha={calibration['alpha']:.2f}"
    _ncols = 3 if tP is not None else 2
    for NORMALIZED_TIME in (True, False):   # one window each way
        fig,axes=plt.subplots(2,_ncols,figsize=(10*_ncols,9),sharex=True,
                              gridspec_kw=dict(height_ratios=[3,1.8]))
        draw(axes[0,0],axes[1,0],tE,TE,gE,sE,
             f"Wheel {WHEEL_ID} -- Exponential, LOO avg over {len(_gap_fits)} "
             f"gap(s) ({astr})")
        draw(axes[0,1],axes[1,1],tC,TC,gC,sC,
             f"Wheel {WHEEL_ID} -- Continuous physics, no gap-fill ({astr})")
        if tP is not None:
            draw(axes[0,2],axes[1,2],tP,TP,gP,sP,
                 f"Wheel {WHEEL_ID} -- Exp gap-1 fixed params "
                 f"(k={_k_gap1*1000:.3f}e-3/s, {astr})")
        _xl = ("Normalized time [0=start, 1=end]" if NORMALIZED_TIME
               else "Time [h] (absolute, from segmentation)")
        for _c in range(_ncols):
            axes[1,_c].set(xlabel=_xl)
        fig.suptitle(
            f"BTEM pipeline -- {DAY_FLIGHTS[0]} ... {DAY_FLIGHTS[-1]}  "
            f"(cal: {FLIGHT_CAL})",
            fontsize=12)
        fig.tight_layout()
    plt.show()

# =============================================================================
# ENTRY POINT
# =============================================================================
def _load_calibration_bundle_epsT(source):
    """Same logic as predict_day_from_csv.py's own load_calibration_bundle(),
    but checks/extracts against THIS script's own 8-key NAMES (eps_T
    included) instead of the shared module's 7-key one -- per Ugo,
    2026-08-20: the shared version silently drops eps_T from the loaded
    core even when the JSON on disk has it, since it only ever extracts
    the keys in its OWN NAMES list. Kept local rather than changing the
    shared function, same reasoning as apply_core() elsewhere in this
    file."""
    with open(source) as f:
        raw = json.load(f)
    if 'core' not in raw or not all(k in raw['core'] for k in NAMES):
        raise core_mod.PipelineError(
            f"{source} does not contain a full {len(NAMES)}-parameter core "
            f"under 'core' -- not an eps_T calibration bundle.")
    core = {k: float(raw['core'][k]) for k in NAMES}
    alpha = float(raw.get('alpha', 1.0))
    h_nat_flight = float(raw.get('h_nat_flight', raw.get('H_nat_flight', m.h_nat)))
    taxi_mode = raw.get('taxi_mode', 'honest')
    pdur_raw = raw.get('pulse_dur', raw.get('taxi_pulse_dur_s'))
    pulse_dur = float(pdur_raw) if (taxi_mode == 'pulse' and pdur_raw) else None
    return dict(core=core, alpha=alpha, h_nat_flight=h_nat_flight,
               taxi_mode=taxi_mode, pulse_dur=pulse_dur, source=str(source))


def main():
    print("="*74)
    print(f"BTEM PIPELINE  |  day: {DAY_FLIGHTS}  |  wheel: {WHEEL_ID}")
    print("="*74)
    paths=[find_csv(f) for f in DAY_FLIGHTS]
    segments,_,_=core_mod.segment_csv_inputs(paths,log=print,wheel=WHEEL_ID)
    if len(segments)!=len(DAY_FLIGHTS):
        raise SystemExit(f"segment_csv_inputs: {len(segments)} flights, expected {len(DAY_FLIGHTS)}")
    seg_cal=segments[0]
    seg1_cal=segments[1] if len(segments)>1 else None
    print(f"\nCalibration flight: {FLIGHT_CAL}  |  wheel {WHEEL_ID}")
    print(f"  Touchdown : {seg_cal['v_td_kt']:.0f} kt")
    print(f"  E_landing : {seg_cal['E_landing_MJ']:.4f} MJ/brake  (single source of truth)")
    CORE,NAMES=run_core_calibration(seg_cal,seg1_cal)
    run_pretouchdown_calibration(seg_cal,CORE,NAMES)
    calibration=_load_calibration_bundle_epsT(STAGE2_FILE)
    raw_flights=[_load_raw(f) for f in DAY_FLIGHTS]
    run_day_prediction(segments,calibration,raw_flights)

if __name__=="__main__":
    main()