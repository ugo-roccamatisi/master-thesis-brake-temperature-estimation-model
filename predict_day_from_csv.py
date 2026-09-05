# =============================================================================
# SANITIZED PUBLIC VERSION
# All calibrated parameter values have been replaced with generic round
# placeholders, and all flight identifiers, timestamps and paths with
# examples. Calibrate on your own data before any quantitative use.
# =============================================================================
# -*- coding: utf-8 -*-
"""
predict_day_from_csv.py
========================
Predicts BTMS (brake temperature) over a whole day, one isolated flight, or a
chosen range -- from either ONE CSV covering the day or SEVERAL per-flight CSVs
laid end to end by their own timestamps.

Column convention (flight_0001-0008): Timestamp, GROUND SPEED, GROSS WEIGHT,
Static Air Temperature [BNR], Temperature Brake n<degree>1, LH/RH L/G COMPRESSED,
Thrust_Reverser_Cowl_Position.

GAP BOUNDARY: neutral h_nat governs from touchdown of flight N to the first
moment ground speed leaves 0 on flight N+1 (t_move_start) -- movement alone
ends the "parked" regime, whether or not braking has started. H_NAT_FLIGHT
resumes from there through the next touchdown.

CALIBRATION -- two modes:
  --calibration default     Average two existing calibration bundles supplied
                            via --core-json-a/--core-json-b. Nothing is
                            hardcoded: without both paths the script stops
                            rather than guess.
  --calibration fit-first    Fit the core AND alpha/H_nat_flight on the FIRST
                            flight in the CSV, then apply that to whichever
                            flights --flights selects.

FLIGHT SEGMENTATION is a tunable heuristic: a parking gap is any stretch where
ground speed stays below --park-kt for at least --min-park-gap-min minutes.
NOT validated against a real multi-flight-per-file CSV -- check the printed
segment table before trusting it, and adjust --min-park-gap-min if a flight
gets split or two get merged.
"""

import argparse
import contextlib
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib
if __name__ == '__main__':
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy import stats as _stats
import btem_v5_full_flight as m

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, '.predict_day_cache')


class PipelineError(Exception):
    """Raised for any user-facing problem this pipeline can't recover
    from (bad CSV, no usable flight, missing calibration file, etc).
    Caught and reported cleanly by main() for the CLI, or by the
    Streamlit app for the web UI -- never left to crash either one."""


def _interp(x, y, fill=0.0):
    """interp1d with the flat-extrapolation settings used throughout."""
    return interp1d(x, y, bounds_error=False, fill_value=fill)


@contextlib.contextmanager
def _module_attr(name, value):
    """Temporarily set m.<name>, restoring the previous value on the way out
    (including on exception -- the hand-written save/restore pairs this
    replaces leaked module state whenever the body raised)."""
    saved = getattr(m, name)
    setattr(m, name, value)
    try:
        yield
    finally:
        setattr(m, name, saved)


@contextlib.contextmanager
def _flight_knobs(alpha, h_nat_flight, c_tube_mult=1.0):
    """Apply the three 'flight'-phase correction knobs, then restore the
    neutral values (alpha=1, mult=1, H_NAT_FLIGHT=h_nat)."""
    m.P_FLIGHT_SCALE, m.C_TUBE_FLIGHT_MULT, m.H_NAT_FLIGHT = (
        alpha, c_tube_mult, h_nat_flight)
    try:
        yield
    finally:
        m.P_FLIGHT_SCALE, m.C_TUBE_FLIGHT_MULT, m.H_NAT_FLIGHT = 1.0, 1.0, m.h_nat


# =============================================================================
# 1. CONSTANTS -- defaults are this project's own A320 values. Every entry
# in AIRCRAFT_DEFAULTS can be overridden (see configure_constants below,
# --rho-air/--a-wing/etc. on the CLI, or the "Aircraft constants" expander
# in the Streamlit app) to reuse this script on a different aircraft type.
# KT2MS and LB2KG are plain unit conversions, not aircraft-specific, and
# are NOT in this dict. The 7-parameter thermal core's own bounds (NAMES/
# BASE/FRAC below) are specific to this project's brake thermal model and
# are NOT covered by this override mechanism -- reusing this script on
# another aircraft's BRAKES (not just its aerodynamics) would need those
# adjusted separately.
# =============================================================================
KT2MS = 0.514444
LB2KG = 0.453592

AIRCRAFT_DEFAULTS = dict(
    n_brakes=4,               # braked wheels sharing the reconstructed energy
    max_e_taxi_mj_brake=0,   # hard cap on taxi-out energy per brake [MJ/brake]
                              # DISABLED: the cap only existed because G(v)
                              # applied one self-calibrated constant across the
                              # whole taxi, so a fast taxi inflated G*distance
                              # whether or not braking happened. Delta-V only
                              # attributes real observed speed loss, so a fast
                              # taxi with hard decelerations legitimately gives
                              # more energy. Set positive to re-enable.
    rho_air=1.225,            # air density [kg/m3], sea level
    a_wing=123.0,             # wing reference area [m2], for aero drag
    c_d=0.12,                 # aerodynamic drag coefficient
    mu_rr=0.009,               # rolling resistance coefficient
    n_sp=10, l_sp=1.6, b_sp=0.625,  # spoiler count / length / width [m]
    theta_max_deg=50.0,       # max spoiler deflection [deg]
    c_sp=1.5,                 # spoiler drag coefficient
    k_rt=15000.0,             # reverse-thrust force [N], in the landing-roll
                              # force balance; applies only while reverse
                              # thrust is active AND ground speed is above
                              # v_th_rev_kt below. 0.0 disables it.
    v_th_rev_kt=40.0,         # ground speed below which reverse thrust cuts off [kt]
    disable_retraction_window=False,  # model the gear-retraction cooling
                              # window by default: the gear is treated as still
                              # deployed (open-bay, ambient-exposed) for
                              # T_RETRACT_S after take-off lift-off, only
                              # switching to the closed-bay T_BAY_C ambient
                              # after that. True skips the transition entirely
                              # and goes closed-bay as soon as airborne.
    n_sub_land=10,            # sub-steps per raw landing-roll interval
    v_td_threshold_kt=110.0,  # rule_100kt: touchdown = last sample above this
    v_roll_end_kt=20.0,       # ground speed below which the roll-out is "done"
    v_td_plausible=(110.0, 165.0),  # sanity range for the detected touchdown speed
    mass_lb_threshold=80000.0,  # GROSS WEIGHT above this -> assumed already in lb
    v_stop_kt=0.5,             # ground speed below which the aircraft is "stopped"
                               # (finer than v_roll_end_kt -- marks full stop, not
                               # just end of the landing roll)
    lg_debounce_s=5.0,        # gap this short in the landing-gear signal is
                               # bridged rather than treated as a real state change
    stop_debounce_s=60.0,     # how long the aircraft must stay under v_stop_kt
                               # to count as genuinely stopped, not just a dip
    # UNCERTAINTY MODEL -- see build_
    # uncertainty_envelope() and its own module-level warning at import.
    # Both of the following are UNVERIFIED PLACEHOLDER VALUES, not sourced
    # from any datasheet or literature -- flagged loudly at runtime every
    # time the envelope is built, specifically so neither quietly ends up
    # in the thesis as if it were a validated figure.
    climb_duration_s=1200.0,  # UNVERIFIED assumption: how long after the
                               # start of the take-off roll the "climb"
                               # phase is assumed to last (the model has no
                               # altitude channel, so there is no direct way
                               # to detect the actual climb-to-cruise
                               # transition -- this is a fixed-duration
                               # stand-in for it)
    u_bts_c=2.0,               # UNVERIFIED placeholder: BTMS sensor
                               # measurement uncertainty [deg C]. No
                               # manufacturer specification or literature
                               # value was found for this project's
                               # thermocouple -- this is a generic
                               # aerospace-thermocouple ballpark, not a
                               # sourced figure. Override with a real value
                               # as soon as one is available.
    u_tol_c=20.0,              # UNVERIFIED placeholder # the governing
                               # tolerance limit U_tol (an engineering/
                               # policy target -- "how much combined
                               # uncertainty is acceptable" -- not a
                               # measured quantity, so there is no
                               # literature value to look up; this needs a
                               # deliberate choice, not a source). Drift is
                               # flagged wherever u_c(Tx) > U_tol.
)


def configure_constants(overrides=None):
    """Applies AIRCRAFT_DEFAULTS, overridden by any non-None entry in
    `overrides` (a dict using the same keys), as module-level globals used
    throughout the rest of this file. Called once before the pipeline
    runs -- by main() for the CLI, or explicitly by the Streamlit app
    before segment_csv_inputs()."""
    values = dict(AIRCRAFT_DEFAULTS)
    if overrides:
        values.update({k: v for k, v in overrides.items() if v is not None})
    g = globals()
    g['N_BRAKES'] = values['n_brakes']
    g['MAX_E_TAXI_MJ_BRAKE'] = values['max_e_taxi_mj_brake']
    g['RHO_AIR'] = values['rho_air']
    g['A_WING'] = values['a_wing']
    g['C_D'] = values['c_d']
    g['MU_RR'] = values['mu_rr']
    g['N_SP'], g['L_SP'], g['B_SP'] = values['n_sp'], values['l_sp'], values['b_sp']
    g['THETA_MAX'] = np.deg2rad(values['theta_max_deg'])
    g['C_SP'] = values['c_sp']
    g['A_SP'] = g['N_SP']*g['L_SP']*g['B_SP']*np.sin(g['THETA_MAX'])
    g['K_RT'] = values['k_rt']
    g['V_TH_REV'] = values['v_th_rev_kt']*KT2MS
    g['DISABLE_RETRACTION_WINDOW'] = values['disable_retraction_window']
    g['N_SUB_LAND'] = values['n_sub_land']
    g['V_TD_THRESHOLD_KT'] = values['v_td_threshold_kt']
    g['V_ROLL_END_KT'] = values['v_roll_end_kt']
    g['V_TD_PLAUSIBLE'] = tuple(values['v_td_plausible'])
    g['MASS_LB_THRESHOLD'] = values['mass_lb_threshold']
    g['V_STOP_KT'] = values['v_stop_kt']
    g['LG_DEBOUNCE_S'] = values['lg_debounce_s']
    g['STOP_DEBOUNCE_S'] = values['stop_debounce_s']
    g['CLIMB_DURATION_S'] = values['climb_duration_s']
    g['U_BTS_C'] = values['u_bts_c']
    g['U_TOL_C'] = values['u_tol_c']
    g['_APPLIED_AIRCRAFT'] = values


# =============================================================================
# 1b. THERMAL MODULE CONSTANTS -- everything in btem_v5_full_flight.py
# (imported as `m`) that ISN'T one of the 7 fitted core parameters (those go
# through NAMES/BASE/FRAC and set_core()). Fixed brake/wheel geometry, material
# and timing -- what a different aircraft's BRAKES would change, as opposed to
# AIRCRAFT_DEFAULTS above, which is about the airframe's aerodynamics.
# Heat capacities are split here as mass_*/cp_* so both stay editable, where
# the module writes them as one product (e.g. C_wheel = 45.0*900.0).
# =============================================================================
THERMAL_MODULE_DEFAULTS = dict(
    # geometry [m]
    r_wheel=0.584,
    R_ext=0.183, R_int=0.118, e_disc=0.022,
    R_rim=0.255, L_rim=0.417,
    A_ct=0.178, A_cw=0.378,  # drawing-measured, not derived from R/L above
    # disc material
    rho_c=1800.0, cp_c=1420.0, kz_c=10.0,
    # heat capacities [kg] x [J/kg/K]
    mass_wheel=45.0, cp_wheel=900.0,
    mass_shield=2.0, cp_shield=500.0,
    mass_pist=4.0, cp_pist=600.0,
    mass_sens=1.0, cp_sens=500.0,
    # fixed conductances [W/K] (G_RW/G_ST/G_sens are FITTED -- not here)
    G_PH=1.5, G_SP=1.0, G_SW=0.5,
    # emissivities [-] (eps_DT and eps_T are FITTED -- not here; eps_T
    # moved here from a fixed constant, 2026-08-20)
    eps_DS=0.80, eps_SW=0.25, eps_end=0.80, eps_W=0.19,
    eps_PH=0.50, eps_S1P=0.70,
    # convection
    h_fan=500.0,               # taxi brake-cooling fan coefficient [W/m2K]
    k_air=0.02588, nu_air=1.608e-5, pr_air=0.728,  # air properties
    l_conv=0.30,                # convection length scale [m]
    # timing / thresholds
    takeoff_accel_thresh=1.0,   # [m/s2] above this on the ground = take-off
    t_retract_s=20.0,           # gear-retraction cooling window [s]
    h_approach_ft=2000.0,       # altitude where approach cooling starts
    approach_fallback_s=100.0,  # used only if no altitude column is found
    t_bay_c=20.0,                # gear-bay temperature, gear stowed [C]
    v_taxi_kt=20.0,              # representative taxi ground speed [kt]
    p_brake_thresh=50e3,        # [W] power above this, in 'flight', counts
                               # as real ground braking -- REQUIRES the
                               # 2026-08-10 module patch that promotes
                               # P_BRAKE_THRESH out of rhs()'s local scope;
                               # silently has NO EFFECT on an unpatched
                               # module (see configure_thermal_module)
    flight_forced_cooling=True,  # model the gear-retraction/approach
                                 # forced-convection windows during 'flight'
    fan_on=False,               # taxi brake-cooling fans (this project's
                                 # own convention: off, matches every other
                                 # script in this pipeline)
    ground_fan_on=False,        # parking ground fans (same convention)
)


def configure_thermal_module(overrides=None):
    """Applies THERMAL_MODULE_DEFAULTS, overridden by any non-None entry
    in `overrides`, directly onto the imported thermal module `m` --
    including recomputing every quantity the module itself derives from
    them (areas, heat capacities, G_brake/G_cool), using the exact same
    formulas as the module's own top-level code. Call before segmenting/
    predicting; like configure_constants, called once at import time with
    the defaults so the module works correctly even if nothing is
    overridden."""
    v = dict(THERMAL_MODULE_DEFAULTS)
    if overrides:
        v.update({k: val for k, val in overrides.items() if val is not None})

    m.r_wheel = v['r_wheel']
    m.R_ext, m.R_int, m.e_disc = v['R_ext'], v['R_int'], v['e_disc']
    m.R_rim, m.L_rim = v['R_rim'], v['L_rim']
    m.A_ct, m.A_cw = v['A_ct'], v['A_cw']
    m.rho_c, m.cp_c, m.kz_c = v['rho_c'], v['cp_c'], v['kz_c']
    m.G_PH, m.G_SP, m.G_SW = v['G_PH'], v['G_SP'], v['G_SW']
    m.eps_DS, m.eps_SW = v['eps_DS'], v['eps_SW']
    m.eps_end, m.eps_W = v['eps_end'], v['eps_W']
    m.eps_PH, m.eps_S1P = v['eps_PH'], v['eps_S1P']
    m.h_fan = v['h_fan']
    m.K_AIR, m.NU_AIR, m.PR_AIR = v['k_air'], v['nu_air'], v['pr_air']
    m.L_CONV = v['l_conv']
    m.TAKEOFF_ACCEL_THRESH = v['takeoff_accel_thresh']
    m.T_RETRACT_S = v['t_retract_s']
    m.H_APPROACH_FT = v['h_approach_ft']
    m.APPROACH_FALLBACK_S = v['approach_fallback_s']
    m.T_BAY_C = v['t_bay_c']
    m.T_BAY_K = m.T_BAY_C + 273.15
    m.V_TAXI_KT = v['v_taxi_kt']
    m.FLIGHT_FORCED_COOLING = v['flight_forced_cooling']
    m.FAN_ON = v['fan_on']
    m.GROUND_FAN_ON = v['ground_fan_on']
    if hasattr(m, 'P_BRAKE_THRESH'):
        m.P_BRAKE_THRESH = v['p_brake_thresh']
    elif v['p_brake_thresh'] != THERMAL_MODULE_DEFAULTS['p_brake_thresh']:
        print(f"  WARNING: p_brake_thresh was overridden to "
             f"{v['p_brake_thresh']:.0f} W, but btem_v5_full_flight.py "
             f"still has it as a LOCAL inside rhs() (pre-2026-08-10 patch) "
             f"-- this override has NO EFFECT until that module is patched.")

    # --- derived quantities, same formulas as the module's own top level ---
    m.A_f = np.pi*(m.R_ext**2 - m.R_int**2)
    m.A_in = 2*np.pi*m.R_int*m.e_disc
    m.A_out = 2*np.pi*m.R_ext*m.e_disc
    m.A_SW = 2*np.pi*m.R_rim*m.L_rim
    m.A_cp = m.A_f
    m.A_disc = m.A_out
    m.A_S1P = m.A_f/2
    m.A_fend_S1 = m.A_f/2
    m.A_fend_S5 = m.A_f
    m.C_disc = m.rho_c*m.A_f*m.e_disc*m.cp_c
    m.C_wheel = v['mass_wheel']*v['cp_wheel']
    m.C_shield = v['mass_shield']*v['cp_shield']
    m.C_pist = v['mass_pist']*v['cp_pist']
    m.C_sens = v['mass_sens']*v['cp_sens']
    m.G_brake = m.A_f/(1e-4 + m.e_disc/m.kz_c)
    m.G_cool = m.A_f/(2.55*m.e_disc/m.kz_c)
    globals()['_APPLIED_THERMAL'] = v


configure_constants()          # apply the A320 aerodynamic defaults
configure_thermal_module()     # apply the A320 brake/wheel geometry defaults
# both so this module works as a library too (e.g. imported by Streamlit)
# without every caller having to remember to configure it


# =============================================================================
# 1c. FIT-FIRST CALIBRATION CACHE -- unlike every other
# script in this pipeline, this one refit from scratch on every run, even
# on unchanged data. Cached by a fingerprint of the CSV content itself
# (not the file path/name, which may not exist for an uploaded file) plus
# every aircraft/thermal constant that could change the fit -- so editing
# a constant automatically invalidates the cache, no manual deletion
# needed (same spirit as this project's other caches, but keyed on
# content instead of a flight name since this script's input is arbitrary
# user-supplied CSVs).
# =============================================================================
def _content_bytes(source):
    """Reads a CSV source (path string or file-like, e.g. Streamlit's
    UploadedFile) as bytes without disturbing its read position, so it
    can still be parsed normally afterward."""
    if hasattr(source, 'getvalue'):
        return source.getvalue()
    if hasattr(source, 'read'):
        pos = source.tell() if hasattr(source, 'tell') else None
        data = source.read()
        if pos is not None:
            source.seek(pos)
        return data if isinstance(data, bytes) else data.encode()
    with open(source, 'rb') as f:
        return f.read()


def data_fingerprint(csv_sources):
    """A short hash covering the CSV content AND every currently-applied
    aircraft/thermal constant -- two calls with the same data but
    different constants (or vice versa) never collide."""
    h = hashlib.sha256()
    for s in csv_sources:
        h.update(_content_bytes(s))
    h.update(json.dumps(globals().get('_APPLIED_AIRCRAFT', {}), sort_keys=True).encode())
    h.update(json.dumps(globals().get('_APPLIED_THERMAL', {}), sort_keys=True).encode())
    return h.hexdigest()[:16]


def _cache_path(fingerprint):
    return os.path.join(CACHE_DIR, f'calib_{fingerprint}.json')


def load_calibration_cache(fingerprint):
    path = _cache_path(fingerprint)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return dict(core=raw['core'], alpha=raw['alpha'],
               h_nat_flight=raw['h_nat_flight'], taxi_mode=raw['taxi_mode'],
               pulse_dur=raw.get('pulse_dur'))


def save_calibration_cache(fingerprint, calib):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(fingerprint), 'w') as f:
        json.dump(dict(calib, fingerprint=fingerprint,
                       written_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())),
                  f, indent=2)


# Anchor weights for fit_core()/fit_alpha_h_nat_flight() below -- synced
# to btem_pipeline.py's own ANCHOR_WEIGHT_CORE/ANCHOR_WEIGHT_PRE (per
# Ugo, 2026-08-19: this module's own fit had drifted to an unweighted
# RMSE with no anchor at all, a real methodological difference from
# btem_pipeline.py that could calibrate to different alpha/core values
# on the same data).
ANCHOR_WEIGHT_CORE = 50.0
ANCHOR_WEIGHT_PRE = 55.0

# The calibrated set swaps eta_disc for eps_T: eta_disc is no longer fitted
# (m.ETA_DISC keeps its own module default, never touched by set_core below),
# while eps_T joins eps_DT as a calibrated emissivity rather than a fixed
# THERMAL_MODULE_DEFAULTS constant -- so it has no CLI flag or UI slider
# either, matching eps_DT.
NAMES = ['C_tube', 'G_ST', 'tau', 'eps_DT', 'G_RW', 'h_nat', 'eps_T']
BASE = dict(C_tube=10000.0, G_ST=2.5, tau=500.0/3.0, eps_DT=0.90, G_RW=2.0,
           h_nat=8.0, eps_T=0.80)
FRAC = dict(C_tube=(0.5, 1.5), G_ST=(0.3/2.5, 1.8), tau=(0.012, 2.5),
           G_RW=(0.5, 4.0), eps_DT=(0.9, 1.1), h_nat=(1.0/8.0, 25.0/8.0),
           eps_T=(0.9, 1.1))
LO = np.array([BASE[k]*FRAC[k][0] for k in NAMES])
HI = np.array([min(BASE[k]*FRAC[k][1], 1.0) if k in ('eps_DT', 'eps_T')
              else BASE[k]*FRAC[k][1] for k in NAMES])


def F_opposing(v, mass):
    return (0.5*RHO_AIR*A_WING*C_D*v**2 + MU_RR*mass*9.81
            + 0.5*RHO_AIR*A_SP*C_SP*v**2)


# =============================================================================
# EXPONENTIAL GAP MODEL an alternative to
# the h_nat/H_NAT_FLIGHT continuous-physics gap (predict()'s default).
# Newton's-law-of-cooling parameters (k, T_amb) are fitted ONCE, on the
# FIRST gap of the day (segments[0] -> segments[1]), and reused unchanged
# for every subsequent gap -- no per-gap refit. See predict()'s own
# gap_model='exp_gap1' branch for how these two numbers get applied.
# =============================================================================
_RISE_THRESHOLD_C = 0.5   # first-local-minimum detection, same convention
_CONFIRM_N = 3            # used throughout this project's other scripts


def _first_local_minimum(t_arr, T_arr, rise_threshold=_RISE_THRESHOLD_C,
                         confirm_n=_CONFIRM_N):
    """First local minimum of T_arr (sorted by t_arr): the running minimum
    up to the first sustained rise of more than `rise_threshold` above it,
    confirmed over `confirm_n` consecutive points (guards against a single
    noisy sample being mistaken for the true minimum)."""
    order = np.argsort(t_arr)
    t_s, T_s = t_arr[order], T_arr[order]
    running_min = T_s[0]
    idx_min = 0
    i = 1
    while i < len(T_s):
        if T_s[i] < running_min:
            running_min = T_s[i]
            idx_min = i
        elif T_s[i] > running_min + rise_threshold:
            if (i + confirm_n <= len(T_s)
                    and np.all(T_s[i:i+confirm_n] > running_min + rise_threshold*0.5)):
                return t_s[idx_min], T_s[idx_min]
        i += 1
    return t_s[idx_min], T_s[idx_min]


def fit_exp_gap1(seg0, seg1, peak_t0_abs, log=print):
    """Fits T(t) = T_amb + (T_peak-T_amb)*exp(-k*(t-t_peak)) on the day's
    FIRST gap (seg0 -> seg1), over every measured point from seg0's MEASURED
    peak (peak_t0_abs) to seg1's first local minimum. Anchoring on the model's
    simulated peak instead would let a timing mismatch pull still-rising points
    in and bias k towards a slower decay.

    T_amb is NOT a free parameter: it's the mean of both flights' own ambient
    sensor, sampled over windows that differ from the regression set -- seg0
    from its peak to the END of its recording, seg1 from its peak to the local
    minimum (matching btem_pipeline.py's _fit_exp). k comes from a linear
    regression of ln(T-T_amb) against time over the whole regression set, not
    just the two anchor points.

    Returns (k, T_amb). Raises PipelineError if the fit can't be done -- it
    needs only T_peak and ambient, so failure signals a real data problem with
    this day's first gap, not a routine edge case."""
    tm_b0, Tm_b0 = seg0['tm_b'], seg0['Tm_b']
    tm_b1, Tm_b1 = seg1['tm_b'], seg1['Tm_b']

    # touchdown_time_s already places every segment on the SAME absolute
    # time axis (segment_csv_inputs() lays the whole day out on one t_full)
    # -- no per-CSV datetime offset needed, unlike a multi-file day.
    tm_b0_abs = tm_b0 + seg0['touchdown_time_s']
    tm_b1_abs = tm_b1 + seg1['touchdown_time_s']

    post0 = tm_b0_abs >= peak_t0_abs
    pre1 = tm_b1 < 0   # seg1's own pre-touchdown window
    if pre1.sum() == 0:
        raise PipelineError(
            "Exponential gap fit failed: flight_02 has no pre-touchdown "
            "measured points to anchor the first local minimum.")

    t_min_abs, T_min = _first_local_minimum(tm_b1_abs[pre1], Tm_b1[pre1])
    if t_min_abs <= peak_t0_abs:
        raise PipelineError(
            "Exponential gap fit failed: flight_02's first local minimum "
            "occurs before flight_01's own peak -- check the data for "
            "this transition.")

    t_pts = np.concatenate([tm_b0_abs[post0],
                            tm_b1_abs[pre1 & (tm_b1_abs > peak_t0_abs)
                                     & (tm_b1_abs <= t_min_abs)]])
    T_pts = np.concatenate([Tm_b0[post0],
                            Tm_b1[pre1 & (tm_b1_abs > peak_t0_abs)
                                 & (tm_b1_abs <= t_min_abs)]])
    order = np.argsort(t_pts)
    t_pts, T_pts = t_pts[order], T_pts[order]

    # ambient temperature: same window as btem_pipeline.py's _fit_exp. Both
    # windows run FROM the peak (t_pk), not to it: amb0 extends to segment 0's
    # recording end, amb1 is bounded t_pk..t_min. Starting amb1 at
    # seg1['tm_min'] instead would reach back into that flight's own cruise
    # altitude, averaging in far colder air and flattening the fitted decay.
    t_pk_local0 = peak_t0_abs - seg0['touchdown_time_s']
    t_pk_local1 = peak_t0_abs - seg1['touchdown_time_s']
    seg0['apply_module_state']()
    amb0_t = np.linspace(t_pk_local0, seg0['tm_max'], 200)
    amb0 = m._T0_interp(np.clip(amb0_t, seg0['tm_min'], seg0['tm_max'])) - 273.15
    seg1['apply_module_state']()
    amb1_t = np.linspace(t_pk_local1, t_min_abs - seg1['touchdown_time_s'], 200)
    amb1 = m._T0_interp(np.clip(amb1_t, seg1['tm_min'], seg1['tm_max'])) - 273.15
    T_amb = float(np.mean(np.concatenate([amb0, amb1])))

    excess = T_pts - T_amb
    valid = excess > 0.5
    if valid.sum() < 3:
        raise PipelineError(
            "Exponential gap fit failed: fewer than 3 usable points "
            "between flight_01's peak and flight_02's first local minimum "
            "(too close to ambient, or too little measured data).")
    x = t_pts[valid] - peak_t0_abs
    y = np.log(excess[valid])
    slope, _, r_value, _, _ = _stats.linregress(x, y)
    k = -slope
    if k <= 0:
        raise PipelineError(
            "Exponential gap fit failed: fitted k<=0 (the data doesn't "
            "show net cooling toward ambient over the first gap).")

    log(f"  Exponential gap fit (first gap only): "
       f"t_peak={peak_t0_abs - seg0['touchdown_time_s']:.0f}s (seg0-relative)  "
       f"T_peak={float(Tm_b0[post0][0]):.1f} C  "
       f"k={k*1000:.4f}e-3/s  T_amb={T_amb:.1f} C  "
       f"R2={r_value**2:.3f}  n={int(valid.sum())} points")
    return k, T_amb


def _safe_exp(logx):
    """np.exp clipped to a range wide enough that it never affects which
    side of any real parameter bound the result falls on, but avoids the
    'overflow encountered in exp' warning Nelder-Mead's exploratory steps
    can trigger (a proposed step can be numerically huge before the
    following np.clip(..., LO, HI) brings it back in range)."""
    return np.exp(np.clip(logx, -50.0, 50.0))


def set_core(xp):
    d = dict(zip(NAMES, xp))
    m.C_tube, m.G_ST, m.G_RW, m.eps_DT = (d['C_tube'], d['G_ST'],
                                          d['G_RW'], d['eps_DT'])
    m.G_sens = m.C_sens/d['tau']
    m.h_nat = d['h_nat']
    m.eps_T = d['eps_T']


def apply_core(core):
    set_core(np.array([core[k] for k in NAMES]))


# =============================================================================
# 2. CSV LOADING -- one file for a whole day, OR several per-flight files
# that, laid end to end by their own real timestamps, make up one day.
# =============================================================================
def _read_one_csv(source):
    name = getattr(source, 'name', str(source))
    df = pd.read_csv(source, sep=None, engine='python', encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    if 'Timestamp' not in df.columns:
        raise PipelineError(f"No 'Timestamp' column in {name}")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'],
                                     format='%Y/%m/%d %H:%M:%S.%fT', errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    if len(df) == 0:
        raise PipelineError(f"Every row failed timestamp parsing in {name}")
    return df


def load_day_csvs(sources, log=print):
    """Reads one or several CSVs (path strings or file-like objects, e.g.
    Streamlit's UploadedFile) and concatenates them into a single day,
    ordered by their own real Timestamp column -- so the order they're
    given in doesn't matter, and the true parking gaps between flights
    come straight from the timestamps rather than being guessed. Returns
    the combined dataframe plus t_full (seconds from the first sample of
    the day) and ground speed [kt]."""
    frames = [_read_one_csv(s) for s in sources]
    names = [getattr(s, 'name', str(s)) for s in sources]

    if len(frames) > 1:
        spans = [(f['Timestamp'].min(), f['Timestamp'].max()) for f in frames]
        for i in range(len(spans)):
            for j in range(i+1, len(spans)):
                if spans[i][0] <= spans[j][1] and spans[j][0] <= spans[i][1]:
                    log(f"  WARNING: {names[i]} and {names[j]} have "
                       f"overlapping timestamp ranges -- are these "
                       f"really two different flights?")

    df = pd.concat(frames, ignore_index=True).sort_values('Timestamp').reset_index(drop=True)
    t_full = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds().values
    gspd_kt = df['GROUND SPEED'].fillna(0).values
    return df, t_full, gspd_kt


# =============================================================================
# 3. FLIGHT SEGMENTATION -- split one continuous day into flight blocks
# =============================================================================
def segment_flights(t_full, gspd_kt, min_park_gap_s, park_kt):
    """Returns (start_idx, end_idx) pairs, one per detected flight, covering
    the whole file. A "parking gap" is any stretch of at least min_park_gap_s
    where ground speed never exceeds park_kt.

    The boundary is placed at the gap's TIME midpoint, not the midpoint of its
    sample INDICES: with a real recording gap between two CSVs the sample
    density differs sharply either side, and an index midpoint lands close to
    whichever side has fewer idle samples -- truncating that side's recording
    before it actually ends (seen cutting a segment 90s before its measured
    BTMS peak, biasing every downstream fit)."""
    moving = gspd_kt > park_kt
    n = len(moving)
    idle_runs = []
    i = 0
    while i < n:
        if not moving[i]:
            j = i
            while j < n and not moving[j]:
                j += 1
            if t_full[j-1] - t_full[i] >= min_park_gap_s:
                idle_runs.append((i, j-1))
            i = j
        else:
            i += 1
    boundary_idxs = []
    for a, b in idle_runs:
        # If the idle run contains a genuine DATA GAP (e.g. two
        # concatenated files with no rows between them), that gap IS the
        # true boundary -- snap right after it so the earlier file's own
        # samples are never truncated. A jump of more than 5x the idle run's
        # median sample spacing counts as a real recording gap, not jitter.
        dt = np.diff(t_full[a:b+1])
        if len(dt) > 0 and dt.max() > 5 * np.median(dt):
            gap_i = a + int(np.argmax(dt))   # last index before the gap
            boundary_idxs.append(gap_i + 1)  # first index after the gap
        else:
            t_mid = 0.5*(t_full[a] + t_full[b])
            boundary_idxs.append(a + int(np.argmin(np.abs(t_full[a:b+1] - t_mid))))
    bounds = sorted(set([0] + boundary_idxs + [n-1]))
    return [(bounds[k], bounds[k+1]) for k in range(len(bounds)-1)]


def find_touchdown(t, gspd, lo, hi):
    """rule_100kt, restricted to index range [lo, hi): the LAST sample in
    the block where ground speed exceeds V_TD_THRESHOLD_KT. Ported from
    flight_0001_0004_phase_flight_gaps.py / flight_0005_0008_phase_flight_
    gaps_v2.py's own rule_100kt, which is proven to work on these exact
    files -- simpler and more robust here than the 'top of the
    deceleration ramp' reconstruction tried first: a pre-departure
    gate-parked stretch is always well under the threshold, so it can
    never be the LAST >100kt sample, unlike a low-speed search which can
    latch onto the wrong end of the block. Returns an absolute index into
    the full-day arrays, or None if this block never exceeds the
    threshold (not a real landing)."""
    above = np.where(gspd[lo:hi] > V_TD_THRESHOLD_KT)[0]
    if len(above) == 0:
        return None
    return lo + int(above[-1])


# =============================================================================
# 4. PER-SEGMENT PHYSICS -- ported from build_flight() in flight_0005_0008_
# phase_flight_gaps_v2.py / day1_avg_core_on_day2.py, adapted to work on a
# SLICE of an already-loaded dataframe instead of its own CSV. Includes the
# taxi-in-energy fix unchanged.
# =============================================================================
def build_segment(df, t_full, gspd_kt, seg_idx, lo, hi, label, log=print,
                  wheel=1):
    """Builds one flight segment from rows [lo, hi) of the day's dataframe:
    detects touchdown, reconstructs taxi/landing/taxi-in energy, and returns
    the segment dict every later stage consumes. wheel picks which brake's
    BTMS column drives the measured data and initial condition (default 1)."""
    i_td_local = find_touchdown(t_full, gspd_kt, lo, hi)
    if i_td_local is None:
        return None   # this block never lands -- not a usable flight (e.g.
                      # a trailing/leading taxi-only fragment)
    i_td = i_td_local   # absolute index
    v_td_kt = gspd_kt[i_td]
    warn = None
    if not (V_TD_PLAUSIBLE[0] <= v_td_kt <= V_TD_PLAUSIBLE[1]):
        warn = f"implausible touchdown speed ({v_td_kt:.0f} kt)"

    tm = t_full[lo:hi] - t_full[i_td]         # touchdown-relative, LOCAL frame
    gspd = gspd_kt[lo:hi]
    v_ms = gspd*KT2MS
    t_local_full = t_full[lo:hi]              # absolute seconds, same slice

    mass_clean = df['GROSS WEIGHT'].replace(0, np.nan).ffill().bfill().values[lo:hi]
    is_lb = np.nanmax(mass_clean) > MASS_LB_THRESHOLD
    mass_kg = mass_clean*(LB2KG if is_lb else 1.0)

    i_td_l = i_td - lo   # touchdown index WITHIN the local slice

    # --- taxi-out energy (delta-V) -----------------------------------------
    # Replaces G(v) self-calibration, which overestimated braking energy on
    # sustained near-constant-speed rolling (its constant was calibrated on
    # brief acceleration bursts). Delta-V attributes the full KE loss of every
    # decelerating interval to braking, nothing to flat/accelerating ones.
    # Per-interval sums telescope to each deceleration phase's own
    # v_start^2-v_end^2, so no phase-boundary bookkeeping is needed.
    taxi_mask = tm < -60.0
    fast = np.where((gspd > 50) & taxi_mask)[0]
    i_to_start = int(fast[0]) if len(fast) else int(np.where(taxi_mask)[0][-1]) if taxi_mask.any() else 0
    TAXI = np.arange(0, max(i_to_start, 1))
    iv = np.arange(TAXI[0], max(TAXI[-1], TAXI[0]+1))
    iv = iv[iv < len(v_ms)-1]
    if len(iv) < 2:
        warn = (warn + "; " if warn else "") + "too little pre-touchdown data for taxi-out energy"
        E_iv, dt_i = np.zeros(0), np.zeros(0)
    else:
        v1, v2 = v_ms[iv], v_ms[iv+1]
        dt_i = t_local_full[iv+1] - t_local_full[iv]
        mass_i = 0.5*(mass_kg[iv]+mass_kg[iv+1])
        dKE = 0.5*mass_i*(v2**2-v1**2)
        moving_m = (0.5*(v1+v2)) > 1.0*KT2MS
        E_iv = np.maximum(-dKE, 0.0)
        E_iv[~moving_m] = 0.0
    E_taxi_total = float(E_iv.sum())

    if MAX_E_TAXI_MJ_BRAKE > 0:
        _cap_J = MAX_E_TAXI_MJ_BRAKE * N_BRAKES * 1e6
        if E_taxi_total > _cap_J:
            log(f"    taxi-out energy capped: {E_taxi_total/N_BRAKES/1e6:.3f} -> "
               f"{MAX_E_TAXI_MJ_BRAKE:.3f} MJ/brake (MAX_E_TAXI_MJ_BRAKE)")
            _scale = _cap_J / E_taxi_total
            E_iv = E_iv * _scale
            E_taxi_total = _cap_J

    if len(iv) > 1:
        v_mid_iv = 0.5*(v_ms[iv[:-1]] + v_ms[iv[1:]])
        moving_idx = np.where(v_mid_iv > 1.0*KT2MS)[0]
    else:
        moving_idx = np.array([], dtype=int)
    t_move_start = float(tm[iv[moving_idx[0]]]) if len(moving_idx) else float(tm[0])

    if len(TAXI) > 0:
        i_pk = TAXI[np.argmax(gspd[TAXI])]
        after_pk = np.arange(i_pk, max(i_to_start, i_pk+1))
        after_pk = after_pk[after_pk < len(gspd)]
        i_tr = after_pk[np.argmin(gspd[after_pk])] if len(after_pk) else i_pk
        t_pulse_end = tm[i_tr]
    else:
        t_pulse_end = tm[0]

    # --- landing energy (sub-stepped force balance) -------------------------
    roll = np.where((tm >= 0) & (gspd <= 20))[0]
    i_land_end_l = int(roll[0]) if len(roll) else i_td_l
    land_idx = np.arange(i_td_l, i_land_end_l)
    if len(land_idx) < 1:
        return None   # no landing roll captured -- unusable

    rev_col = 'Thrust_Reverser_Cowl_Position'
    rev_raw_seg = (pd.to_numeric(df[rev_col], errors='coerce').fillna(0.0).values[lo:hi] > 0
                  if rev_col in df.columns else np.zeros(hi-lo, dtype=bool))

    # BACK-EXTRAPOLATION with 30s sampling,
    # a landing roll of 40-60s is only captured by 1-2 raw points, so the
    # reverser signal is under-sampled by construction -- a single active
    # sample at 35kt just means the sampler happened to catch the very end
    # of reverser activity. Physically, reversers operate from touchdown
    # until the aircraft decelerates through ~V_TH_REV (~70kt). If ANY
    # sample in the landing roll has the signal active, we therefore assume
    # reversers were active from touchdown (i_td) through the last active
    # sample, rather than trusting the exact per-sample boolean.
    land_idx_seg = np.arange(i_td_l, min(i_land_end_l+1, hi-lo))
    if len(land_idx_seg) and rev_raw_seg[land_idx_seg].any():
        _last_active = int(land_idx_seg[rev_raw_seg[land_idx_seg].nonzero()[0][-1]])
        rev_active = np.zeros(hi-lo, dtype=bool)
        rev_active[land_idx_seg[0]:_last_active+1] = True
    else:
        rev_active = rev_raw_seg

    def _landing_force_balance(idx, n_sub):
        t_s, v_s, F_s, P_s = [], [], [], []
        for i in idx:
            v1, v2 = v_ms[i], v_ms[i+1]
            m1, m2 = mass_kg[i], mass_kg[i+1]
            t1, t2 = t_local_full[i], t_local_full[i+1]
            dt_raw = t2 - t1
            a_const = (v2-v1)/dt_raw
            rev_i = rev_active[i] | rev_active[i+1]
            edges = np.linspace(0.0, 1.0, n_sub+1)
            t_e = t1 + edges*dt_raw
            v_e = v1 + edges*(v2-v1)
            m_e = m1 + edges*(m2-m1)
            for j in range(n_sub):
                v_mid = 0.5*(v_e[j]+v_e[j+1])
                m_mid = 0.5*(m_e[j]+m_e[j+1])
                F_rev = K_RT if (rev_i and v_mid > V_TH_REV) else 0.0
                F_b = max(m_mid*abs(a_const) - F_opposing(v_mid, m_mid) - F_rev, 0.0)
                t_s.append(t_e[j]); v_s.append(v_mid)
                F_s.append(F_b); P_s.append(F_b*v_mid/N_BRAKES)
        return np.array(t_s), np.array(v_s), np.array(F_s), np.array(P_s)

    t_sub_l, v_sub_l, F_sub_l, P_sub_l = _landing_force_balance(land_idx, N_SUB_LAND)
    dt_sub_l = np.diff(np.append(t_sub_l, t_local_full[land_idx[-1]+1]))
    t_land_sub = tm[i_td_l] + (t_sub_l - t_local_full[i_td_l])
    dt_land_sub, P_land_sub = dt_sub_l, P_sub_l

    E_landing_MJ = float(np.sum(P_land_sub*dt_land_sub))*N_BRAKES/1e6/N_BRAKES

    # Crude, single-point-per-interval landing energy (n_sub=1), exposed via
    # predict()'s landing_mode='crude' and this segment's landing_scale_crude
    # ratio. Not used by the default uncertainty envelope any more, but kept
    # as a manual sensitivity knob.
    _t_sub_l1, _v_sub_l1, _F_sub_l1, _P_sub_l1 = _landing_force_balance(land_idx, 1)
    _dt_sub_l1 = np.diff(np.append(_t_sub_l1, t_local_full[land_idx[-1]+1]))
    E_landing_crude_MJ = float(np.sum(_P_sub_l1*_dt_sub_l1))*N_BRAKES/1e6/N_BRAKES
    landing_scale_crude = (E_landing_crude_MJ/E_landing_MJ
                           if E_landing_MJ > 0 else 1.0)

    # --- taxi-in energy after the landing roll (delta-V, replaces G_in) ----
    post_idx = np.arange(i_land_end_l, len(t_local_full)-1)
    if len(post_idx) >= 1:
        v1p, v2p = v_ms[post_idx], v_ms[post_idx+1]
        dt_p = t_local_full[post_idx+1] - t_local_full[post_idx]
        mass_p = 0.5*(mass_kg[post_idx]+mass_kg[post_idx+1])
        dKE_p = 0.5*mass_p*(v2p**2-v1p**2)
        MOVING_P = (0.5*(v1p+v2p)) > 1.0*KT2MS
        E_iv_p = np.maximum(-dKE_p, 0.0)
        E_iv_p[~MOVING_P] = 0.0
    else:
        dt_p, E_iv_p = np.zeros(0), np.zeros(0)
    E_taxi_in_MJ = float(E_iv_p.sum())/N_BRAKES/1e6
    log(f"  {label}: touchdown {v_td_kt:.0f} kt{' (' + warn + ')' if warn else ''}, "
       f"taxi-in energy = {E_taxi_in_MJ:.2f} MJ/brake")

    def build_power_trace(taxi_scale=1.0, pulse_dur=None, landing_scale=1.0):
        t_nodes = [tm[0]-1.0]
        P_nodes = [0.0]
        if pulse_dur is None:
            for k, i in enumerate(iv):
                if E_iv[k] <= 0:
                    continue
                p = taxi_scale*E_iv[k]/(N_BRAKES*max(dt_i[k], 1e-6))
                t_nodes += [tm[i], tm[i+1]]
                P_nodes += [p, p]
        else:
            t_pulse_start = t_pulse_end - pulse_dur
            P_pulse = taxi_scale*E_taxi_total/(N_BRAKES*max(pulse_dur, 1e-3))
            eps = min(0.5, pulse_dur*0.05)
            t_nodes += [t_pulse_start-eps, t_pulse_start, t_pulse_end, t_pulse_end+eps]
            P_nodes += [0.0, P_pulse, P_pulse, 0.0]
        t_nodes += [tm[i_td_l]-1e-3]
        P_nodes += [0.0]
        for k in range(len(t_land_sub)):
            t_end_k = t_land_sub[k] + dt_land_sub[k]
            t_nodes += [t_land_sub[k], t_end_k]
            P_nodes += [P_land_sub[k]*landing_scale, P_land_sub[k]*landing_scale]
        t_nodes += [tm[i_land_end_l]+1e-3]
        P_nodes += [0.0]
        for k, i in enumerate(post_idx):
            if E_iv_p[k] <= 0:
                continue
            p = E_iv_p[k]/(N_BRAKES*max(dt_p[k], 1e-6))
            t_nodes += [tm[i], tm[i+1]]
            P_nodes += [p, p]
        t_nodes.append(tm[-1]+1.0); P_nodes.append(0.0)
        order = np.argsort(t_nodes)
        return np.array(t_nodes)[order], np.array(P_nodes)[order]

    t_stop_real_exposed = [None]   # populated by apply_module_state() below;
                                   # list-wrapped so the closure can mutate
                                   # it without a `nonlocal` on every access

    def apply_module_state():
        """Sets every m.* global this segment's own physics needs. Must be
        called again before EVERY simulation of this segment (core-only
        fit loops, alpha/H_nat_flight fit loops, or the final continuous
        prediction), since module state is shared/global and other
        segments will have overwritten it in between."""
        t_nodes, P_nodes = build_power_trace()
        m.SELECTED = label
        m._P_interp = _interp(t_nodes, P_nodes)
        m._v_interp = _interp(tm, v_ms)
        amb = df['Static Air Temperature [BNR]']
        ok_amb = amb.notna().values[lo:hi]
        amb_seg = amb.values[lo:hi]
        m._T0_interp = _interp(tm[ok_amb], amb_seg[ok_amb]+273.15,
                               fill=(amb_seg[ok_amb][0]+273.15,
                                     amb_seg[ok_amb][-1]+273.15))
        lg_col = 'LH L/G COMPRESSED' if 'LH L/G COMPRESSED' in df.columns else 'RH L/G COMPRESSED'
        lg_raw = (df[lg_col].fillna(0).astype(float).values[lo:hi] > 0.5)
        dt_med = float(np.nanmedian(np.diff(t_local_full))) if len(t_local_full) > 1 else 1.0
        min_n = max(1, int(round(LG_DEBOUNCE_S/max(dt_med, 1e-6))))
        lg_deb = lg_raw.copy()
        i = 0
        while i < len(lg_deb):
            if not lg_deb[i]:
                jj = i
                while jj < len(lg_deb) and not lg_deb[jj]:
                    jj += 1
                if (jj-i) < min_n and i > 0 and jj < len(lg_deb):
                    lg_deb[i:jj] = True
                i = jj
            else:
                i += 1
        m._on_ground_interp = _interp(tm, lg_deb.astype(float), fill=(0.0, 0.0))
        btms = df[f'Temperature Brake n°{wheel}']
        ok_b = btms.notna().values[lo:hi]
        btms_seg = btms.values[lo:hi]
        m._btms = (tm[ok_b], btms_seg[ok_b])
        v_stop = V_STOP_KT
        stopped = (tm >= 0) & (gspd <= v_stop)
        min_stop_n = max(1, int(round(STOP_DEBOUNCE_S/max(dt_med, 1e-6))))
        t_stop_real = tm[-1]
        jx = 0
        while jx < len(stopped):
            if stopped[jx]:
                k = jx
                while k < len(stopped) and stopped[k]:
                    k += 1
                if (k-jx) >= min_stop_n:
                    t_stop_real = tm[jx]
                    break
                jx = k
            else:
                jx += 1
        m._t_stop_C = tm[i_land_end_l]
        m._t_speed0_C = t_stop_real
        t_stop_real_exposed[0] = t_stop_real
        m.T_TAXI = max(t_stop_real-tm[i_land_end_l], 30.0)
        m._t_record = tm[-1]
        m._t_min_C = tm[0]
        m.T0 = float(amb_seg[ok_amb][0])+273.15
        # FAN_ON/GROUND_FAN_ON are set once by configure_thermal_module()
        # (default False/False), never hardcoded per segment here.
        pre = np.where(tm < 0)[0]
        og = lg_deb[pre]
        falls = np.where(og[:-1] & ~og[1:])[0] if len(og) > 1 else np.array([], dtype=int)
        m._t_liftoff = float(tm[pre[falls[-1]+1]]) if len(falls) else None
        m._alt_interp = None
        m._t_app_start = -m.APPROACH_FALLBACK_S
        if DISABLE_RETRACTION_WINDOW:
            m._t_liftoff = None
        return t_nodes, P_nodes

    apply_module_state()
    T_INIT_FULL = m.btms_at(m.t_min())
    T_INIT_TD = m.btms_at(0.0)
    tm_b, Tm_b = m._btms
    pos = np.where(build_power_trace(1.0, None)[1] > 0.0)[0]
    t_nodes0, _ = build_power_trace(1.0, None)
    t_energy_start = float(t_nodes0[pos[0]]) if len(pos) else float(t_move_start)

    return dict(label=label, seg_idx=seg_idx, warn=warn, i_td=i_td,
               touchdown_time_s=float(t_full[i_td]),
               apply_module_state=apply_module_state,
               build_power_trace=build_power_trace,
               T_INIT_FULL=T_INIT_FULL, T_INIT_TD=T_INIT_TD,
               tm_min=tm[0], tm_max=tm[-1],
               tm_b=tm_b, Tm_b=Tm_b, v_td_kt=v_td_kt,
               t_move_start=t_move_start, t_energy_start=t_energy_start,
               t_taxi_accel=tm[i_to_start],
               t_land_end=tm[i_land_end_l],
               t_full_stop=t_stop_real_exposed[0],
               E_landing_MJ=E_landing_MJ, E_taxi_in_MJ=E_taxi_in_MJ,
               landing_scale_crude=landing_scale_crude)


# =============================================================================
# 5. CALIBRATION -- core (7 params) + alpha/H_nat_flight, both taxi shapes
# tried and compared, per-flight. Ported from day1_avg_core_on_day2.py's
# own Part A.
# =============================================================================
def fit_core(seg, log=print):
    seg['apply_module_state']()
    tm_b, Tm_b, T_init = seg['tm_b'], seg['Tm_b'], seg['T_INIT_TD']
    # Scores post-touchdown points only, so it uses m.run() (touchdown-only),
    # not m.run_full() -- the latter would re-integrate hours of cruise on
    # every evaluation. T_init is therefore T_INIT_TD, what run() expects.
    fit_t, fit_T = tm_b[tm_b >= 0], Tm_b[tm_b >= 0]
    # Anchor at the last post-touchdown measured point, weighted
    # ANCHOR_WEIGHT_CORE -- synced to btem_pipeline.py's run_core_calibration.
    anchor_t, anchor_T = float(fit_t[-1]), float(fit_T[-1])

    def obj(logx):
        xr = _safe_exp(logx)
        xp = np.clip(xr, LO, HI)
        pen = 50.0*np.sum(np.abs(np.log(xr)-np.log(xp)))
        set_core(xp)
        try:
            t1, Tc1, _ = m.run(T_init)
            model_td = np.interp(fit_t, t1, Tc1[m.I_SENS])
            se = (model_td - fit_T)**2
            sa = (float(np.interp(anchor_t, t1, Tc1[m.I_SENS])) - anchor_T)**2
            rmse = float(np.sqrt((se.sum() + ANCHOR_WEIGHT_CORE*sa)
                                 / (len(se) + ANCHOR_WEIGHT_CORE)))
        except Exception:
            return 1e6+pen
        return rmse+pen

    x0 = np.array([BASE[k] for k in NAMES])
    best_x, best_v = np.log(x0), obj(np.log(x0))
    for _ in range(4):
        res = minimize(obj, best_x, method='Nelder-Mead',
                       options=dict(maxfev=900, xatol=1e-4, fatol=1e-3, adaptive=True))
        if res.fun < best_v - 1e-4:
            best_v, best_x = res.fun, res.x
        else:
            break
    X1 = np.clip(_safe_exp(best_x), LO, HI)
    core = {k: float(v) for k, v in zip(NAMES, X1)}
    log(f"  core fit on {seg['label']}: RMSE={best_v:.2f} C, "
       + ", ".join(f"{k}={core[k]:.4g}" for k in NAMES))
    return core


def fit_alpha_h_nat_flight(seg, core, log=print):
    seg['apply_module_state']()
    apply_core(core)
    tm_b, Tm_b = seg['tm_b'], seg['Tm_b']
    # Window, anchor, and taxi-shape choice synced to btem_pipeline.py's
    # own run_pretouchdown_calibration() never does (it always fits the
    # pulse shape). All four are now the same fit as that function's).
    T_START = max(seg['t_move_start'] - 60.0, float(tm_b[0]))
    T_END = min(seg['t_taxi_accel'] + 1800.0, -60.0)
    if T_END <= T_START:
        T_START, T_END = seg['tm_min'], min(-30.0, seg['tm_max'])
    T_INIT_SEG = float(Tm_b[np.argmin(np.abs(tm_b - T_START))])
    w = (tm_b >= T_START) & (tm_b <= T_END)
    tm_w, Tm_w = tm_b[w], Tm_b[w]
    bpt = seg['build_power_trace']

    _pre = np.where(tm_b < 0.0)[0]
    _ANCHOR_TARGET = -8 * 60.0
    _ai = _pre[np.argmin(np.abs(tm_b[_pre] - _ANCHOR_TARGET))]
    anchor_t, anchor_T = float(tm_b[_ai]), float(Tm_b[_ai])
    T_END_ANCHOR = min(anchor_t + 1.0, -1e-3)

    def score3(xp):
        alpha, hnf, pdur = xp
        tn, pn = bpt(1.0, pdur, 1.0)
        with _flight_knobs(alpha, hnf), _module_attr('_P_interp', _interp(tn, pn)):
            try:
                t, Tc = m.run_flight_window(T_INIT_SEG, T_START, T_END_ANCHOR)
                se = (np.interp(tm_w, t, Tc[m.I_SENS]) - Tm_w)**2
                sa = (float(np.interp(anchor_t, t, Tc[m.I_SENS])) - anchor_T)**2
                return float(np.sqrt((se.sum() + ANCHOR_WEIGHT_PRE*sa)
                                     / (len(se) + ANCHOR_WEIGHT_PRE)))
            except Exception:
                return 1e6

    seeds = [0.5, 1.0, 2.0, 4.0, 7.0]
    p3_lo, p3_hi = np.array([0.1, 1.0, 10.0]), np.array([10.0, 25.0, 1800.0])

    def obj3(logx):
        return score3(np.clip(_safe_exp(logx), p3_lo, p3_hi))

    bx, bv = None, np.inf
    for i, a0 in enumerate(seeds):
        cx = np.log(np.array([a0, m.h_nat, 30.0]))
        cv = obj3(cx)
        for _ in range(4):
            res = minimize(obj3, cx, method='Nelder-Mead',
                           options=dict(maxfev=300, xatol=1e-3, fatol=1e-2, adaptive=True))
            if res.fun < cv - 1e-3:
                cv, cx = res.fun, res.x
            else:
                break
        log(f"    start {i+1}/{len(seeds)} (alpha0={a0:.1f}): RMSE={cv:.2f} C")
        if cv < bv - 1e-3:
            bv, bx = cv, cx

    ALPHA, H_NAT, PULSE = (float(v) for v in np.clip(np.exp(bx), p3_lo, p3_hi))
    log(f"  alpha/H_nat_flight fit on {seg['label']}: alpha={ALPHA:.2f}, "
       f"H_nat_flight={H_NAT:.2f} W/m2K, pulse_dur={PULSE:.0f}s, RMSE={bv:.2f} C")
    return dict(alpha=ALPHA, h_nat_flight=H_NAT, taxi_mode='pulse', pulse_dur=PULSE)


# =============================================================================
# 6. DEFAULT CALIBRATION -- average of two existing calibration bundles
# =============================================================================
def load_calibration_bundle(source):
    """Reads a calibration JSON from ANY of this project's own calibration
    scripts and extracts (core, alpha, h_nat_flight, taxi_mode, pulse_dur)
    with the same keys wherever possible, falling back to neutral values
    (alpha=1.0, honest shape) for a core-only file that never fit those.
    source may be a path string or a file-like object (e.g. Streamlit's
    UploadedFile)."""
    name = getattr(source, 'name', str(source))
    if hasattr(source, 'read'):
        raw = json.load(source)
    else:
        with open(source) as f:
            raw = json.load(f)
    if 'core' not in raw or not all(k in raw['core'] for k in NAMES):
        raise PipelineError(f"{name} does not contain a full 7-parameter core "
                         f"under 'core' -- not a calibration bundle this "
                         f"script recognizes.")
    core = {k: float(raw['core'][k]) for k in NAMES}
    alpha = float(raw.get('alpha', 1.0))
    h_nat_flight = float(raw.get('h_nat_flight', raw.get('H_nat_flight', m.h_nat)))
    taxi_mode = raw.get('taxi_mode', 'honest')
    pdur_raw = raw.get('pulse_dur', raw.get('taxi_pulse_dur_s'))
    pulse_dur = float(pdur_raw) if (taxi_mode == 'pulse' and pdur_raw) else None
    return dict(core=core, alpha=alpha, h_nat_flight=h_nat_flight,
               taxi_mode=taxi_mode, pulse_dur=pulse_dur, source=name)


def average_bundles(bundle_a, bundle_b, log=print):
    core = {k: 0.5*(bundle_a['core'][k]+bundle_b['core'][k]) for k in NAMES}
    alpha = 0.5*(bundle_a['alpha']+bundle_b['alpha'])
    h_nat_flight = 0.5*(bundle_a['h_nat_flight']+bundle_b['h_nat_flight'])
    # pulse_dur only averages if BOTH bundles used pulse mode; otherwise
    # honest shape (the safer, less parametrized default)
    if bundle_a['taxi_mode'] == 'pulse' and bundle_b['taxi_mode'] == 'pulse':
        taxi_mode, pulse_dur = 'pulse', 0.5*(bundle_a['pulse_dur']+bundle_b['pulse_dur'])
    else:
        taxi_mode, pulse_dur = 'honest', None
    log(f"  default calibration = average of:\n"
       f"    A: {bundle_a.get('source', '(manual input)')}\n"
       f"    B: {bundle_b.get('source', '(manual input)')}")
    log(f"  averaged: alpha={alpha:.2f}, H_nat_flight={h_nat_flight:.2f} "
       f"W/m2K, taxi_mode={taxi_mode}"
       + (f", pulse_dur={pulse_dur:.0f}s" if pulse_dur is not None else ""))
    return dict(core=core, alpha=alpha, h_nat_flight=h_nat_flight,
               taxi_mode=taxi_mode, pulse_dur=pulse_dur)


# =============================================================================
# 7. CONTINUOUS PREDICTION across the selected flight segments
# =============================================================================
def predict(segments, calib, alpha_override=None, landing_mode='refined',
           gap_model='h_nat', log=print):
    """alpha_override replaces calib['alpha'] for every segment EXCEPT the
    first, which always keeps its own alpha so it stays comparable across
    sensitivity checks. Not used by the default uncertainty envelope.

    landing_mode picks the landing-energy reconstruction: 'refined' (default,
    sub-stepped force balance) or 'crude' (single-point-per-interval, via each
    segment's landing_scale_crude). Unlike alpha this applies to EVERY segment
    -- it's a modeling choice, not something fitted on one flight.

    gap_model fills inter-flight gaps:
      'h_nat'    (default) continuous physics -- neutral h_nat from touchdown
                 to the next flight's t_move_start, then H_NAT_FLIGHT resumes.
      'exp_gap1' Newton cooling, T = T_amb + (T_peak-T_amb)*exp(-k*(t-t_peak)),
                 with k/T_amb fitted ONCE on the day's first gap
                 (fit_exp_gap1) and reused for every later gap -- only T_peak
                 varies. Physics resumes at the next flight's recording start
                 (tm_min), not t_move_start, so one smooth curve covers the
                 whole parked+taxi-out period."""
    apply_core(calib['core'])
    t_all, T_all = [], []
    flight_report, gap_report = [], []
    T_init_next = None
    t_start_next = None   # only used by gap_model='exp_gap1'
    _exp_k, _exp_T_amb = None, None   # fitted once, on the first gap

    for idx, seg in enumerate(segments):
        seg['apply_module_state']()
        apply_core(calib['core'])
        alpha = calib['alpha'] if (idx == 0 or alpha_override is None) else alpha_override
        m.P_FLIGHT_SCALE = alpha
        m.C_TUBE_FLIGHT_MULT = 1.0
        m.H_NAT_FLIGHT = calib['h_nat_flight']
        landing_scale = seg['landing_scale_crude'] if landing_mode == 'crude' else 1.0
        t_nodes, P_nodes = seg['build_power_trace'](1.0, calib['pulse_dur'], landing_scale)
        m._P_interp = _interp(t_nodes, P_nodes)

        T_init = seg['T_INIT_FULL'] if idx == 0 else T_init_next
        if idx == 0:
            t_local, Tc, _ = m.run_full(T_init)
        elif gap_model == 'h_nat':
            # The h_nat -> H_NAT_FLIGHT boundary is t_move_start (ground speed
            # leaves 0), NOT t_energy_start (first braking). Braking only ever
            # starts once already moving, so using it would extend the "parked"
            # window into real taxiing and bias the gap cooling low: push-back
            # through touchdown is 'taxi'-like on movement alone.
            with _module_attr('_btms', None):
                t_local, Tc, _ = m.run_full(T_init, t_start=seg['t_move_start'])
        else:   # gap_model == 'exp_gap1'
            # Resumes at the start of THIS flight's own recording (not
            # t_move_start) -- the exponential covers the whole gap on
            # its own, so physics only needs to pick up from where the
            # exponential handed off, at the very start of this CSV.
            with _module_attr('_btms', None):
                t_local, Tc, _ = m.run_full(T_init, t_start=t_start_next)
        if gap_model == 'h_nat':
            m.H_NAT_FLIGHT = m.h_nat   # neutral for the gap that follows (Decouverte n11)

        t_abs = t_local + seg['touchdown_time_s']
        t_all.append(t_abs); T_all.append(Tc[m.I_SENS])

        tm_b, Tm_b = seg['tm_b'], seg['Tm_b']
        sel = tm_b >= 0
        if sel.sum():
            model_td = np.interp(tm_b[sel], t_local, Tc[m.I_SENS])
            rmse_f = float(np.sqrt(np.mean((model_td-Tm_b[sel])**2)))
            bias_f = float(np.mean(model_td-Tm_b[sel]))
        else:
            rmse_f = bias_f = float('nan')
        flight_report.append(dict(flight=seg['label'], n=int(sel.sum()),
                                  rmse=rmse_f, bias=bias_f, prediction=(idx > 0)))

        if idx < len(segments)-1:
            nxt = segments[idx+1]
            if gap_model == 'h_nat':
                # Same fix as above, same boundary, applied symmetrically to
                # where the gap (h_nat) ends and the next flight's own
                # H_NAT_FLIGHT window begins.
                t_gap_end_abs = nxt['touchdown_time_s'] + nxt['t_move_start']
                t_gap_start_local = t_local[-1]
                t_gap_end_local = t_gap_end_abs - seg['touchdown_time_s']
                if t_gap_end_local > t_gap_start_local:
                    y0 = Tc[:, -1] + 273.15
                    sol = solve_ivp(m.rhs, (t_gap_start_local, t_gap_end_local), y0,
                                    args=('flight',), method='LSODA', max_step=60.0,
                                    rtol=1e-6, atol=1e-3, dense_output=True)
                    tm_b_nxt_abs = nxt['tm_b'] + nxt['touchdown_time_s']
                    sel_g = ((tm_b_nxt_abs >= seg['touchdown_time_s']+t_gap_start_local)
                            & (tm_b_nxt_abs <= t_gap_end_abs))
                    t_tgt_local = tm_b_nxt_abs[sel_g] - seg['touchdown_time_s']
                    T_tgt = nxt['Tm_b'][sel_g]
                    if len(t_tgt_local):
                        pred = sol.sol(t_tgt_local)[m.I_SENS]-273.15
                        rmse_g = float(np.sqrt(np.mean((pred-T_tgt)**2)))
                        bias_g = float(np.mean(pred-T_tgt))
                    else:
                        rmse_g = bias_g = float('nan')
                    gap_report.append(dict(gap=f"{seg['label']}->{nxt['label']}",
                                           n=len(t_tgt_local), rmse=rmse_g, bias=bias_g))
                    t_gap = np.linspace(t_gap_start_local, t_gap_end_local,
                                        max(int((t_gap_end_local-t_gap_start_local)/30), 5))
                    Tc_gap = sol.sol(t_gap)
                    t_all.append(t_gap+seg['touchdown_time_s']); T_all.append(Tc_gap[m.I_SENS]-273.15)
                    T_init_next = float(sol.y[m.I_SENS, -1]-273.15)
                else:
                    T_init_next = float(Tc[m.I_SENS, -1])

            else:   # gap_model == 'exp_gap1'
                post_mask = t_local >= 0
                i_pk = int(np.argmax(Tc[m.I_SENS][post_mask]))
                peak_T = float(Tc[m.I_SENS][post_mask][i_pk])
                peak_t_abs = float(t_abs[post_mask][i_pk])

                if idx == 0:
                    # Anchor the k/T_amb REGRESSION on the MEASURED peak, not
                    # the model's simulated one (as btem_pipeline.py's _fit_exp
                    # does): a peak-timing mismatch otherwise pulls still-rising
                    # points into the decay window and flattens k. The
                    # PROJECTION still starts from the model's own peak.
                    seg0 = segments[0]
                    post0_meas = seg0['tm_b'] >= 0
                    i_pk0_meas = int(np.argmax(seg0['Tm_b'][post0_meas]))
                    peak_t_abs_measured = float(
                        seg0['tm_b'][post0_meas][i_pk0_meas]
                        + seg0['touchdown_time_s'])
                    _exp_k, _exp_T_amb = fit_exp_gap1(
                        segments[0], segments[1], peak_t_abs_measured, log=log)

                rec_start_own = nxt['tm_min']
                rec_start_abs = nxt['touchdown_time_s'] + rec_start_own
                dt_s = rec_start_abs - peak_t_abs
                T_init_next = _exp_T_amb + (peak_T - _exp_T_amb) * np.exp(-_exp_k * dt_s)
                t_start_next = rec_start_own

                # Fill the gap visually with the exponential curve itself
                # (matches this pipeline's convention of always returning
                # one continuous (t_all, T_all) trace, gap included).
                t_gap_abs = np.linspace(peak_t_abs, rec_start_abs,
                                        max(int((rec_start_abs-peak_t_abs)/30), 5))
                T_gap = _exp_T_amb + (peak_T - _exp_T_amb) * np.exp(
                    -_exp_k * (t_gap_abs - peak_t_abs))
                t_all.append(t_gap_abs); T_all.append(T_gap)

                # Gap-report RMSE/bias against nxt's own measured points
                # that fall within the gap window.
                tm_b_nxt_abs = nxt['tm_b'] + nxt['touchdown_time_s']
                sel_g = ((tm_b_nxt_abs >= peak_t_abs)
                        & (tm_b_nxt_abs <= rec_start_abs))
                T_tgt = nxt['Tm_b'][sel_g]
                if len(T_tgt):
                    pred = _exp_T_amb + (peak_T - _exp_T_amb) * np.exp(
                        -_exp_k * (tm_b_nxt_abs[sel_g] - peak_t_abs))
                    rmse_g = float(np.sqrt(np.mean((pred-T_tgt)**2)))
                    bias_g = float(np.mean(pred-T_tgt))
                else:
                    rmse_g = bias_g = float('nan')
                gap_report.append(dict(gap=f"{seg['label']}->{nxt['label']}",
                                       n=int(sel_g.sum()), rmse=rmse_g, bias=bias_g))

    t_all = np.concatenate(t_all); T_all = np.concatenate(T_all)
    order = np.argsort(t_all)
    return t_all[order], T_all[order], flight_report, gap_report


# =============================================================================
# 8. REPORTING + PLOT
# =============================================================================
def print_report(flight_report, gap_report):
    print(f"\n  {'segment':22s}{'n':>5}{'RMSE':>9}{'bias':>9}")
    for r in flight_report:
        tag = ' (prediction)' if r['prediction'] else ' (I.C. flight)'
        print(f"  {r['flight']+tag:22s}{r['n']:5d}{r['rmse']:9.1f}{r['bias']:+9.1f}")
    for g in gap_report:
        print(f"  {g['gap']:22s}{g['n']:5d}{g['rmse']:9.1f}{g['bias']:+9.1f}")
    pred_rmse = [r['rmse'] for r in flight_report if r['prediction'] and not np.isnan(r['rmse'])]
    pred_rmse += [g['rmse'] for g in gap_report if not np.isnan(g['rmse'])]
    if pred_rmse:
        print(f"  combined RMSE over genuine predictions: "
             f"{float(np.sqrt(np.mean(np.array(pred_rmse)**2))):.1f} C")


def _time_bounds(segments):
    """0 = first segment's own departure gate, 1 = last segment's own
    arrival gate -- the shared time-normalization convention for every
    figure in this file. Shared by build_figure/build_comparison_figure
    so the two can never drift apart on what [0,1] means."""
    t0 = float(segments[0]['touchdown_time_s'] + segments[0]['tm_min'])
    t1 = float(segments[-1]['touchdown_time_s'] + segments[-1]['tm_max'])
    return t0, t1


_FLIGHT_BAND_COLORS = ['#dbe9f6', '#fdf1db']   # light blue / light peach, alternating


def _draw_flight_markers(ax, segments, xn):
    """Shades each flight's ACTIVE span -- first movement (t_move_start) to
    sustained full stop (t_full_stop) -- with a boundary line and a bold
    flight-name label. Deliberately NOT tm_min/tm_max: a flight's recording can
    start before its take-off or run past its full stop, overlapping the
    parking gap either side, so shading by raw recording extent bleeds straight
    through the gap with no visible break.

    Take-off ('T/O', blue dotted) and touchdown ('TD', black dashed) are
    separate EVENTS within the span. xn maps absolute time to the axes' x
    coordinate (identity, or [0,1] for build_figure); labels use the x-axis
    transform so they sit correctly whether y is normalized or in degrees C."""
    trans = ax.get_xaxis_transform()
    for i, s in enumerate(segments):
        t_start = xn(s['touchdown_time_s'] + s['t_move_start'])
        t_end = xn(s['touchdown_time_s'] + s['t_full_stop'])

        ax.axvspan(t_start, t_end, color=_FLIGHT_BAND_COLORS[i % 2],
                  alpha=0.7, lw=0, zorder=0)
        ax.axvline(t_start, color='0.4', ls='-', lw=1.0, zorder=1)
        ax.axvline(t_end, color='0.4', ls='-', lw=1.0, zorder=1)
        ax.text((t_start+t_end)/2, 0.96, s['label'], ha='center', va='top',
               fontsize=9, fontweight='bold', color='0.2', transform=trans,
               bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5))

        t_takeoff = xn(s['touchdown_time_s'] + s['t_taxi_accel'])
        ax.axvline(t_takeoff, color='steelblue', ls=':', lw=1.1, zorder=1)
        ax.text(t_takeoff, 0.02, 'T/O', rotation=90, va='bottom', ha='right',
               fontsize=7, color='steelblue', transform=trans)

        t_td = xn(s['touchdown_time_s'])
        ax.axvline(t_td, color='k', ls='--', lw=0.9, zorder=1)
        ax.text(t_td, 0.02, 'TD', rotation=90, va='bottom', ha='left',
               fontsize=7, color='0.2', transform=trans)


def _split_curve_by_mask(t, T, mask_t, mask):
    """Splits (t, T) into contiguous (t_seg, T_seg, is_drift) runs according
    to a boolean `mask` sampled on its own, possibly coarser grid `mask_t` --
    used to recolour the line itself where u_c > U_tol, instead of axvspan
    shading that was unreadable under the uncertainty band. Nearest-neighbour
    lookup, not interpolation, since drift is a step function. Consecutive
    runs share their boundary sample so the drawn line has no gaps at the
    colour change."""
    if len(mask_t) == 0 or not mask.any():
        return [(t, T, False)]
    idx = np.clip(np.searchsorted(mask_t, t), 0, len(mask_t)-1)
    idx_prev = np.clip(idx-1, 0, len(mask_t)-1)
    left_closer = np.abs(mask_t[idx_prev]-t) <= np.abs(mask_t[idx]-t)
    idx_final = np.where(left_closer, idx_prev, idx)
    drift_on_t = mask[idx_final]

    segs = []
    start = 0
    cur = drift_on_t[0]
    for i in range(1, len(drift_on_t)):
        if drift_on_t[i] != cur:
            segs.append((t[start:i+1], T[start:i+1], bool(cur)))
            start = i
            cur = drift_on_t[i]
    segs.append((t[start:], T[start:], bool(cur)))
    return segs


def overall_fit_metric(t_all, T_all, segments, normalize=False):
    """Model-vs-measurement metric over every measured point in `segments`,
    for display on the prediction figure.

    Returns (value, label) or (None, None) if the two never overlap in time.
    RMSE [degC] when normalize is False, MAPE [%] when it is True: an RMSE in
    degrees is what you want to read off an absolute plot, while a normalized
    plot has no degree axis to read it against, and MAPE is dimensionless.
    Both are computed on the PHYSICAL values regardless -- normalizing first
    would strip the RMSE of its unit and rescale the MAPE denominator.
    """
    tm_all = np.concatenate([s['tm_b']+s['touchdown_time_s'] for s in segments])
    Tm_all = np.concatenate([s['Tm_b'] for s in segments])
    sel = (tm_all >= t_all.min()) & (tm_all <= t_all.max())
    if sel.sum() == 0:
        return None, None
    resid = np.interp(tm_all[sel], t_all, T_all) - Tm_all[sel]
    if normalize:
        denom = np.clip(np.abs(Tm_all[sel]), 1.0, None)
        return float(np.mean(np.abs(resid/denom))*100.0), 'MAPE'
    return float(np.sqrt(np.mean(resid**2))), 'RMSE'


def build_figure(t_all, T_all, segments, env=None, normalize=True):
    """Returns the prediction-vs-measured figure. Does NOT call plt.show() --
    the CLI does that (see plot_result); Streamlit calls st.pyplot(fig). env,
    if given, adds the shaded combined-uncertainty band, the tolerance band
    (T_model +/- U_tol) and a marker wherever u_c(Tx) > U_tol.

    normalize=True (default) maps time and BTMS to [0,1] (0 = first selected
    flight's departure gate, 1 = last one's arrival gate; 0 = coldest,
    1 = hottest). False keeps seconds and degrees C, for reading real
    durations and temperatures off the plot. Only the axis mapping and labels
    differ between the two."""
    tm_all = np.concatenate([s['tm_b']+s['touchdown_time_s'] for s in segments])
    Tm_all = np.concatenate([s['Tm_b'] for s in segments])
    t0, t1 = _time_bounds(segments)

    # t_all/T_all come from the FULL chronological chain
    # (needed for correct gap continuity, see run_prediction), which can
    # extend well before t0 when the selection starts partway through the
    # day (a range or a single flight). Clip to the selected span so only
    # the selected flight(s) actually get plotted/normalized -- otherwise
    # earlier, unselected flights spill outside [0,1].
    sel_mask = (t_all >= t0) & (t_all <= t1)
    t_all, T_all = t_all[sel_mask], T_all[sel_mask]
    if env is not None:
        env_mask = (env['t'] >= t0) & (env['t'] <= t1)
        env = dict(env, t=env['t'][env_mask], lo=env['lo'][env_mask],
                  hi=env['hi'][env_mask], tol_lo=env['tol_lo'][env_mask],
                  tol_hi=env['tol_hi'][env_mask], drift=env['drift'][env_mask])

    if normalize:
        temp_pool = [Tm_all, T_all]
        if env is not None:
            temp_pool += [env['lo'], env['hi'], env['tol_lo'], env['tol_hi']]
        T0 = float(min(float(np.nanmin(a)) for a in temp_pool))
        T1 = float(max(float(np.nanmax(a)) for a in temp_pool))

        def xn(t):
            return (np.asarray(t, dtype=float)-t0)/(t1-t0)

        def Tn(T):
            return (np.asarray(T, dtype=float)-T0)/(T1-T0)
    else:
        def xn(t):   # seconds -> hours: a day of flights is unreadable in seconds
            return np.asarray(t, dtype=float)/3600.0

        def Tn(T):
            return np.asarray(T, dtype=float)

    fig, ax = plt.subplots(figsize=(14, 6))
    _draw_flight_markers(ax, segments, xn)
    if env is not None:
        ax.fill_between(xn(env['t']), Tn(env['lo']), Tn(env['hi']), color='green',
                        alpha=0.15, lw=0, label='combined uncertainty (u_BDT + u_BTS)')
        # Tolerance limit in dotted orange, turning solid purple wherever
        # u_c(Tx) > U_tol. The predicted curve stays plain green -- only the
        # tolerance line flags a breach.
        if env['drift'].any():
            first_drift, first_tol = True, True
            for tol_key in ('tol_hi', 'tol_lo'):
                segs_tol = _split_curve_by_mask(env['t'], env[tol_key], env['t'], env['drift'])
                for t_seg, tol_seg, is_drift in segs_tol:
                    if len(t_seg) < 2:
                        continue
                    if is_drift:
                        ax.plot(xn(t_seg), Tn(tol_seg), ':', color='purple', lw=1.6, zorder=5,
                               label='tolerance exceeded (u_c > U_tol)' if first_drift else None)
                        first_drift = False
                    else:
                        ax.plot(xn(t_seg), Tn(tol_seg), ':', color='darkorange', lw=1.2,
                               label='tolerance limit (T_model +/- U_tol)' if first_tol else None)
                        first_tol = False
        else:
            ax.plot(xn(env['t']), Tn(env['tol_hi']), ':', color='darkorange', lw=1.2,
                   label='tolerance limit (T_model +/- U_tol)')
            ax.plot(xn(env['t']), Tn(env['tol_lo']), ':', color='darkorange', lw=1.2)
    ax.plot(xn(tm_all), Tn(Tm_all), 'r.', ms=2, alpha=0.5, label='measured')
    ax.plot(xn(t_all), Tn(T_all), 'g-', lw=1.3, label='predicted')
    val, lbl = overall_fit_metric(t_all, T_all, segments, normalize=normalize)
    if normalize:
        suffix = f"  --  {lbl} = {val:.1f} %" if val is not None else ""
        ax.set(xlabel=f"Normalized time [0-1] (0 = {segments[0]['label']}'s "
                     f"departure gate, 1 = {segments[-1]['label']}'s arrival gate)",
              ylabel='Normalized BTEM [0-1] (0 = coldest, 1 = hottest)',
              title='BTEM prediction (normalized)' + suffix)
    else:
        suffix = f"  --  {lbl} = {val:.1f} \u00b0C" if val is not None else ""
        ax.set(xlabel='Time [h] (absolute, from segmentation)',
              ylabel='BTEM [\u00b0C]',
              title='BTEM prediction (absolute)' + suffix)
    ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def build_uncertainty_error_figure(t_all, T_all, segments, env, normalize=False):
    """Three-panel diagnostic figure:
      top    predicted vs measured BTMS, combined-uncertainty band and
             tolerance limit (purple wherever exceeded);
      middle the combined uncertainty u_c(t) against U_tol, so its magnitude
             over time is visible, not just the band width;
      bottom relative error [%] at every measured point.
    Drift points are marked the same way in all three. Every panel also shows
    each flight's span, take-off and touchdown (see _draw_flight_markers).

    normalize=False (default) keeps absolute units. True maps time and each
    panel's y-quantity to its own [0,1] range (BTMS pooled across measured/
    predicted/uncertainty/tolerance as build_figure does; u_c/U_tol from 0 to
    the larger max; relative error pooled with the +/-20% reference lines) --
    for comparing the SHAPE of drift across days without one absolute scale
    dominating.

    Requires env (see build_uncertainty_envelope); raises PipelineError if
    None, since there is nothing to draw without it."""
    if env is None:
        raise PipelineError(
            "build_uncertainty_error_figure needs the combined-uncertainty "
            "envelope (run_prediction(..., include_envelope=True)) -- "
            "nothing to plot without it.")

    tm_all = np.concatenate([s['tm_b']+s['touchdown_time_s'] for s in segments])
    Tm_all = np.concatenate([s['Tm_b'] for s in segments])
    t0, t1 = _time_bounds(segments)

    sel_mask = (t_all >= t0) & (t_all <= t1)
    t_all, T_all = t_all[sel_mask], T_all[sel_mask]
    env_mask = (env['t'] >= t0) & (env['t'] <= t1)
    env = dict(env, t=env['t'][env_mask], lo=env['lo'][env_mask],
              hi=env['hi'][env_mask], u_c=env['u_c'][env_mask],
              tol_lo=env['tol_lo'][env_mask], tol_hi=env['tol_hi'][env_mask],
              drift=env['drift'][env_mask])
    U_TOL_C = AIRCRAFT_DEFAULTS['u_tol_c']   # only used as a flat reference
                                              # line in the middle panel
    u_tol_flat = np.full_like(env['t'], U_TOL_C)

    # Relative error at measured points, computed up front (needed either
    # way, and its own pool is needed before plotting when normalizing).
    sel = np.abs(Tm_all) > 5.0
    tb_sel = tm_all[sel]
    err_pct = np.array([])
    is_drift_pt = np.array([], dtype=bool)
    if sel.sum():
        T_pred_at_meas = np.interp(tb_sel, t_all, T_all)
        err_pct = (T_pred_at_meas - Tm_all[sel]) / Tm_all[sel] * 100.0
        idx = np.clip(np.searchsorted(env['t'], tb_sel), 0, len(env['t'])-1)
        idx_prev = np.clip(idx-1, 0, len(env['t'])-1)
        left_closer = np.abs(env['t'][idx_prev]-tb_sel) <= np.abs(env['t'][idx]-tb_sel)
        idx_final = np.where(left_closer, idx_prev, idx)
        is_drift_pt = env['drift'][idx_final]

    if normalize:
        def xn(t):
            return (np.asarray(t, dtype=float)-t0)/(t1-t0)

        btms_pool = [Tm_all, T_all, env['lo'], env['hi'],
                    env['tol_lo'], env['tol_hi']]
        B0 = float(min(float(np.nanmin(a)) for a in btms_pool))
        B1 = float(max(float(np.nanmax(a)) for a in btms_pool))

        def btms_n(T):
            return (np.asarray(T, dtype=float)-B0)/(B1-B0)

        U1 = float(max(float(np.nanmax(env['u_c'])), U_TOL_C))

        def unc_n(u):
            return np.asarray(u, dtype=float)/U1

        err_pool = (np.concatenate([err_pct, [20., -20.]]) if len(err_pct)
                   else np.array([20., -20.]))
        E0, E1 = float(np.nanmin(err_pool)), float(np.nanmax(err_pool))

        def err_n(e):
            return (np.asarray(e, dtype=float)-E0)/(E1-E0)

        btms_label = 'Normalized BTEM [0-1] (0 = coldest, 1 = hottest)'
        unc_label = 'Normalized u_c(t) [0-1]\n(1 = larger of max u_c, U_tol)'
        err_label = 'Normalized rel. error [0-1]\n(model-meas)/meas'
        xlabel = (f"Normalized time [0-1] (0 = {segments[0]['label']}'s "
                 f"departure gate, 1 = {segments[-1]['label']}'s arrival gate)")
    else:
        def xn(t):   # seconds -> hours, as in build_figure
            return np.asarray(t, dtype=float)/3600.0

        def btms_n(T):
            return np.asarray(T, dtype=float)

        def unc_n(u):
            return np.asarray(u, dtype=float)

        def err_n(e):
            return np.asarray(e, dtype=float)

        btms_label = 'BTEM [\u00b0C]'
        unc_label = 'Combined uncertainty\nu_c(t) [\u00b0C]'
        err_label = 'Rel. error [%]\n(model-meas)/meas'
        xlabel = 'Time [h] (absolute)'

    fig, (ax, axu, axe) = plt.subplots(
        3, 1, figsize=(14, 11), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1.5, 1.5]))

    # --- top: prediction + uncertainty band + tolerance + drift ---
    _draw_flight_markers(ax, segments, xn)
    ax.fill_between(xn(env['t']), btms_n(env['lo']), btms_n(env['hi']), color='green',
                    alpha=0.15, lw=0, label='combined uncertainty (u_BDT + u_BTS)')
    # Tolerance limit shown in its normal dotted orange, turning solid
    # purple wherever it is actually exceeded (u_c(Tx) > U_tol) -- the
    # predicted curve itself stays plain green throughout.
    if env['drift'].any():
        first_drift, first_tol = True, True
        for tol_key in ('tol_hi', 'tol_lo'):
            segs_tol = _split_curve_by_mask(env['t'], env[tol_key], env['t'], env['drift'])
            for t_seg, tol_seg, is_drift in segs_tol:
                if len(t_seg) < 2:
                    continue
                if is_drift:
                    ax.plot(xn(t_seg), btms_n(tol_seg), ':', color='purple', lw=1.8, zorder=5,
                           label='tolerance exceeded (u_c > U_tol)' if first_drift else None)
                    first_drift = False
                else:
                    ax.plot(xn(t_seg), btms_n(tol_seg), ':', color='darkorange', lw=1.2,
                           label='tolerance limit (T_model +/- U_tol)' if first_tol else None)
                    first_tol = False
    else:
        ax.plot(xn(env['t']), btms_n(env['tol_hi']), ':', color='darkorange', lw=1.2,
               label='tolerance limit (T_model +/- U_tol)')
        ax.plot(xn(env['t']), btms_n(env['tol_lo']), ':', color='darkorange', lw=1.2)
    ax.plot(xn(tm_all), btms_n(Tm_all), 'r.', ms=2, alpha=0.5, label='measured')
    ax.plot(xn(t_all), btms_n(T_all), 'g-', lw=1.3, label='predicted')

    # Same convention as build_figure: RMSE in degrees on the absolute plot,
    # where there is a degree axis to read it against, MAPE on the normalized
    # one, where there is not. Computed on the physical values either way.
    _val, _lbl = overall_fit_metric(t_all, T_all, segments, normalize=normalize)
    if _val is None:
        _fit = ""
    else:
        _fit = (f"  --  {_lbl} = {_val:.1f} %" if normalize
                else f"  --  {_lbl} = {_val:.1f} \u00b0C")
    ax.set(ylabel=btms_label,
          title='Prediction, combined uncertainty, and tolerance' + _fit)
    ax.grid(alpha=0.3)

    # --- middle: u_c(t) itself against U_tol ---
    _draw_flight_markers(axu, segments, xn)
    axu.plot(xn(env['t']), unc_n(env['u_c']), '-', color='green', lw=1.3, label='u_c(t)')
    # U_tol reference line, same purple-when-exceeded convention as the
    # tolerance lines in the top panel.
    if env['drift'].any():
        first_drift, first_tol = True, True
        segs_tol = _split_curve_by_mask(env['t'], u_tol_flat, env['t'], env['drift'])
        for t_seg, tol_seg, is_drift in segs_tol:
            if len(t_seg) < 2:
                continue
            if is_drift:
                axu.plot(xn(t_seg), unc_n(tol_seg), ':', color='purple', lw=1.8, zorder=5,
                        label='U_tol exceeded' if first_drift else None)
                first_drift = False
            else:
                axu.plot(xn(t_seg), unc_n(tol_seg), ':', color='darkorange', lw=1.2,
                        label='U_tol' if first_tol else None)
                first_tol = False
    else:
        axu.plot(xn(env['t']), unc_n(u_tol_flat), ':', color='darkorange', lw=1.2, label='U_tol')
    axu.set(ylabel=unc_label)
    axu.grid(alpha=0.3)

    # --- bottom: relative error at measured points ---
    _draw_flight_markers(axe, segments, xn)
    if sel.sum():
        axe.plot(xn(tb_sel[~is_drift_pt]), err_n(err_pct[~is_drift_pt]),
                 '.', ms=3, color='tab:green', alpha=0.7,
                 label='relative error')
        if is_drift_pt.any():
            axe.plot(xn(tb_sel[is_drift_pt]), err_n(err_pct[is_drift_pt]),
                     '.', ms=4, color='purple', alpha=0.85,
                     label='relative error (drift)')
    axe.axhline(float(err_n(0.)) if normalize else 0., color='k', lw=0.8, ls='--')
    axe.axhline(float(err_n(20.)) if normalize else 20., color='0.7', lw=0.6, ls=':')
    axe.axhline(float(err_n(-20.)) if normalize else -20., color='0.7', lw=0.6, ls=':')
    axe.set(xlabel=xlabel, ylabel=err_label)
    axe.grid(alpha=0.3)

    # Single combined legend below the whole figure rather than one per panel -- each panel contributes its
    # own handles/labels, deduplicated by label so entries shared across
    # panels don't repeat.
    handles, labels = [], []
    for a in (ax, axu, axe):
        h, l = a.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    fig.legend(handles, labels, fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, 0.02), ncol=4)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def plot_result(t_all, T_all, segments, env=None):
    """CLI convenience wrapper: build the figure and show it interactively."""
    build_figure(t_all, T_all, segments, env=env)
    plt.show()


def build_result_dataframe(t_all, T_all, segments):
    """Tidy long-format table (time_s, value_c, series) of the predicted trace
    plus the measured points from the selected segments, for exporting a
    prediction to CSV rather than only seeing the figure."""
    tm_all = np.concatenate([s['tm_b']+s['touchdown_time_s'] for s in segments])
    Tm_all = np.concatenate([s['Tm_b'] for s in segments])
    pred = pd.DataFrame({'time_s': t_all, 'value_c': T_all, 'series': 'predicted'})
    meas = pd.DataFrame({'time_s': tm_all, 'value_c': Tm_all, 'series': 'measured'})
    return pd.concat([pred, meas], ignore_index=True).sort_values('time_s')


def build_segmentation_figure(t_full, gspd_kt, segments):
    """Ground speed over the whole loaded span, each detected flight's window
    shaded and its touchdown marked -- so a bad segmentation (a flight split
    in two, or two merged) is visible BEFORE calibrating on it."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_full/3600, gspd_kt, color='0.4', lw=0.8)
    colors = plt.cm.tab10.colors
    for i, seg in enumerate(segments):
        t0 = (seg['touchdown_time_s']+seg['tm_min'])/3600
        t1 = (seg['touchdown_time_s']+seg['tm_max'])/3600
        ax.axvspan(t0, t1, color=colors[i % len(colors)], alpha=0.15)
        ax.axvline(seg['touchdown_time_s']/3600, color=colors[i % len(colors)],
                  ls='--', lw=1.4)
        ax.text(seg['touchdown_time_s']/3600, ax.get_ylim()[1]*0.95, seg['label'],
               rotation=90, va='top', ha='right', fontsize=8,
               color=colors[i % len(colors)])
    ax.set(xlabel='time [h]', ylabel='ground speed [kt]',
          title='Segmentation check -- shaded = one detected flight, '
                'dashed = its touchdown')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# =============================================================================
# 9. CLI
# =============================================================================
def parse_flight_spec(spec, n_segments):
    """'all' | '2' | '1-3' | '1,3,4' -> sorted list of 0-based indices."""
    if spec == 'all':
        return list(range(n_segments))
    idx = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            idx.update(range(int(a)-1, int(b)))
        else:
            idx.add(int(part)-1)
    bad = [i+1 for i in idx if not (0 <= i < n_segments)]
    if bad:
        raise PipelineError(f"--flights refers to flight(s) {bad} but only "
                         f"{n_segments} were detected in this CSV.")
    return sorted(idx)


def segment_csv_inputs(csv_files, min_park_gap_min=20.0, park_kt=2.0, log=print,
                       wheel=1):
    """Stage 1: load the CSV(s) and split them into flight segments.
    csv_files may be paths or file-like objects (e.g. Streamlit's
    UploadedFile); wheel picks which brake's BTMS column drives every
    segment's measured data and initial condition (default 1).

    Returns (segments, t_full, gspd_kt) -- the usable segment dicts (see
    build_segment) plus the raw ground-speed trace, so a caller can plot the
    segmentation before trusting it (see build_segmentation_figure). Raises
    PipelineError if none are usable."""
    names = [getattr(f, 'name', str(f)) for f in csv_files]
    df, t_full, gspd_kt = load_day_csvs(csv_files, log=log)
    log(f"Loaded {len(csv_files)} file(s) ({', '.join(names)}): "
       f"{len(df)} rows, {(t_full[-1]-t_full[0])/3600:.1f} h span")

    blocks = segment_flights(t_full, gspd_kt, min_park_gap_min*60.0, park_kt)
    log(f"{len(blocks)} candidate block(s) found "
       f"(min-park-gap={min_park_gap_min:.0f} min):")
    segments = []
    for k, (lo, hi) in enumerate(blocks):
        label = f"flight_{k+1:02d}"
        seg = build_segment(df, t_full, gspd_kt, k, lo, hi, label, log=log,
                            wheel=wheel)
        if seg is None:
            gs = gspd_kt[lo:hi]
            log(f"  block {k+1}: [{t_full[lo]/3600:.2f}, {t_full[hi]/3600:.2f}] h "
               f"-- no landing detected, skipped")
            log(f"    diagnostic: ground speed -- min={gs.min():.1f}, "
               f"median={np.median(gs):.1f}, max={gs.max():.1f} kt; "
               f"{int((gs <= V_ROLL_END_KT).sum())}/{len(gs)} points "
               f"<= {V_ROLL_END_KT:.0f} kt, "
               f"{int((gs <= park_kt).sum())}/{len(gs)} points <= {park_kt:.0f} kt")
            continue
        segments.append(seg)
        log(f"  {label}: touchdown {seg['v_td_kt']:.0f} kt at "
           f"{seg['touchdown_time_s']/3600:.2f} h"
           + (f"  WARNING: {seg['warn']}" if seg['warn'] else ""))
    if not segments:
        raise PipelineError("No usable flight detected in this CSV -- check "
                            "min_park_gap_min and park_kt.")
    return segments, t_full, gspd_kt


def calibrate(segments, mode, core_json_a=None, core_json_b=None,
              bundle_a=None, bundle_b=None, data_fingerprint=None,
              use_cache=True, log=print):
    """Stage 2: build a calibration bundle (core + alpha + H_nat_flight +
    taxi_mode/pulse_dur), either by fitting the first segment, or by
    averaging two existing bundles -- given either directly as dicts
    (bundle_a/bundle_b, e.g. typed into a form) or as JSON sources
    (core_json_a/core_json_b, a path or file-like object).

    For mode='fit-first', data_fingerprint (see data_fingerprint()) is
    used to cache the result -- pass use_cache=False to force a refit."""
    if mode == 'fit-first':
        if use_cache and data_fingerprint:
            cached = load_calibration_cache(data_fingerprint)
            if cached is not None:
                log(f"Using cached fit-first calibration "
                   f"(fingerprint {data_fingerprint}, delete "
                   f"{os.path.basename(_cache_path(data_fingerprint))} in "
                   f".predict_day_cache/ to force a refit)")
                return cached
        log(f"Calibrating on {segments[0]['label']} (first flight in this CSV)...")
        core = fit_core(segments[0], log=log)
        stage2 = fit_alpha_h_nat_flight(segments[0], core, log=log)
        calib = dict(core=core, **stage2)
        if use_cache and data_fingerprint:
            save_calibration_cache(data_fingerprint, calib)
            log(f"Cached under fingerprint {data_fingerprint}")
        return calib
    bundle_a = bundle_a or (load_calibration_bundle(core_json_a) if core_json_a else None)
    bundle_b = bundle_b or (load_calibration_bundle(core_json_b) if core_json_b else None)
    if bundle_a is None and bundle_b is None:
        raise PipelineError("calibration='default' requires at least one "
                            "bundle (bundle_a/bundle_b, or core_json_a/"
                            "core_json_b) -- or use calibration='fit-first'.")
    if bundle_a is not None and bundle_b is not None:
        return average_bundles(bundle_a, bundle_b, log=log)
    single = bundle_a if bundle_a is not None else bundle_b
    log(f"  using single calibration bundle as-is (no averaging): "
       f"{single.get('source', '(manual input)')}")
    return single


PHASE_NAMES = ['taxi_before_takeoff', 'takeoff_climb', 'flight',
              'touchdown', 'taxi_in', 'parking']


def phase_windows(seg):
    """The six phase boundaries for ONE segment, as (phase_name, t0, t1)
    tuples in ABSOLUTE time (the same base as t_all/predict()'s output).
    Used by the relative-error uncertainty envelope below.

    'takeoff_climb' has no real end boundary available -- the model has
    no altitude channel, so CLIMB_DURATION_S (see AIRCRAFT_DEFAULTS, an
    UNVERIFIED placeholder) stands in for the climb-to-cruise transition.
    Every other boundary comes directly from build_segment()'s own
    detected values (t_taxi_accel, touchdown, t_land_end, t_full_stop)."""
    td = seg['touchdown_time_s']
    t_climb_end = min(td + seg['t_taxi_accel'] + CLIMB_DURATION_S, td)
    return [
        ('taxi_before_takeoff', td+seg['tm_min'], td+seg['t_taxi_accel']),
        ('takeoff_climb', td+seg['t_taxi_accel'], t_climb_end),
        ('flight', t_climb_end, td),
        ('touchdown', td, td+seg['t_land_end']),
        ('taxi_in', td+seg['t_land_end'], td+seg['t_full_stop']),
        ('parking', td+seg['t_full_stop'], td+seg['tm_max']),
    ]


def classify_phases(t_query, segments):
    """Phase label for every time in t_query, given the full chronological
    chain of segments. Anything outside every segment's own [tm_min, tm_max]
    window (i.e. an inter-flight gap) is labelled 'gap':
    build_uncertainty_envelope excludes those points from the combined
    uncertainty rather than applying a phase rate that doesn't hold there."""
    labels = np.full(len(t_query), 'gap', dtype=object)
    for seg in segments:
        for name, t0, t1 in phase_windows(seg):
            if t1 <= t0:
                continue
            sel = (t_query >= t0) & (t_query < t1)
            labels[sel] = name
    return labels


def relative_error_by_phase(calib_segment, calib, log=print):
    """Runs the calibrated model on its OWN calibration flight (in-sample by
    construction) and measures the model/measurement relative error,
    RMS-averaged within each of the six phases. That is u_BDT's
    phase-dependent rate: multiplied by a later prediction's T_model(t) it
    gives u_BDT(Tx) at that point.

    Returns {phase_name: rate}, dimensionless (0.05 = 5% of the local model
    temperature). A phase with no measured points on the calibration flight
    falls back to the mean rate over the phases that do have data."""
    t_cal, T_cal, _, _ = predict([calib_segment], calib)
    tm_b, Tm_b = calib_segment['tm_b']+calib_segment['touchdown_time_s'], calib_segment['Tm_b']
    T_model_at_meas = np.interp(tm_b, t_cal, T_cal)
    labels = classify_phases(tm_b, [calib_segment])

    rates = {}
    for name in PHASE_NAMES:
        sel = labels == name
        if sel.sum() < 2:
            continue
        # relative to the MODEL value at that point (u_BDT scales T_model,
        # not T_measured -- the envelope is drawn around the model curve)
        denom = np.where(np.abs(T_model_at_meas[sel]) < 1e-6, np.nan,
                         T_model_at_meas[sel])
        rel = (T_model_at_meas[sel] - Tm_b[sel]) / denom
        rates[name] = float(np.sqrt(np.nanmean(rel**2)))

    missing = [n for n in PHASE_NAMES if n not in rates]
    if missing and rates:
        fallback = float(np.mean(list(rates.values())))
        log(f"  relative-error rate: no calibration-flight data in phase(s) "
           f"{missing} -- using the {fallback:.3f} mean of the other "
           f"phases for {'them' if len(missing) > 1 else 'it'}")
        for n in missing:
            rates[n] = fallback
    elif not rates:
        raise PipelineError("relative_error_by_phase: no phase had enough "
                            "calibration-flight data to compute anything.")

    log("  relative error rate by phase (RMS, on the calibration flight):")
    for name in PHASE_NAMES:
        log(f"    {name:<20s} {100*rates[name]:5.1f}%")
    return rates


def build_uncertainty_envelope(t_all, T_all, segments, calib, log=print):
    """Combined-uncertainty envelope around the model curve. For every point,

        u_BDT(Tx) = relative_error_rate(phase at that point) * T_model(Tx)
        u_c(Tx)   = sqrt(u_BDT(Tx)^2 + u_BTS^2)

    then the envelope is T_model +/- u_c(T_model) -- a combined-standard-
    uncertainty band (GUM-style quadrature sum of two independent error
    sources), NOT a sensitivity study over modelling choices like the
    envelope this replaces. u_BTS (AIRCRAFT_DEFAULTS['u_bts_c']) is an
    UNVERIFIED placeholder -- see its own definition and the warning
    printed below.

    Also returns the governing TOLERANCE band T_model +/- U_tol and a drift
    flag wherever u_c(Tx) > U_tol (the BDT drift-detection rule). U_tol
    (AIRCRAFT_DEFAULTS['u_tol_c']) is a POLICY choice, not a measured
    quantity -- also an UNVERIFIED placeholder.

    The relative-error rate always comes from segments[0] (the calibration
    flight, whichever mode was used), whatever flights are being predicted."""
    log(f"  ! U_BTS_C = {U_BTS_C:.2f} C is an UNVERIFIED placeholder "
       f"(no manufacturer/literature source found) -- see AIRCRAFT_DEFAULTS")
    log(f"  ! CLIMB_DURATION_S = {CLIMB_DURATION_S:.0f}s is an UNVERIFIED "
       f"assumption (no altitude channel to detect the real climb end)")
    log(f"  ! U_TOL_C = {U_TOL_C:.2f} C is an UNVERIFIED placeholder "
       f"(a policy/engineering target, not a measured or literature value)")
    rates = relative_error_by_phase(segments[0], calib, log=log)

    labels = classify_phases(t_all, segments)
    # 'gap' true inter-flight gaps get NaN here --
    # no phase rate was ever fitted to represent a multi-hour gap between
    # flights, so u_bdt/u_c/lo/hi/drift all correctly come out NaN at
    # these points too, rather than reusing 'parking''s own short-tail
    # rate. The plotting code already uses nanmin/nanmax and
    # fill_between's own NaN handling throughout, so this needs no
    # further changes downstream.
    rate_arr = np.array([rates.get(l, np.nan) for l in labels])
    n_gap = int(np.sum(labels == 'gap'))
    if n_gap:
        log(f"  combined uncertainty: {n_gap} point(s) ({100.0*n_gap/len(labels):.1f}%) "
           f"fall in an inter-flight gap -- u_c left undefined (NaN) there, "
           f"not assigned 'parking''s own short-tail rate")
    u_bdt = np.abs(rate_arr*T_all)
    u_c = np.sqrt(u_bdt**2 + U_BTS_C**2)

    drift = u_c > U_TOL_C
    if drift.any():
        frac = 100.0*drift.mean()
        log(f"  drift check: u_c(Tx) > U_tol on {drift.sum()} points "
           f"({frac:.1f}% of the prediction) -- see the 'drift' shading "
           f"on the plot")
    else:
        log(f"  drift check: u_c(Tx) never exceeds U_tol={U_TOL_C:.2f} C "
           f"over this prediction")

    return dict(t=t_all, lo=T_all-u_c, hi=T_all+u_c, u_c=u_c,
               tol_lo=T_all-U_TOL_C, tol_hi=T_all+U_TOL_C,
               drift=drift)


def run_prediction(segments, sel_idx, calib, include_envelope=False,
                   gap_model='h_nat', log=print):
    """Stage 3: predict across the full chronological chain up to the
    last selected flight (gap continuity needs the whole chain even if
    the selection skips one), then trims the report to the selection.
    If include_envelope, also returns a combined-uncertainty band (see
    build_uncertainty_envelope) -- one extra full simulation of the chain
    (to characterise the calibration flight's own relative error), on top
    of the main prediction above."""
    predicted_segments = [segments[i] for i in sel_idx]
    chain = segments[:sel_idx[-1]+1]
    log("Predicting...")
    t_all, T_all, flight_report, gap_report = predict(
        chain, calib, gap_model=gap_model, log=log)
    keep_labels = {segments[i]['label'] for i in sel_idx}
    flight_report = [r for r in flight_report if r['flight'] in keep_labels]
    gap_report = [g for g in gap_report
                  if g['gap'].split('->')[0] in keep_labels
                  or g['gap'].split('->')[1] in keep_labels]
    env = (build_uncertainty_envelope(t_all, T_all, chain, calib, log=log)
          if include_envelope else None)
    return t_all, T_all, predicted_segments, flight_report, gap_report, env


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True, nargs='+',
                    help="path(s) to the day's CSV(s) -- pass one file for "
                         "a single whole-day file, or several per-flight "
                         "files (e.g. flight_0001.csv flight_0002.csv ...) "
                         "to concatenate into one day. Order on the command "
                         "line doesn't matter -- they're sorted by their "
                         "own Timestamp column.")
    ap.add_argument('--flights', default='all',
                    help="'all' (default), a single flight '2', a range "
                         "'2-4', or a comma list '1,3,4' -- 1-based, in "
                         "chronological order as detected in the CSV")
    ap.add_argument('--calibration', choices=['default', 'fit-first'],
                    default='default',
                    help="'default': average --core-json-a/--core-json-b. "
                         "'fit-first': calibrate on the first flight found "
                         "in this CSV, then predict the rest")
    ap.add_argument('--core-json-a', help="calibration bundle A for "
                    "--calibration default (e.g. flight_0001's own)")
    ap.add_argument('--core-json-b', help="calibration bundle B for "
                    "--calibration default (e.g. flight_0005's own)")
    ap.add_argument('--min-park-gap-min', type=float, default=20.0,
                    help="minimum parked duration [min] to count as a gap "
                         "between two flights (default 20)")
    ap.add_argument('--park-kt', type=float, default=0.0,
                    help="ground speed [kt] below which the aircraft is "
                         "considered parked (default 0 -- a parked "
                         "aircraft doesn't move)")
    ap.add_argument('--envelope', action='store_true',
                    help="also compute and plot the combined-uncertainty "
                         "band (calibration-flight relative error by phase "
                         "+ sensor uncertainty, one extra simulation)")
    ap.add_argument('--gap-model', default='h_nat',
                    choices=['h_nat', 'exp_gap1'],
                    help="how inter-flight gaps are filled (default "
                         "h_nat): 'h_nat' runs continuous physics through "
                         "the gap; 'exp_gap1' fits a Newton's-law-of-"
                         "cooling exponential once on the day's first gap "
                         "and reuses it, unchanged, for every later gap")
    ap.add_argument('--no-cache', action='store_true',
                    help="skip the fit-first calibration cache (see "
                         ".predict_day_cache/), always refit")
    ap.add_argument('--check-segmentation', action='store_true',
                    help="show the segmentation-check figure (ground speed "
                         "with detected flight windows) before calibrating")
    ap.add_argument('--out-calibration-json',
                    help="write the calibration used (core+alpha+H_nat_flight"
                         "+taxi_mode) to this JSON path")
    ap.add_argument('--out-csv',
                    help="write the predicted+measured series to this CSV "
                         "path (long format: time_s, value_c, series)")

    ac = ap.add_argument_group(
        'aircraft constants',
        'Override this project\'s A320 defaults to reuse this script on a '
        'different aircraft type. Any not given keeps its A320 default '
        '(see AIRCRAFT_DEFAULTS). --aircraft-json overrides any subset by '
        'key in one file instead of one flag each.')
    ac.add_argument('--aircraft-json', help="JSON file with any subset of "
                    "AIRCRAFT_DEFAULTS' keys")
    ac.add_argument('--n-brakes', type=int, help="braked wheels (default 4)")
    ac.add_argument('--max-e-taxi-mj-brake', type=float,
                   help="hard cap on taxi-out energy per brake [MJ/brake], 0=disabled (default 0, disabled)")
    ac.add_argument('--rho-air', type=float, help="air density [kg/m3] (default 1.225)")
    ac.add_argument('--a-wing', type=float, help="wing reference area [m2] (default 123.0)")
    ac.add_argument('--c-d', type=float, help="aerodynamic drag coefficient (default 0.12)")
    ac.add_argument('--mu-rr', type=float, help="rolling resistance coefficient (default 0.009)")
    ac.add_argument('--n-sp', type=int, help="spoiler count (default 10)")
    ac.add_argument('--l-sp', type=float, help="spoiler length [m] (default 1.6)")
    ac.add_argument('--b-sp', type=float, help="spoiler width [m] (default 0.625)")
    ac.add_argument('--theta-max-deg', type=float, help="max spoiler deflection [deg] (default 50.0)")
    ac.add_argument('--c-sp', type=float, help="spoiler drag coefficient (default 1.5)")
    ac.add_argument('--k-rt', type=float, help="reverse-thrust force [N], 0=disabled (default 0.0)")
    ac.add_argument('--v-th-rev-kt', type=float,
                    help="ground speed below which reverse thrust cuts off [kt] (default 40.0)")
    ac.add_argument('--enable-retraction-window', action='store_true', default=None,
                    help="model the gear-retraction cooling window (default: disabled)")
    ac.add_argument('--n-sub-land', type=int, help="landing-roll sub-steps per raw interval (default 10)")
    ac.add_argument('--v-td-threshold-kt', type=float,
                    help="rule_100kt threshold [kt] (default 100.0)")
    ac.add_argument('--v-roll-end-kt', type=float,
                    help="ground speed [kt] marking the roll-out as done (default 20.0)")
    ac.add_argument('--v-td-plausible', type=float, nargs=2, metavar=('LO', 'HI'),
                    help="plausible touchdown speed range [kt] (default 110 165)")
    ac.add_argument('--mass-lb-threshold', type=float,
                    help="GROSS WEIGHT above this is assumed already in lb, "
                         "not kg (default 80000)")
    ac.add_argument('--v-stop-kt', type=float,
                    help="ground speed [kt] marking a full stop, finer than "
                         "--v-roll-end-kt (default 0.5)")
    ac.add_argument('--lg-debounce-s', type=float,
                    help="landing-gear signal gaps shorter than this [s] "
                         "are bridged rather than treated as a real state "
                         "change (default 5.0)")
    ac.add_argument('--stop-debounce-s', type=float,
                    help="how long [s] the aircraft must stay under "
                         "--v-stop-kt to count as genuinely stopped "
                         "(default 60.0)")

    tm = ap.add_argument_group(
        'thermal module constants (brake/wheel hardware)',
        'Override btem_v5_full_flight.py\'s own fixed geometry, '
        'material, conductances, emissivities, convection, and timing -- '
        'distinct from the "aircraft constants" group above, which is '
        'about the airframe\'s aerodynamics, not its brakes. The 7 FITTED '
        'core parameters (C_tube, G_ST, tau, eps_DT, G_RW, h_nat, '
        'eps_T) are calibrated by this pipeline and are not here. '
        '--thermal-json overrides any subset by key (see '
        'THERMAL_MODULE_DEFAULTS) in one file instead of one flag each.')
    tm.add_argument('--thermal-json', help="JSON file with any subset of "
                    "THERMAL_MODULE_DEFAULTS' keys")
    tm.add_argument('--r-wheel', type=float, help="wheel rolling radius [m] (default 0.584)")
    tm.add_argument('--r-ext', type=float, help="disc external radius [m] (default 0.183)")
    tm.add_argument('--r-int', type=float, help="disc internal radius [m] (default 0.118)")
    tm.add_argument('--e-disc', type=float, help="disc thickness [m] (default 0.022)")
    tm.add_argument('--r-rim', type=float, help="wheel rim radius [m] (default 0.255)")
    tm.add_argument('--l-rim', type=float, help="wheel rim length [m] (default 0.417)")
    tm.add_argument('--a-ct', type=float, help="tube contact area [m2] (default 0.178)")
    tm.add_argument('--a-cw', type=float, help="housing contact area [m2] (default 0.378)")
    tm.add_argument('--rho-c', type=float, help="disc material density [kg/m3] (default 1800.0)")
    tm.add_argument('--cp-c', type=float, help="disc material specific heat [J/kg/K] (default 1420.0)")
    tm.add_argument('--kz-c', type=float, help="disc material through-thickness conductivity [W/m/K] (default 10.0)")
    tm.add_argument('--mass-wheel', type=float, help="wheel/rim mass [kg] (default 45.0)")
    tm.add_argument('--cp-wheel', type=float, help="wheel/rim specific heat [J/kg/K] (default 900.0)")
    tm.add_argument('--mass-shield', type=float, help="heat shield mass [kg] (default 2.0)")
    tm.add_argument('--cp-shield', type=float, help="heat shield specific heat [J/kg/K] (default 500.0)")
    tm.add_argument('--mass-pist', type=float, help="piston housing mass [kg] (default 4.0)")
    tm.add_argument('--cp-pist', type=float, help="piston housing specific heat [J/kg/K] (default 600.0)")
    tm.add_argument('--mass-sens', type=float, help="BTMS sensor mass [kg] (default 1.0)")
    tm.add_argument('--cp-sens', type=float, help="BTMS sensor specific heat [J/kg/K] (default 500.0)")
    tm.add_argument('--g-ph', type=float, help="tube->housing conductance [W/K] (default 1.5)")
    tm.add_argument('--g-sp', type=float, help="S1->housing conductance [W/K] (default 1.0)")
    tm.add_argument('--g-sw', type=float, help="shield->wheel conductance [W/K] (default 0.5)")
    tm.add_argument('--eps-ds', type=float, help="disc->shield emissivity (default 0.80)")
    tm.add_argument('--eps-sw', type=float, help="shield->wheel emissivity (default 0.25)")
    tm.add_argument('--eps-end', type=float, help="disc end face->ambient emissivity (default 0.80)")
    tm.add_argument('--eps-w', type=float, help="wheel->ambient emissivity (default 0.19)")
    tm.add_argument('--eps-ph', type=float, help="housing->ambient emissivity (default 0.50)")
    tm.add_argument('--eps-s1p', type=float, help="S1->housing (taxi gap) emissivity (default 0.70)")
    tm.add_argument('--h-fan', type=float, help="taxi brake-cooling fan coefficient [W/m2K] (default 500.0)")
    tm.add_argument('--k-air', type=float, help="air thermal conductivity [W/m/K] (default 0.02588)")
    tm.add_argument('--nu-air', type=float, help="air kinematic viscosity [m2/s] (default 1.608e-5)")
    tm.add_argument('--pr-air', type=float, help="air Prandtl number (default 0.728)")
    tm.add_argument('--l-conv', type=float, help="convection length scale [m] (default 0.30)")
    tm.add_argument('--takeoff-accel-thresh', type=float,
                    help="ground acceleration [m/s2] above which the aircraft "
                         "is considered taking off (default 1.0)")
    tm.add_argument('--t-retract-s', type=float, help="gear-retraction cooling window [s] (default 20.0)")
    tm.add_argument('--h-approach-ft', type=float, help="altitude where approach cooling starts [ft] (default 2000.0)")
    tm.add_argument('--approach-fallback-s', type=float,
                    help="fallback approach-cooling duration if no altitude "
                         "column is found [s] (default 100.0)")
    tm.add_argument('--t-bay-c', type=float, help="gear-bay temperature, gear stowed [C] (default 30.0)")
    tm.add_argument('--v-taxi-kt', type=float, help="representative taxi ground speed [kt] (default 20.0)")
    tm.add_argument('--p-brake-thresh', type=float,
                    help="power [W] above which 'flight'-phase ground contact "
                         "counts as real braking (default 50000 -- requires "
                         "the 2026-08-10 module patch, see configure_thermal_module)")
    tm.add_argument('--disable-flight-forced-cooling', action='store_true', default=None,
                    help="skip the gear-retraction/approach forced-convection "
                         "windows during 'flight' (default: modeled)")
    tm.add_argument('--fan-on', action='store_true', default=None,
                    help="model taxi brake-cooling fans as on (default: off, "
                         "this project's own convention)")
    tm.add_argument('--ground-fan-on', action='store_true', default=None,
                    help="model parking ground fans as on (default: off, "
                         "this project's own convention)")

    args = ap.parse_args()

    if args.thermal_json:
        with open(args.thermal_json) as f:
            thermal_overrides = json.load(f)
    else:
        thermal_overrides = {}
    thermal_overrides.update({k: v for k, v in dict(
        r_wheel=args.r_wheel, R_ext=args.r_ext, R_int=args.r_int, e_disc=args.e_disc,
        R_rim=args.r_rim, L_rim=args.l_rim, A_ct=args.a_ct, A_cw=args.a_cw,
        rho_c=args.rho_c, cp_c=args.cp_c, kz_c=args.kz_c,
        mass_wheel=args.mass_wheel, cp_wheel=args.cp_wheel,
        mass_shield=args.mass_shield, cp_shield=args.cp_shield,
        mass_pist=args.mass_pist, cp_pist=args.cp_pist,
        mass_sens=args.mass_sens, cp_sens=args.cp_sens,
        G_PH=args.g_ph, G_SP=args.g_sp, G_SW=args.g_sw,
        eps_DS=args.eps_ds, eps_SW=args.eps_sw, eps_end=args.eps_end,
        eps_W=args.eps_w, eps_PH=args.eps_ph, eps_S1P=args.eps_s1p,
        h_fan=args.h_fan, k_air=args.k_air, nu_air=args.nu_air, pr_air=args.pr_air,
        l_conv=args.l_conv,
        takeoff_accel_thresh=args.takeoff_accel_thresh, t_retract_s=args.t_retract_s,
        h_approach_ft=args.h_approach_ft, approach_fallback_s=args.approach_fallback_s,
        t_bay_c=args.t_bay_c, v_taxi_kt=args.v_taxi_kt,
        p_brake_thresh=args.p_brake_thresh,
        flight_forced_cooling=(not args.disable_flight_forced_cooling
                               if args.disable_flight_forced_cooling is not None else None),
        fan_on=args.fan_on, ground_fan_on=args.ground_fan_on,
    ).items() if v is not None})
    configure_thermal_module(thermal_overrides)

    if args.aircraft_json:
        with open(args.aircraft_json) as f:
            aircraft_overrides = json.load(f)
    else:
        aircraft_overrides = {}
    aircraft_overrides.update({k: v for k, v in dict(
        n_brakes=args.n_brakes, max_e_taxi_mj_brake=args.max_e_taxi_mj_brake,
        rho_air=args.rho_air, a_wing=args.a_wing,
        c_d=args.c_d, mu_rr=args.mu_rr, n_sp=args.n_sp, l_sp=args.l_sp,
        b_sp=args.b_sp, theta_max_deg=args.theta_max_deg, c_sp=args.c_sp,
        k_rt=args.k_rt, v_th_rev_kt=args.v_th_rev_kt,
        disable_retraction_window=(not args.enable_retraction_window
                                   if args.enable_retraction_window is not None else None),
        n_sub_land=args.n_sub_land, v_td_threshold_kt=args.v_td_threshold_kt,
        v_roll_end_kt=args.v_roll_end_kt, v_td_plausible=args.v_td_plausible,
        mass_lb_threshold=args.mass_lb_threshold, v_stop_kt=args.v_stop_kt,
        lg_debounce_s=args.lg_debounce_s, stop_debounce_s=args.stop_debounce_s,
    ).items() if v is not None})
    configure_constants(aircraft_overrides)

    try:
        segments, t_full, gspd_kt = segment_csv_inputs(
            args.csv, args.min_park_gap_min, args.park_kt)
        if args.check_segmentation:
            build_segmentation_figure(t_full, gspd_kt, segments)
            plt.show()
        sel_idx = parse_flight_spec(args.flights, len(segments))
        print(f"Selected flight(s): {[segments[i]['label'] for i in sel_idx]}")
        fp = data_fingerprint(args.csv) if args.calibration == 'fit-first' else None
        calib = calibrate(segments, args.calibration, args.core_json_a, args.core_json_b,
                          data_fingerprint=fp, use_cache=not args.no_cache)
        t_all, T_all, predicted_segments, flight_report, gap_report, env = \
            run_prediction(segments, sel_idx, calib, include_envelope=args.envelope,
                          gap_model=args.gap_model)
    except PipelineError as e:
        raise SystemExit(str(e))

    print_report(flight_report, gap_report)
    if args.out_calibration_json:
        with open(args.out_calibration_json, 'w') as f:
            json.dump(calib, f, indent=2)
        print(f"Calibration written to {args.out_calibration_json}")
    if args.out_csv:
        build_result_dataframe(t_all, T_all, predicted_segments).to_csv(
            args.out_csv, index=False)
        print(f"Predicted+measured series written to {args.out_csv}")
    plot_result(t_all, T_all, predicted_segments, env=env)


if __name__ == '__main__':
    main()