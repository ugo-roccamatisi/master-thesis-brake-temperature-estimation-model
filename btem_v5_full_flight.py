# =============================================================================
# SANITIZED PUBLIC VERSION
# All calibrated parameter values have been replaced with generic round
# placeholders, and all flight identifiers, timestamps and paths with
# examples. Calibrate on your own data before any quantitative use.
# =============================================================================
# -*- coding: utf-8 -*-
"""
================================================================================
BTEM v5 - Brake Temperature Estimation Model (Airbus A320 carbon-carbon brake)
14-node lumped-mass thermal model, Method C (data-driven) only.

NODES (14): 9 discs (S1-R1-S2-R2-S3-R3-S4-R4-S5) + torque tube + wheel/rim
            + BTMS sensor + heat shield + piston housing.

PHASES: LANDING -> TAXI -> PARKING
  - generation q from measured torque
  - inter-disc conduction: G_brake (landing & parking), G_cool (taxi)
  - S1<->housing: conduction G_SP (landing, parking), radiation eps_S1P (taxi)
  - disc airflow convection: h_disc(v) = eta_disc*h_forced(v) [landing, taxi]
  - fan cooling on discs: h_fan*A_disc [FAN_ON taxi, GROUND_FAN_ON parking]
  - carrier convection: h_forced(v) [landing, taxi], h_nat [parking]

HEAT SHIELD: intermediate node (discs -> shield -> wheel); NO ambient loss.
PARAMETER PROVENANCE:  [S] sourced  [E] estimated
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# =============================================================================
# 0. FLIGHT / CSV SELECTION
# =============================================================================
FLIGHTS = {
    'FLIGHT_A': dict(path='data/flight_a.csv', start='2020-01-01 12:00:00'),
}
SELECTED   = 'FLIGHT_A'
WHEEL_ID   = 4
T_END      = 3600.0

# --- phase scenario ----------------------------------------------------------
T_TAXI        = 300.0       # [E] taxi duration after rollout [s]

# --- spurious brake power during the take-off roll (see load_csv) -----------
SUPPRESS_TAKEOFF_ROLL = True
TAKEOFF_ACCEL_THRESH  = 1.0   # [E] m/s^2, above this on the ground = take-off

# --- pre-touchdown correction knobs -----------------------------------------
# All act on the 'flight' phase only, so the touchdown calibration is untouched
# by construction. Neutral value = disabled.
#   C_TUBE_FLIGHT_MULT: touchdown C_tube is fitted on a 36s transient where the
#     tube barely participates; a longer pre-takeoff event needs a larger
#     EFFECTIVE capacity to stand in for more hardware (axle, hub, gear leg).
#   P_FLIGHT_SCALE (alpha): scales torque-derived energy reaching the pack --
#     over ~7 min of convection, less of it arrives than in a near-adiabatic 36s.
#   H_NAT_FLIGHT: natural convection for 'flight' only, decoupled from h_nat
#     (fitted where it barely matters). Assigned below, once h_nat exists.
C_TUBE_FLIGHT_MULT = 1.0
P_FLIGHT_SCALE = 1.0

P_BRAKE_THRESH = 50e3   # [E] braking vs rolling/noise during 'flight' (rhs)

# --- forced-convection windows inside the 'flight' phase ---------------------
# Two airborne moments where the assembly still sees a real airstream even
# though on_ground(t) is already False:
#   (1) gear retraction: T_RETRACT_S after lift-off, bay doors still open
#   (2) final approach: from H_APPROACH_FT down to touchdown, gear extended
# Both treated like roll-out (disc airflow + forced carrier convection at
# measured speed). BETWEEN them the gear is stowed and the brakes see T_BAY_C.
FLIGHT_FORCED_COOLING = True
T_RETRACT_S    = 20.0        # [E] gear-retraction cooling window [s]
H_APPROACH_FT  = 2000.0      # [E] altitude at which approach cooling starts
APPROACH_FALLBACK_S = 100.0  # [E] used only if the CSV has no altitude column
T_BAY_C        = 20.0        # [E] landing-gear bay temperature, gear stowed
T_BAY_K        = T_BAY_C + 273.15
V_TAXI_KT     = 20.0        # [E] representative taxi ground speed [kt]
FAN_ON        = True        # taxi brake-cooling fans
GROUND_FAN_ON = True        # parking ground fans

# --- calibrated parameters ----------------------------------------------------
CALIBRATED = True  # True: use calibrated values below, False: use defaults
if CALIBRATED:
    C_tube   = 10000.0
    G_ST     = 2.5000
    G_sens   = 100.0000   # placeholder
    eps_DT   = 0.9000
    G_RW     = 2.0000
    h_nat    = 10.0000
    ETA_DISC = 0.5000
else:
    G_ST     = 2.5
    C_tube   = 10000
    G_sens   = 3.0
    G_RW     = 2.0
    eps_DT   = 0.90
    h_nat    = 8.0
    ETA_DISC = 0.5

H_NAT_FLIGHT = h_nat   # default: same as touchdown, until explicitly overridden

# =============================================================================
# 1. CONSTANTS
# =============================================================================
SIGMA = 5.670e-8
KT2MS = 0.514444
T0    = 300.0
r_wheel = 0.584              # [S] wheel rolling radius [m]

# =============================================================================
# 2. GEOMETRY (updated from drawings + literature)
# =============================================================================
R_ext, R_int, e_disc = 0.183, 0.118, 0.022       # [S] lit + drawing
A_f    = np.pi * (R_ext**2 - R_int**2)            # 6.15e-2 m2
A_in   = 2*np.pi*R_int*e_disc                     # 1.63e-2 m2 (per disc -> tube)
A_out  = 2*np.pi*R_ext*e_disc                     # 2.53e-2 m2 (per disc -> shield)
R_rim, L_rim = 0.255, 0.417                       # [E] R_rim = mean(R_ext, R_int) of rim; L_rim from drawing
A_SW   = 2*np.pi * R_rim * L_rim                  # 6.68e-1 m2 (shield<->wheel contact area, rim cylinder)
A_ct   = 0.178                                    # [S] drawing: 2*pi*R_tube*L_exposed
A_cw   = 0.378                                    # [S] drawing: pi*R_rim^2 + E*2*pi*R_rim
A_cp   = A_f                                      # [S] drawing: housing face ~ A_f
A_disc = A_out                                    # disc area for airflow/fan (per disc)
A_S1P  = A_f / 2                                  # [E] S1<->piston area (taxi, ~half face)
A_fend_S1 = A_f / 2                               # [E] S1 end face (partly occluded)
A_fend_S5 = A_f                                   # S5 end face (unobstructed)

# =============================================================================
# 3. MATERIAL & HEAT CAPACITIES
# =============================================================================
rho_c, cp_c, kz_c = 1800.0, 1420.0, 10.0         # [S] Youssef
C_disc  = rho_c * A_f * e_disc * cp_c             # 3458 J/K
# C_tube: defined in the CALIBRATED block above
C_wheel = 45.0 * 900.0                            # [E]
C_shield= 2.0  * 500.0                            # [E]
C_pist  = 4.0  * 600.0                            # [E]
C_sens  = 1.0  * 500.0                            # [E]*

# =============================================================================
# 4. CONDUCTANCES [W/K]
# =============================================================================
G_brake = A_f / (1e-4 + e_disc/kz_c)              # 26.7 [S] Bauzin
G_cool  = A_f / (2.55 * e_disc/kz_c)              # 11.0 [S] Meunier
# G_RW, G_ST, G_sens: defined in the CALIBRATED block above
G_PH   = 1.5        # [E]  tube->housing
G_SP   = 1.0        # [E]  S1->housing (landing & parking)
G_SW   = 0.5        # [E]  shield->wheel (brackets)
is_rotor = np.array([0,1,0,1,0,1,0,1,0], dtype=bool)

# =============================================================================
# 5. RADIATION EMISSIVITIES
# =============================================================================
eps_DS  = 0.80      # [S] disc->shield
eps_SW  = 0.25      # [E] shield->wheel
eps_end = 0.80      # [S] disc end face->ambient
eps_W   = 0.19      # [S] wheel->ambient
eps_PH  = 0.50      # [E] housing->ambient
eps_S1P = 0.70      # [E] S1->housing (taxi gap)
eps_T    = 0.80      # default here; calibrated via
                     # predict_day_from_csv.py's NAMES/set_core()
# eps_DT: defined in the CALIBRATED block above

# =============================================================================
# 6. CONVECTION
# =============================================================================
# h_nat: defined in the CALIBRATED block above
h_fan     = 50.0                                   # [E] fan coefficient
# ETA_DISC: defined in the CALIBRATED block above; not calibrated
K_AIR, NU_AIR, PR_AIR = 0.02588, 1.608e-5, 0.728  # [S] IESL
L_CONV = 0.30                                      # [E]

def h_forced(v):
    if v <= 0:
        return 0.0
    Re = v * L_CONV / NU_AIR
    if Re <= 5e5:
        Nu = 2*0.3387*Re**0.5*PR_AIR**(1/3) / (1+(0.0468/PR_AIR)**(2/3))**0.25
    else:
        Nu = (0.037*Re**0.8 - 871)*PR_AIR**(1/3)
    return Nu * K_AIR / L_CONV

def h_disc_airflow(v):
    """Disc forced convection from wheel-cavity airflow = eta_disc * h_forced(v)."""
    return ETA_DISC * h_forced(v)

# =============================================================================
# 7. NODE INDICES
# =============================================================================
labels = ['S1','R1','S2','R2','S3','R3','S4','R4','S5']
ND = 9
I_TUBE, I_WHEEL, I_SENS, I_SHIELD, I_PIST = 9, 10, 11, 12, 13
N = 14

# =============================================================================
# 8. HEAT SOURCE (CSV, Method C) — robust loading
# =============================================================================
_P_interp = _v_interp = _t_stop_C = _t_speed0_C = _btms = None
_t_record = None
_T0_interp = None          # ambient temperature as a function of time
_t_min_C   = None          # earliest time in the recording (relative to touchdown)
_on_ground_interp = None   # debounced on-ground indicator
_t_liftoff   = None        # last lift-off before touchdown (wheel speed -> 0)
_t_app_start = None        # time the aircraft descends through H_APPROACH_FT
_alt_interp  = None        # altitude [ft] vs time, if the CSV provides it

def _norm(s):
    """Column-name key: lowercase, no spaces or underscores."""
    return str(s).lower().replace(' ', '').replace('_', '')

def _interp(x, y, fill=0.0):
    """interp1d with the flat-extrapolation settings used throughout."""
    return interp1d(x, y, bounds_error=False, fill_value=fill)

def _read_csv_robust(path):
    df = pd.read_csv(path, sep=None, engine='python')
    df.columns = [str(c).strip() for c in df.columns]
    if 'GMTs' not in df.columns:
        norm = {_norm(c): c for c in df.columns}
        for key in ('gmts','gmt','time','utc','timestamp','datetime'):
            if key in norm:
                df = df.rename(columns={norm[key]: 'GMTs'}); break
    if 'GMTs' not in df.columns:
        raise ValueError(
            f"No time column 'GMTs' found.\nColumns: {list(df.columns)}\n"
            "-> rename the time column to 'GMTs', or check the separator.")
    df['GMTs'] = pd.to_datetime(df['GMTs'], errors='coerce')
    return df.dropna(subset=['GMTs']).sort_values('GMTs').reset_index(drop=True)

def _col(df, name):
    """Column by name, tolerant of spacing/underscore/case; zeros if absent."""
    if name in df.columns: return df[name]
    norm = {_norm(c): c for c in df.columns}
    if _norm(name) in norm: return df[norm[_norm(name)]]
    print(f"[warning] column '{name}' not found -> filled with 0")
    return pd.Series(np.zeros(len(df)), index=df.index)

def load_csv(path, start_time, wheel_id=1):
    global _P_interp, _v_interp, _t_stop_C, _t_speed0_C, _btms, _t_record, T0
    global _T0_interp, _t_min_C, _on_ground_interp, T_TAXI
    global _t_liftoff, _t_app_start, _alt_interp
    df = _read_csv_robust(path)
    df['t'] = (df['GMTs'] - df['GMTs'].iloc[0]).dt.total_seconds()
    w = str(wheel_id)
    v_wheel = _col(df, f'Wheel{w}_Speed(Kt)').fillna(0)*KT2MS
    omega   = v_wheel / r_wheel
    trq     = _col(df, f'Brake{w}_TRQ').fillna(0)*10.0
    gspd    = _col(df, 'Ground_Speed(Kt)').fillna(0)*KT2MS
    i0 = (df['GMTs'] >= pd.to_datetime(start_time)).idxmax()
    V_TAXI = 20*KT2MS
    roll = np.where((df.index >= i0) & (gspd.values <= V_TAXI))[0]
    i1 = roll[0] if len(roll) else len(df)-1
    # sustained stop only (>=60s below threshold), not a brief taxi pause
    V_STOP = 0.5*KT2MS
    MIN_STOP_S = 60.0
    dt_probe = float(np.median(np.diff(df['t'].values))) if len(df) > 1 else 1.0
    min_stop_n = max(1, int(round(MIN_STOP_S / max(dt_probe, 1e-6))))
    stopped = (df.index >= i0) & (gspd.values <= V_STOP)
    i2 = len(df) - 1
    j = 0
    n = len(stopped)
    while j < n:
        if stopped[j]:
            k = j
            while k < n and stopped[k]:
                k += 1
            if (k - j) >= min_stop_n:
                i2 = j
                break
            j = k
        else:
            j += 1
    t0 = df['t'].iloc[i0]

    # rolling (not single-point) bias: a fixed bias from t=0 doesn't hold
    # over a multi-hour recording (sensor drift confirmed on real data)
    dt_full = float(np.median(np.diff(df['t'].values))) if len(df) > 1 else 1.0
    W_MIN = 30.0
    w_n = max(3, int(round(W_MIN*60.0/max(dt_full, 1e-6))))
    if w_n % 2 == 0:
        w_n += 1
    bias_roll = trq.rolling(window=w_n, center=True, min_periods=1).median()
    P = ((trq-bias_roll).abs()*omega).clip(lower=0)
    tm = df['t'].values - t0
    tamb = _col(df, 'Ambient_Temp(deg C)').iloc[i0:i1+1].dropna()
    T0 = (tamb.median()+273.15) if len(tamb) else 300.0

    # time-varying ambient temperature over the full recording (not just
    # the touchdown window used for the scalar T0 above)
    tamb_full = _col(df, 'Ambient_Temp(deg C)')
    valid = tamb_full.notna()
    if valid.sum() >= 2:
        _T0_interp = _interp(tm[valid.values], tamb_full[valid].values + 273.15,
                             fill=(tamb_full[valid].iloc[0] + 273.15,
                                   tamb_full[valid].iloc[-1] + 273.15))
    else:   # not enough data -> scalar T0 everywhere
        _T0_interp = _interp([tm.min(), tm.max()], [T0, T0], fill=(T0, T0))
    _t_min_C = float(tm.min())

    # --- remove the rectified-bias artefact of the take-off roll -------------
    # P = |trq - bias_roll|*omega: when the rolling median lags sensor drift the
    # residual can carry either sign, and abs() turns it positive -- multiplied
    # by a large omega during a take-off roll this manufactures a spurious
    # "braking event" (confirmed on the reference flight). Brakes can't be
    # applied while accelerating hard, and taxi accel (~0.3-0.5 m/s^2) is well
    # clear of a take-off roll (~1.5-2.5), so the threshold separates them.
    # Data conditioning, not a fitted correction.
    if SUPPRESS_TAKEOFF_ROLL:
        v_g = gspd.values
        a_g = np.gradient(v_g, tm)
        k5 = max(3, int(round(5.0/max(dt_full, 1e-6))))
        a_s = pd.Series(a_g).rolling(k5, center=True, min_periods=1).median().values
        rolling_out = (v_g > 5.0*KT2MS) & (a_s > TAKEOFF_ACCEL_THRESH)
        if rolling_out.any():
            E_removed = float(np.trapezoid(P.values*rolling_out, tm))/1e6
            P = P.where(~rolling_out, 0.0)
            print(f"[{SELECTED}] take-off roll suppression: "
                  f"{int(rolling_out.sum())} samples zeroed, "
                  f"{E_removed:.2f} MJ/brake of rectified-bias power removed")

    _P_interp = _interp(tm, P.values)
    _v_interp = _interp(tm, gspd.values)

    # debounced on-ground flag: bridges brief dips below threshold (e.g.
    # anti-skid modulation during heavy braking) shorter than MIN_GAP_S,
    # so a real braking event doesn't flicker the phase back and forth
    WHEEL_STOP_KT_ = 2.0
    MIN_GAP_S = 5.0
    on_ground_raw = (v_wheel.values > WHEEL_STOP_KT_*KT2MS)
    dt_med = float(np.median(np.diff(tm))) if len(tm) > 1 else 1.0
    min_gap_n = max(1, int(round(MIN_GAP_S / max(dt_med, 1e-6))))
    on_ground_deb = on_ground_raw.copy()
    i = 0
    while i < len(on_ground_deb):
        if not on_ground_deb[i]:
            j = i
            while j < len(on_ground_deb) and not on_ground_deb[j]:
                j += 1
            if (j - i) < min_gap_n and i > 0 and j < len(on_ground_deb):
                on_ground_deb[i:j] = True
            i = j
        else:
            i += 1
    _on_ground_interp = _interp(tm, on_ground_deb.astype(float), fill=(0.0, 0.0))

    # --- lift-off: last on-ground -> airborne transition before touchdown ----
    # (the take-off of the leg we simulate; wheel speed drops to 0 there)
    _t_liftoff = None
    pre = np.where(tm < 0)[0]
    if len(pre) > 1:
        og = on_ground_deb[pre]
        falls = np.where((og[:-1]) & (~og[1:]))[0]
        if len(falls):
            _t_liftoff = float(tm[pre[falls[-1] + 1]])

    # --- approach: last descent through H_APPROACH_FT before touchdown ------
    _alt_interp = None
    alt_ft = None
    norm = {_norm(c): c for c in df.columns}
    for cand in ('Pressure_Altitude(ft)', 'Altitude(ft)', 'Alt(ft)',
                 'Baro_Altitude(ft)', 'ALT_STD', 'Altitude', 'Alt', 'ALT'):
        col = norm.get(_norm(cand))
        if col is not None and df[col].notna().sum() >= 2:
            alt_ft = df[col].astype(float).values
            _alt_interp = _interp(tm, np.nan_to_num(alt_ft, nan=0.0), fill=(0.0, 0.0))
            print(f"[{SELECTED}] altitude column used for approach window: '{col}'")
            break

    _t_app_start = None
    if alt_ft is not None:
        m_pre = tm < 0
        a = np.nan_to_num(alt_ft[m_pre], nan=0.0)
        tp = tm[m_pre]
        cross = np.where((a[:-1] > H_APPROACH_FT) & (a[1:] <= H_APPROACH_FT))[0]
        if len(cross):
            _t_app_start = float(tp[cross[-1] + 1])
    if _t_app_start is None:
        _t_app_start = -APPROACH_FALLBACK_S
        if alt_ft is None:
            print(f"[{SELECTED}] no altitude column -> approach cooling window "
                  f"falls back to the last {APPROACH_FALLBACK_S:.0f}s before "
                  f"touchdown (set APPROACH_FALLBACK_S to change).")

    print(f"[{SELECTED}] forced-cooling windows: "
          f"retraction [{(_t_liftoff or float('nan'))/60:+.1f}, "
          f"{((_t_liftoff + T_RETRACT_S) if _t_liftoff is not None else float('nan'))/60:+.1f}] min | "
          f"approach [{_t_app_start/60:+.1f}, 0.0] min")

    _t_stop_C = df['t'].iloc[i1] - t0
    _t_speed0_C = df['t'].iloc[i2] - t0
    _t_record = tm[-1]

    # Real taxi duration (rollout end -> sustained full stop), replacing the
    # T_TAXI=300s guess. Computed here so every caller of load_csv() gets it
    # automatically, whether or not they know to override T_TAXI themselves.
    if _t_speed0_C > _t_stop_C:
        T_TAXI = _t_speed0_C - _t_stop_C
        print(f"[{SELECTED}] real taxi duration detected: {T_TAXI:.0f}s "
              f"(overriding the default 300s assumption)")
    bt = _col(df, f'Wheel{w}_Brk_Temp(deg C)'); valid = bt.notna()
    _btms = (df['t'].values[valid]-t0, bt[valid].values)
    init = bt.iloc[:i0+1].dropna()
    m = (tm >= 0) & (tm <= _t_stop_C)
    E = np.trapezoid(P.values[m], tm[m])/1e6
    print(f"[{SELECTED}] rollout {_t_stop_C:.0f}s | speed=0 at {_t_speed0_C:.0f}s "
          f"| recording {_t_record/60:.1f} min | E={E:.1f} MJ/brake | T0={T0-273.15:.0f}C")
    return init.iloc[-1] if len(init) else T0-273.15

def power(t):     return float(_P_interp(t))
def airspeed(t):  return float(_v_interp(t))
def on_ground(t): return _on_ground_interp(t) > 0.5
def T0_of(t):
    """Recorded OAT [K] at t. Gear-stowed brakes don't see this -- rhs()
    substitutes T_BAY_K while airborne with the bay closed."""
    return float(_T0_interp(t))
def t_stop():     return _t_stop_C
def t_speed0():   return _t_speed0_C
def t_record():   return _t_record
def t_min():      return _t_min_C

def t_liftoff():  return _t_liftoff
def t_app_start():return _t_app_start
def altitude(t):
    """Altitude [ft] at t, or None if the recording has no altitude column."""
    return None if _alt_interp is None else float(_alt_interp(t))

def in_cooling_window(t):
    """True inside an airborne forced-convection window (gear retraction, or
    approach below H_APPROACH_FT). Only consulted while airborne."""
    if not FLIGHT_FORCED_COOLING:
        return False
    if _t_liftoff is not None and _t_liftoff <= t <= _t_liftoff + T_RETRACT_S:
        return True
    if _t_app_start is not None and _t_app_start <= t <= 0.0:
        return True
    return False

def btms_at(t_target):
    """Nearest measured BTMS to t_target. Use this -- NOT load_csv()'s return
    value, which is the reading at TOUCHDOWN -- to seed run_full()."""
    if _btms is None:
        return None
    tm, Tm = _btms
    i = np.argmin(np.abs(tm - t_target))
    return float(Tm[i])

# =============================================================================
# 9. ODE SYSTEM
# =============================================================================
def carrier_h(t, phase):
    if phase == 'landing': return max(h_forced(airspeed(t)), h_nat)
    if phase == 'taxi':    return max(h_forced(V_TAXI_KT*KT2MS), h_nat)
    return h_nat

def rhs(t, y, phase):
    dT = np.zeros(N)
    Td = y[:ND]
    Tt, Tw, Ts = y[I_TUBE], y[I_WHEEL], y[I_SENS]
    Tsh, Tp     = y[I_SHIELD], y[I_PIST]

    # 'flight' maps onto the existing regimes: airborne -> 'parking'-like,
    # on ground not braking -> 'taxi'-like, on ground braking -> 'landing'-like.
    # Braking is decided by TORQUE, not speed: a heavy-braking event can sit
    # right at the 20kt landing/taxi threshold (confirmed on real data).
    # Computed before Tamb -- the gear-bay exception below depends on it.
    if phase == 'flight':
        airborne = not on_ground(t)
        clamped  = airborne or (power(t) > P_BRAKE_THRESH)
        cooling_win = airborne and in_cooling_window(t)
    else:
        airborne = False
        clamped  = None   # unused outside 'flight'
        cooling_win = False

    # Ambient: normally T0_of(t) (real, time-varying OAT). Exception: gear
    # fully retracted, bay doors closed -> brakes see T_BAY_K instead,
    # reverting to real ambient once the approach window reopens the bay.
    if phase == 'flight' and airborne and not cooling_win:
        Tamb = T_BAY_K
    else:
        Tamb = T0_of(t)

    # 9.1 friction (landing, taxi & flight from CSV; none in parking)
    if phase != 'parking':
        q = power(t) / 8.0
        if phase == 'flight':
            q *= P_FLIGHT_SCALE
        for k in range(8):
            dT[k] += 0.5*q; dT[k+1] += 0.5*q

    # 9.2 inter-disc conduction
    if phase == 'flight':
        G = G_brake if clamped else G_cool
    else:
        G = G_cool if phase == 'taxi' else G_brake
    for k in range(ND-1):
        f = G*(Td[k+1]-Td[k]); dT[k] += f; dT[k+1] -= f

    # 9.3 disc -> carrier conduction
    for i in range(ND):
        if is_rotor[i]:
            f = G_RW*(Tw-Td[i]); dT[i] += f; dT[I_WHEEL] -= f
        else:
            f = G_ST*(Tt-Td[i]); dT[i] += f; dT[I_TUBE]  -= f

    # 9.4 radiation: disc->tube, disc->shield, shield->wheel
    for i in range(ND):
        ft  = eps_DT*SIGMA*A_in *(Td[i]**4-Tt**4);  dT[i] -= ft;  dT[I_TUBE]   += ft
        fsh = eps_DS*SIGMA*A_out*(Td[i]**4-Tsh**4); dT[i] -= fsh; dT[I_SHIELD] += fsh
    fsw = eps_SW*SIGMA*A_SW*(Tsh**4-Tw**4); dT[I_SHIELD] -= fsw; dT[I_WHEEL] += fsw
    fcw = G_SW*(Tsh-Tw);                    dT[I_SHIELD] -= fcw; dT[I_WHEEL] += fcw

    # 9.5 end-face radiation (S1 and S5 only, different areas)
    dT[0]    -= eps_end*SIGMA*A_fend_S1*(Td[0]**4 - Tamb**4)
    dT[ND-1] -= eps_end*SIGMA*A_fend_S5*(Td[ND-1]**4 - Tamb**4)

    # 9.6 S1 <-> piston housing
    if phase == 'flight':
        s1_conductive = clamped
    else:
        s1_conductive = (phase != 'taxi')
    if s1_conductive:
        fS1 = G_SP*(Td[0]-Tp)
    else:
        fS1 = eps_S1P*SIGMA*A_S1P*(Td[0]**4-Tp**4)
    dT[0] -= fS1; dT[I_PIST] += fS1

    # 9.7 tube -> housing; tube -> sensor (back-reaction)
    fph = G_PH*(Tt-Tp);   dT[I_TUBE] -= fph; dT[I_PIST] += fph
    dT[I_TUBE] -= G_sens*(Tt-Ts)

    # 9.8 disc airflow convection (landing & taxi; real speed if 'flight')
    if phase == 'flight':
        # airborne cooling windows (retraction / approach) are treated like
        # roll-out: real airstream over the assembly at aircraft speed
        do_disc_conv = (not airborne) or cooling_win
        v_conv = airspeed(t)
    else:
        do_disc_conv = phase in ('landing', 'taxi')
        v_conv = airspeed(t) if phase == 'landing' else V_TAXI_KT*KT2MS
    if do_disc_conv:
        hd = h_disc_airflow(v_conv)
        for i in range(ND):
            dT[i] -= hd * A_disc * (Td[i] - Tamb)

    # 9.9 fan cooling on discs (optional; fan state unknown for 'flight')
    fan = (phase == 'taxi' and FAN_ON) or (phase == 'parking' and GROUND_FAN_ON)
    if fan:
        for i in range(ND):
            dT[i] -= h_fan * A_disc * (Td[i] - Tamb)

    # 9.10 ambient losses: tube, wheel, housing (shield has NONE).
    # H_NAT_FLIGHT applies to 'flight' only; carrier_h() keeps touchdown h_nat.
    if phase == 'flight':
        h = (max(h_forced(airspeed(t)), H_NAT_FLIGHT)
             if (not airborne or cooling_win) else H_NAT_FLIGHT)
    else:
        h = carrier_h(t, phase)
    dT[I_TUBE]  -= h*A_ct*(Tt-Tamb) + eps_T *SIGMA*A_ct*(Tt**4-Tamb**4)
    dT[I_WHEEL] -= h*A_cw*(Tw-Tamb) + eps_W *SIGMA*A_cw*(Tw**4-Tamb**4)
    dT[I_PIST]  -= h*A_cp*(Tp-Tamb) + eps_PH*SIGMA*A_cp*(Tp**4-Tamb**4)

    # 9.11 heat capacities
    dT[:ND]      /= C_disc
    dT[I_TUBE]   /= (C_tube * C_TUBE_FLIGHT_MULT if phase == 'flight' else C_tube)
    dT[I_WHEEL]  /= C_wheel
    dT[I_SHIELD] /= C_shield
    dT[I_PIST]   /= C_pist

    # 9.12 sensor
    dT[I_SENS] = G_sens*(Tt-Ts) / C_sens
    return dT

# =============================================================================
# 10. INTEGRATION
# =============================================================================
def _integrate_post_touchdown(y0, kw):
    """Landing -> taxi -> parking. Shared by run() and run_full() -- they
    only differ in what happens BEFORE this (nothing, vs a 'flight' leg),
    so the two used to duplicate this block verbatim. Returns time,
    temperature in KELVIN (callers convert once, after any pre-touchdown
    leg has been concatenated on) and the two phase-boundary instants."""
    t_land = t_stop()
    t_taxi_end = t_land + T_TAXI
    t_end = t_record() if t_record() else max(T_END, t_taxi_end+60)
    t_end = max(t_end, t_taxi_end+60)  # but at least cover taxi
    s1 = solve_ivp(rhs, (0, t_land), y0, args=('landing',), max_step=0.5, **kw)
    s2 = solve_ivp(rhs, (t_land, t_taxi_end), s1.y[:,-1], args=('taxi',), **kw)
    s3 = solve_ivp(rhs, (t_taxi_end, t_end), s2.y[:,-1], args=('parking',), **kw)
    t1 = np.linspace(0, t_land, max(int(t_land*10),400))
    t2 = np.linspace(t_land, t_taxi_end, 400)
    t3 = np.linspace(t_taxi_end, t_end, 1200)
    t = np.concatenate([t1,t2,t3])
    Tc_K = np.concatenate([s1.sol(t1), s2.sol(t2), s3.sol(t3)], axis=1)
    return t, Tc_K, t_land, t_taxi_end

def run(T_init):
    y0 = np.full(N, T_init+273.15)
    kw = dict(method='LSODA', dense_output=True, rtol=1e-6, atol=1e-3)
    t, Tc_K, t_land, t_taxi_end = _integrate_post_touchdown(y0, kw)
    return t, Tc_K - 273.15, (t_land, t_taxi_end)

class _PiecewiseFlight:
    """Dense-output shim over several solve_ivp results covering one span."""
    def __init__(self, sols, spans):
        self.sols, self.spans = sols, spans
        self.y = sols[-1].y
    def sol(self, tq):
        tq = np.atleast_1d(tq)
        out = np.empty((N, tq.size))
        for i, (a, b) in enumerate(self.spans):
            msk = (tq >= a) & (tq <= b) if i == len(self.spans)-1 else \
                  (tq >= a) & (tq < b)
            if msk.any():
                out[:, msk] = self.sols[i].sol(tq[msk])
        return out


def _integrate_flight(y0, t_start, t_end, kw):
    """Integrate the 'flight' phase, breaking the span at the forced-convection
    window boundaries. The retraction window is only ~T_RETRACT_S (20s) wide,
    so a single pass at max_step=8s would step across its edges and smear the
    discontinuity in h; inside a window the step drops to 1s.
    """
    brk = [t_start]
    for b in (t_liftoff(),
              (t_liftoff() + T_RETRACT_S) if t_liftoff() is not None else None,
              t_app_start()):
        if b is not None and t_start < b < t_end:
            brk.append(b)
    brk = sorted(set(brk)) + [t_end]

    seg_sols, seg_spans = [], []
    y_cur = y0
    for a, b in zip(brk[:-1], brk[1:]):
        if b <= a:
            continue
        ms = 1.0 if in_cooling_window(0.5*(a + b)) else 8.0
        sol = solve_ivp(rhs, (a, b), y_cur, args=('flight',), max_step=ms, **kw)
        seg_sols.append(sol); seg_spans.append((a, b))
        y_cur = sol.y[:, -1]
    return _PiecewiseFlight(seg_sols, seg_spans)


def run_flight_window(T_init, t_start, t_end, n_out=2000):
    """Integrate ONLY the 'flight' phase over [t_start, t_end] (both < 0).
    Lets a calibration score one braking segment in seconds instead of the
    whole recording. Same integrator as run_full(), not a copy.
    """
    kw = dict(method='LSODA', dense_output=True, rtol=1e-6, atol=1e-3)
    y0 = np.full(N, T_init + 273.15)
    sol = _integrate_flight(y0, t_start, t_end, kw)
    t = np.linspace(t_start, t_end, n_out)
    return t, sol.sol(t) - 273.15


def run_full(T_init, t_start=None):
    """Same physics as run(), but starts at t_min() (negative, before
    touchdown) and adds a 'flight' phase over [t_start, 0]. Fans are always
    off during 'flight'; ambient is time-varying (T0_of), which matters a lot
    pre-touchdown; heat generation stays torque-driven, so any earlier braking
    in the recording is captured automatically. See rhs() for how 'flight'
    maps onto the landing/taxi/parking regimes.

    t_start: where to begin (default t_min()). Pass it explicitly to skip past
    an unrelated earlier flight leg in the same recording.
    """
    if t_start is None:
        t_start = t_min()
    if t_start is None or t_start >= 0:
        # no pre-touchdown data available -> nothing to add, same as run()
        return run(T_init)

    # T_init is the temperature AT TOUCHDOWN (correct initial condition for
    # run(), which starts there) -- wrong here, since t_start can be hours
    # earlier with a very different real temperature. Look up the actual
    # measured BTMS value near t_start instead, if available.
    T_init_flight = T_init
    if _btms is not None:
        tm_b, Tm_b = _btms
        near = np.abs(tm_b - t_start) < 120.0   # within 2 min of t_start
        if near.any():
            T_init_flight = float(np.median(Tm_b[near]))

    y0 = np.full(N, T_init_flight+273.15)
    kw = dict(method='LSODA', dense_output=True, rtol=1e-6, atol=1e-3)
    s0 = _integrate_flight(y0, t_start, 0.0, kw)
    t_post, Tc_post_K, t_land, t_taxi_end = _integrate_post_touchdown(
        s0.y[:,-1], kw)

    t0_ = np.linspace(t_start, 0, max(int(abs(t_start)/20), 200))
    # the coarse full-flight grid (1 point per ~20s) would barely sample a
    # 20s retraction window -> add dense points inside both windows
    extra = []
    if t_liftoff() is not None and t_start < t_liftoff() < 0:
        extra.append(np.linspace(max(t_liftoff()-5, t_start),
                                 min(t_liftoff()+T_RETRACT_S+5, 0), 120))
    if t_app_start() is not None and t_start < t_app_start() < 0:
        extra.append(np.linspace(t_app_start(), 0, 200))
    if extra:
        t0_ = np.unique(np.concatenate([t0_] + extra))
    t = np.concatenate([t0_, t_post])
    Tc = np.concatenate([s0.sol(t0_), Tc_post_K], axis=1) - 273.15
    return t, Tc, (t_land, t_taxi_end, t_start)

def btms_error(t, Ts):
    if _btms is None: return None
    tm, Tm = _btms
    mask = (tm >= 0) & (tm <= t.max())
    tmm, Tmm = tm[mask], Tm[mask]
    Tmod = np.interp(tmm, t, Ts)
    abs_err = Tmod - Tmm
    rel_err = 100.0*abs_err/np.clip(Tmm, 1.0, None)
    rmse = float(np.sqrt(np.mean(abs_err**2)))
    mape = float(np.mean(np.abs(rel_err)))
    return tmm, Tmm, Tmod, abs_err, rel_err, rmse, mape

# =============================================================================
# 11. MAIN
# =============================================================================
if __name__ == '__main__':
    fl = FLIGHTS[SELECTED]
    if not os.path.isfile(fl['path']):
        raise SystemExit(f"[ERROR] CSV not found: {fl['path']}")
    T_init = load_csv(fl['path'], fl['start'], WHEEL_ID)
    # T_TAXI is now set to the real detected taxi duration inside load_csv()
    # itself (see the note there) -- no override needed here any more.

    t, Tc, (t_land, t_taxi_end) = run(T_init)
    iR2 = labels.index('R2')

    err = btms_error(t, Tc[I_SENS])
    if err is not None:
        tmm, Tmm, Tmod, abs_err, rel_err, rmse, mape = err
        print(f"BTMS fit: RMSE={rmse:.1f} C | MAPE={mape:.1f} % | max|abs|={np.max(np.abs(abs_err)):.1f} C "
              f"| max|rel|={np.max(np.abs(rel_err)):.1f} %")
    print(f"Peak  R2={Tc[iR2].max():.0f}  tube={Tc[I_TUBE].max():.0f}  "
          f"wheel={Tc[I_WHEEL].max():.0f}  shield={Tc[I_SHIELD].max():.0f}  "
          f"housing={Tc[I_PIST].max():.0f}  sensor={Tc[I_SENS].max():.0f} [C]")

    TDISP = (t_record() or t.max())/60.0

    # Relative error, smoothed for display. See _smooth() below: same
    # Savitzky-Golay approach used elsewhere in this project for the BTMS
    # derivative, window given in SECONDS.
    SMOOTH_S = 60.0

    def _smooth(t_arr, y_arr, window_s=SMOOTH_S, polyorder=2):
        if len(t_arr) < polyorder + 2:
            return y_arr
        dt = float(np.median(np.diff(t_arr))) if len(t_arr) > 1 else 1.0
        n = max(polyorder + 2, int(round(window_s / max(dt, 1e-9))))
        if n % 2 == 0:
            n += 1
        n = min(n, len(y_arr) - (1 - len(y_arr) % 2))
        if n <= polyorder or n > len(y_arr):
            return y_arr
        return savgol_filter(y_arr, window_length=n, polyorder=polyorder)

    def plines(a, u=1.0):
        """Vertical markers for the landing->taxi and taxi->parking phase
        boundaries. Labelled so every subplot that already calls
        a.legend() picks them up automatically."""
        a.axvline(t_land/u, color='firebrick', ls='--', lw=1.6, zorder=3,
                  label='Rollout end (taxi start)')
        a.axvline(t_taxi_end/u, color='darkorange', ls='--', lw=1.6, zorder=3,
                  label='Taxi end (parking start)')

    # (node index, label, colour, linestyle, linewidth) -- one table so a given
    # node keeps the same look in every figure below.
    NODE_STYLES = [(I_TUBE,   'Tube',    'k', '--', 1.8),
                   (I_WHEEL,  'Wheel',   'b', '--', 1.4),
                   (I_SHIELD, 'Shield',  'm', '-',  1.4),
                   (I_PIST,   'Housing', 'c', '-',  1.4),
                   (I_SENS,   'BTMS',    'g', '-',  2.0)]

    def pnodes(a, x, scale=1.0):
        """Plot the carrier/sensor nodes on `a` against `x` (T divided by
        `scale`, =1 for degC, =T_max for the normalised figure)."""
        for idx, lab, col, ls, lw in NODE_STYLES:
            a.plot(x, Tc[idx]/scale, color=col, ls=ls, lw=lw, label=lab)

    # ===================== FIGURE 1: absolute values =========================
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))

    # [0,0] landing + taxi
    a = ax[0,0]
    for i,lab in enumerate(labels):
        if lab in ('R2','S5'): a.plot(t, Tc[i], lw=1.2, label=lab)
    pnodes(a, t); plines(a)
    a.set(xlim=(0,600),xlabel='Time [s]',ylabel='T [°C]',
          title='Landing + taxi'); a.legend(fontsize=9, ncol=2, framealpha=0.9); a.grid(alpha=0.3)

    # [0,1] full cycle
    a = ax[0,1]
    a.plot(t/60, Tc[iR2],'C0-',lw=2,label='R2')
    pnodes(a, t/60)
    plines(a,60); a.set(xlim=(0,TDISP),xlabel='Time [min]',ylabel='T [°C]',
          title=f'Full recording ({TDISP:.0f} min)')
    a.legend(fontsize=9, ncol=2, framealpha=0.9); a.grid(alpha=0.3)

    # [0,2] BTMS model vs measured + rel error
    a = ax[0,2]
    a.plot(t/60, Tc[I_SENS],'g-',lw=2,label='BTMS model')
    if _btms is not None:
        tm,Tm = _btms; mm = tm >= 0
        a.plot(tm[mm]/60, Tm[mm],'r.',ms=2,alpha=0.5,label='BTMS measured')
    a.set(xlim=(0,TDISP),xlabel='Time [min]',ylabel='T [°C]'); a.grid(alpha=0.3)
    ttl = 'BTMS: model vs data + error'
    if err is not None:
        a2 = a.twinx()
        a2.plot(tmm/60, _smooth(tmm, rel_err), 'purple', lw=1.5, alpha=0.9,
                label='rel. error [%]')
        a2.axhline(0,color='purple',lw=0.5,ls=':')
        a2.set_ylabel('Relative error [%]',color='purple')
        a2.tick_params(axis='y',colors='purple')
        l1,lb1 = a.get_legend_handles_labels(); l2,lb2 = a2.get_legend_handles_labels()
        a.legend(l1+l2, lb1+lb2, fontsize=9, loc='lower right', framealpha=0.9)
        ttl = f'BTMS: model vs data (RMSE={rmse:.1f} °C, MAPE={mape:.1f} %)'
    else: a.legend()
    a.set_title(ttl)

    # [1,0] brake power
    a = ax[1,0]
    ts = t[t <= t_land+10]
    a.plot(ts, [power(x)/1000 for x in ts],'g-',lw=1.5)
    a.set(xlabel='Time [s]',ylabel='Power [kW]',title='Brake power (one brake)')
    a.grid(alpha=0.3)

    # [1,1] ground speed
    a = ax[1,1]
    tsp = np.linspace(0, TDISP*60, 1200)
    a.plot(tsp/60, [airspeed(x)/KT2MS for x in tsp],'b-',lw=1.5,label='ground speed')
    a.axhline(V_TAXI_KT, color='steelblue', ls=':', lw=1.3,
              label=f'taxi speed ({V_TAXI_KT:.0f} kt)')
    plines(a,60)
    a.set(xlim=(0,TDISP),xlabel='Time [min]',ylabel='Ground speed [kt]',
          title='Aircraft ground speed')
    a.legend(fontsize=9, framealpha=0.9); a.grid(alpha=0.3)

    ax[1,2].axis('off')

    fig.suptitle(f'BTEM v5 - {SELECTED} - 14 nodes, 3 phases (Method C)', fontsize=13)
    fig.tight_layout()

    # ===================== FIGURE 2: normalised values =======================
    # x-axis: t / t_total  (0 to 1, full recording)
    # y-axis: T / T_max    (temperature normalised by max measured BTMS)
    t_total = t[-1]                                # total simulation = recording duration
    # T_max = max measured BTMS if available, else max model BTMS
    if _btms is not None:
        tm_b, Tm_b = _btms; mm = tm_b >= 0
        T_max = np.max(Tm_b[mm])
    else:
        T_max = np.max(Tc[I_SENS])

    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))

    # [0] normalised model curves (all nodes)
    a = ax2[0]
    t_norm = t / t_total
    a.plot(t_norm, Tc[iR2]/T_max, color='C0', ls='-', lw=1.5, label='R2')
    pnodes(a, t_norm, scale=T_max)
    a.axhline(1.0, color='gray', ls='--', lw=1.2, label='T_max')
    a.set(xlim=(0, 1), xlabel='t / t_total', ylabel='T / T_max',
          title='Normalised temperatures (all nodes)')
    a.legend(fontsize=9, ncol=2, framealpha=0.9); a.grid(alpha=0.3)

    # [1] normalised BTMS model vs measured + relative error + ground speed
    a = ax2[1]
    Ts_norm = Tc[I_SENS] / T_max
    a.plot(t_norm, Ts_norm, 'g-', lw=2, label='BTMS model')
    if _btms is not None:
        t_norm_meas = tm_b[mm] / t_total
        T_norm_meas = Tm_b[mm] / T_max
        a.plot(t_norm_meas, T_norm_meas, 'r.', ms=2, alpha=0.5, label='BTMS measured')

    # ground speed, normalised by its own max over the recording -> shares
    # the same [0, ~1] axis as T/T_max, no extra axis/units needed
    v_curve = np.array([airspeed(x) for x in t])
    v_max = v_curve.max()
    v_norm = v_curve / v_max if v_max > 0 else v_curve
    a.plot(t_norm, v_norm, 'b-', lw=1.2, alpha=0.6, label='ground speed (norm.)')

    # vertical markers: ground speed = 20 kt (t_land, start of taxi) and
    # ground speed = 0 kt (full stop)
    t_v20 = t_land / t_total
    t_v0  = (t_speed0() or t_taxi_end) / t_total
    a.axvline(t_v20, color='firebrick', ls='--', lw=1.6, zorder=3,
              label='v = 20 kt (taxi start)')
    a.axvline(t_v0,  color='darkorange', ls='--', lw=1.6, zorder=3,
              label='v = 0 kt (parking start)')

    handles, labels_ = a.get_legend_handles_labels()
    if _btms is not None:
        # relative error on twin axis
        Tmod_interp = np.interp(tm_b[mm], t, Tc[I_SENS])
        rel_err_norm = 100.0*(Tmod_interp - Tm_b[mm]) / np.clip(Tm_b[mm], 1.0, None)
        a2 = a.twinx()
        a2.plot(t_norm_meas, _smooth(tm_b[mm], rel_err_norm), 'purple',
                lw=1.5, alpha=0.9, label='rel. error [%]')
        a2.axhline(0, color='purple', lw=0.5, ls=':')
        a2.set_ylabel('Relative error [%]', color='purple')
        a2.tick_params(axis='y', colors='purple')
        h2, l2 = a2.get_legend_handles_labels()
        handles += h2; labels_ += l2

    a.legend(handles, labels_, fontsize=9, loc='lower right', framealpha=0.9)
    a.axhline(1.0, color='gray', ls='--', lw=1.2)
    a.set(xlim=(0, 1), xlabel='t / t_total', ylabel='Normalised value  (T/T_max  or  v/v_max)',
          title=f'Normalised BTMS: model vs data vs ground speed (MAPE={mape:.1f} %)'
                if err is not None else 'Normalised BTMS: model vs data vs ground speed')
    a.grid(alpha=0.3)

    fig2.suptitle(f'BTEM v5 [{SELECTED}] - Normalised view', fontsize=12)
    fig2.tight_layout()
    plt.show()