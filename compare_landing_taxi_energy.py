# =============================================================================
# SANITIZED PUBLIC VERSION
# All calibrated parameter values have been replaced with generic round
# placeholders, and all flight identifiers, timestamps and paths with
# examples. Calibrate on your own data before any quantitative use.
# =============================================================================
# -*- coding: utf-8 -*-
"""
Landing-roll and taxi energy: torque-based (ground truth) vs.
kinematically reconstructed (force-balance for landing; G(v) and, for
taxi, a third method comparing kinetic energy per deceleration phase),
on FLIGHT_A.

Torque-based power comes from btem_v5_full_flight.py's power(t),
unchanged (never touches thermal simulation). The kinematic methods are
the ones from predict_day_from_csv.py's build_segment(): F_opposing(),
the sub-stepped landing force balance, and the G(v) taxi calibration.

TOUCHDOWN ORIGIN: find_touchdown() (per Ugo), not the touchdown implied
by btem_v5_full_flight.py's own fl['start']-based load_csv(). Those
two can differ by a few seconds, and m.power()/m.t_stop()/m.t_speed0()
are only available relative to load_csv()'s own origin -- so this script
computes the offset between the two touchdown instants once, and shifts
every m.* query by it, rather than re-deriving power from raw torque
itself (which would throw away load_csv()'s own bias-removal and
take-off-roll suppression). Everything downstream of that (mass, ground
speed, reverser flag, the two reconstructions) is then expressed
relative to find_touchdown()'s instant, consistently.

WHY REIMPLEMENTED RATHER THAN CALLING build_segment() DIRECTLY.
build_segment() is written for the commercial-data pipeline: it expects
df['GROSS WEIGHT'] and 'Thrust_Reverser_Cowl_Position'. FLIGHT_A's CSV uses
'Aircraft WEIGHT(Kg)' and 'Thrust_Reverser_Deploy' instead. Every
constant (RHO_AIR, A_WING, C_D, MU_RR, A_SP, C_SP, K_RT, V_TH_REV,
N_SUB_LAND, N_BRAKES) is read LIVE from
predict_day_from_csv.configure_constants(), not a hardcoded second copy.
The equations (F_opposing, the sub-stepped landing force balance, the
G(v) taxi block) are copied verbatim from build_segment() -- only the
data loading (column names) and the touchdown origin (via
find_touchdown(), see above) change.

GRADE TERM (per Ugo): intentionally omitted. F_opposing() in
predict_day_from_csv.py has no runway-grade term (F_grade = m*g*sin
(gamma) from the paper's general equation) -- it implicitly assumes
gamma=0, flat runway. Reproduced as-is here, matching the real code.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

import btem_v5_full_flight as m
import predict_day_from_csv as pdc

FLIGHT = 'FLIGHT_B'
WHEEL_ID = 4   # <-- double-check this against the wheel you actually want

pdc.configure_constants()   # RHO_AIR, A_WING, C_D, MU_RR, A_SP, C_SP,
                             # K_RT, V_TH_REV, N_SUB_LAND, N_BRAKES -- all
                             # read live from pdc.* below, never copied
KT2MS = pdc.KT2MS

# =============================================================================
# LOAD -- touchdown origin from find_touchdown() (per Ugo); m.power() etc.
# stay usable via a computed time offset, see the module docstring above.
# =============================================================================
fl = m.FLIGHTS[FLIGHT]
m.SELECTED = FLIGHT
m.load_csv(fl['path'], fl['start'], WHEEL_ID)   # needed for power()/airspeed()

df = pd.read_csv(fl['path'])
df['GMTs'] = pd.to_datetime(df['GMTs'])
t_full = (df['GMTs'] - df['GMTs'].iloc[0]).dt.total_seconds().values
gspd_kt = df['Ground_Speed(Kt)'].fillna(0).values   # knots, for find_touchdown()

# find_touchdown()'s own instant -- the new, authoritative t=0.
i_td = pdc.find_touchdown(t_full, gspd_kt, 0, len(df))
if i_td is None:
    raise SystemExit(f"find_touchdown() found no touchdown in {FLIGHT}'s CSV.")
t0_fb = t_full[i_td]

# load_csv()'s own touchdown instant, re-derived the same way it computes
# it internally, purely to get the offset -- m.power() etc. are not
# re-implemented, just queried at a shifted time.
i0_m = (df['GMTs'] >= pd.to_datetime(fl['start'])).idxmax()
t0_m = t_full[i0_m]
delta = t0_fb - t0_m   # add this to a new-frame time before calling any m.* function

print("=" * 74)
print(f"Landing and taxi energy: torque vs. kinematic reconstruction -- "
      f"{FLIGHT}, wheel {WHEEL_ID}")
print("=" * 74)
print(f"  touchdown (find_touchdown)      : t_full = {t0_fb:.1f} s")
print(f"  touchdown (load_csv's own)      : t_full = {t0_m:.1f} s")
print(f"  offset applied to every m.* call: {delta:+.1f} s")


def m_power(t_new):
    """m.power(), queried at the time that corresponds to t_new in THIS
    script's find_touchdown()-based frame."""
    return m.power(t_new + delta)


T_STOP = m.t_stop() - delta      # end of roll-out, start of taxi
T_END = m.t_speed0() - delta     # ground speed reaches zero, end of taxi

tm_raw = t_full - t0_fb   # find_touchdown()-relative time, for everything below

mass_clean = df['Aircraft WEIGHT(Kg)'].replace(0, np.nan).ffill().bfill().values
# Despite the column name, FLIGHT_A's 'Aircraft WEIGHT(Kg)' is actually in lb
# (confirmed by Ugo -- raw values ~116,200, impossible in kg for an A320,
# ~52,700 kg once converted). Auto-detected here with the exact same rule
# build_segment() already uses for the commercial pipeline (MASS_LB_
# THRESHOLD=80,000, read live via configure_constants()), rather than
# hardcoding "always lb" for this one flight -- keeps this working
# unchanged if a differently-labelled test-flight CSV is ever loaded here.
is_lb = np.nanmax(mass_clean) > pdc.MASS_LB_THRESHOLD
mass_kg = mass_clean * (pdc.LB2KG if is_lb else 1.0)
print(f"  mass column detected as: {'lb -> converted to kg' if is_lb else 'kg, unchanged'}")
v_ms_raw = df['Ground_Speed(Kt)'].fillna(0).values * KT2MS
rev_active_raw = (pd.to_numeric(df['Thrust_Reverser_Deploy'],
                                errors='coerce').fillna(0.0).values > 0)

print(f"  roll-out : [0, {T_STOP:.0f}] s")
print(f"  taxi     : [{T_STOP:.0f}, {T_END:.0f}] s")


def F_opposing(v, mass):
    """Verbatim from predict_day_from_csv.py -- no grade term, see the
    module docstring above."""
    return (0.5 * pdc.RHO_AIR * pdc.A_WING * pdc.C_D * v**2
            + pdc.MU_RR * mass * 9.81
            + 0.5 * pdc.RHO_AIR * pdc.A_SP * pdc.C_SP * v**2)


# =============================================================================
# LANDING ROLL: torque vs. force balance
# =============================================================================
t_land = np.linspace(0.0, T_STOP, 2000)
P_torque_land_W = np.array([m_power(x) for x in t_land])
E_torque_land_cum = cumulative_trapezoid(P_torque_land_W, t_land, initial=0.0) / 1e6

# Raw-sample intervals within the roll-out window, sub-stepped N_SUB_LAND
# times each -- same method as build_segment()'s _landing_force_balance().
land_idx = np.where((tm_raw >= 0.0) & (tm_raw <= T_STOP))[0]
land_idx = land_idx[land_idx < len(tm_raw) - 1]

# Reverser back-extrapolation (per Ugo, session 2026-08-15, see
# build_segment()): if ANY sample in the roll-out has the reverser flag
# active, assume it was active from touchdown through the last active
# sample, rather than trusting the exact per-sample boolean (30s-style
# under-sampling risk, kept for consistency even at 1Hz).
if len(land_idx) and rev_active_raw[land_idx].any():
    last_active = land_idx[rev_active_raw[land_idx]][-1]
    rev_active = np.zeros_like(rev_active_raw)
    rev_active[land_idx[0]:last_active + 1] = True
else:
    rev_active = rev_active_raw

t_sub, v_sub, P_sub = [], [], []
for i in land_idx:
    v1, v2 = v_ms_raw[i], v_ms_raw[i + 1]
    m1, m2 = mass_kg[i], mass_kg[i + 1]
    t1, t2 = tm_raw[i], tm_raw[i + 1]
    dt_raw = t2 - t1
    if dt_raw <= 0:
        continue
    a_const = (v2 - v1) / dt_raw
    rev_i = bool(rev_active[i] or rev_active[i + 1])
    edges = np.linspace(0.0, 1.0, pdc.N_SUB_LAND + 1)
    t_e = t1 + edges * dt_raw
    v_e = v1 + edges * (v2 - v1)
    m_e = m1 + edges * (m2 - m1)
    for j in range(pdc.N_SUB_LAND):
        v_mid = 0.5 * (v_e[j] + v_e[j + 1])
        m_mid = 0.5 * (m_e[j] + m_e[j + 1])
        F_rev = pdc.K_RT if (rev_i and v_mid > pdc.V_TH_REV) else 0.0
        F_b = max(m_mid * abs(a_const) - F_opposing(v_mid, m_mid) - F_rev, 0.0)
        t_sub.append(t_e[j])
        v_sub.append(v_mid)
        P_sub.append(F_b * v_mid / pdc.N_BRAKES)

t_sub = np.array(t_sub)
P_sub = np.array(P_sub)
order = np.argsort(t_sub)
t_sub, P_sub = t_sub[order], P_sub[order]
E_fb_land_cum = (cumulative_trapezoid(P_sub, t_sub, initial=0.0) / 1e6
                if len(t_sub) > 1 else np.zeros_like(t_sub))

E_torque_land_total = float(E_torque_land_cum[-1])
E_fb_land_total = float(E_fb_land_cum[-1]) if len(E_fb_land_cum) else 0.0

print(f"\n--- landing roll, [0, {T_STOP:.0f}] s -----------------------------")
print(f"  torque-based energy          : {E_torque_land_total:.2f} MJ/brake")
print(f"  force-balance energy         : {E_fb_land_total:.2f} MJ/brake")
if E_torque_land_total > 0:
    print(f"  ratio (force-balance/torque) : "
         f"{E_fb_land_total / E_torque_land_total:.2f}x")

# =============================================================================
# TAXI: torque vs. delta-V per deceleration phase
# (G(v) is still computed and printed below for reference, but no longer
# plotted -- delta-V replaces it as the method going forward, per Ugo.)
# =============================================================================
t_taxi = np.linspace(T_STOP, T_END, 2000)
P_torque_taxi_W = np.array([m_power(x) for x in t_taxi])
E_torque_taxi_cum = (cumulative_trapezoid(P_torque_taxi_W, t_taxi, initial=0.0)
                     / 1e6)

taxi_iv = np.where((tm_raw >= T_STOP) & (tm_raw <= T_END))[0]
taxi_iv = taxi_iv[taxi_iv < len(tm_raw) - 1]

v1, v2 = v_ms_raw[taxi_iv], v_ms_raw[taxi_iv + 1]
dt_i = tm_raw[taxi_iv + 1] - tm_raw[taxi_iv]
dist = 0.5 * (v1 + v2) * dt_i
mass_i = 0.5 * (mass_kg[taxi_iv] + mass_kg[taxi_iv + 1])
dKE = 0.5 * mass_i * (v2**2 - v1**2)
moving = (0.5 * (v1 + v2)) > 1.0 * KT2MS
accel = moving & (v2 > v1 + 0.3 * KT2MS)
G_med = (float(np.median(dKE[accel] / np.maximum(dist[accel], 1e-6)))
        if accel.sum() >= 3 else 0.0)
# E_iv is WHOLE-AIRCRAFT energy (G_med*dist and dKE both use the full
# aircraft mass, matching build_segment()) -- /N_BRAKES below to report
# it on the same per-brake basis as everything else in this script. This
# was missing before (bug fix, per Ugo): the earlier version of this
# script reported E_iv's whole-aircraft total as if it were per-brake.
E_iv = np.maximum(G_med * dist - dKE, 0.0)
E_iv[~moving] = 0.0

t_iv_mid = 0.5 * (tm_raw[taxi_iv] + tm_raw[taxi_iv + 1])
sort_i = np.argsort(t_iv_mid)
t_iv_sorted, dt_i_sorted, E_iv_sorted = (t_iv_mid[sort_i], dt_i[sort_i],
                                         E_iv[sort_i])
P_G_taxi_kW = (np.where(dt_i_sorted > 0, E_iv_sorted / dt_i_sorted, 0.0)
              / pdc.N_BRAKES / 1e3)
E_G_taxi_cum = np.cumsum(E_iv_sorted) / pdc.N_BRAKES / 1e6

E_torque_taxi_total = float(E_torque_taxi_cum[-1])
E_G_taxi_total = float(E_G_taxi_cum[-1]) if len(E_G_taxi_cum) else 0.0

# --- third method: kinetic energy per deceleration phase (delta-V) -------
# A "deceleration phase" is a maximal run of consecutive raw samples where
# ground speed drops by more than DECEL_THRESH_KT between samples (the
# deceleration counterpart of the ACCEL threshold above). Each phase runs
# from the last speed peak before it to wherever the drop stops -- either
# the aircraft starts accelerating again, or it reaches a full stop. The
# WHOLE kinetic-energy drop of the phase, 0.5*m*(v_start^2-v_end^2), is
# attributed to braking -- unlike G(v), nothing here separates braking
# from aerodynamic/rolling drag, so this method is expected to run high
# relative to the other two.
DECEL_THRESH_KT = 0.3
decel_thresh_ms = DECEL_THRESH_KT * KT2MS

taxi_pts = np.where((tm_raw >= T_STOP) & (tm_raw <= T_END))[0]
v_taxi_pts = v_ms_raw[taxi_pts]
t_taxi_pts = tm_raw[taxi_pts]
mass_taxi_pts = mass_kg[taxi_pts]

dv = np.diff(v_taxi_pts)
is_decel = dv < -decel_thresh_ms

phase_t0, phase_t1, phase_E, phase_dV_kt = [], [], [], []
i, n = 0, len(is_decel)
while i < n:
    if is_decel[i]:
        j = i
        while j < n and is_decel[j]:
            j += 1
        v_start, v_end = v_taxi_pts[i], v_taxi_pts[j]
        dV = v_start - v_end
        if dV > 0:
            mass_ph = 0.5 * (mass_taxi_pts[i] + mass_taxi_pts[j])
            E_ph = 0.5 * mass_ph * (v_start**2 - v_end**2) / pdc.N_BRAKES
            phase_t0.append(t_taxi_pts[i])
            phase_t1.append(t_taxi_pts[j])
            phase_E.append(E_ph)
            phase_dV_kt.append(dV / KT2MS)
        i = j
    else:
        i += 1

phase_t0 = np.array(phase_t0)
phase_t1 = np.array(phase_t1)
phase_E = np.array(phase_E)
phase_dV_kt = np.array(phase_dV_kt)
order = np.argsort(phase_t0)
phase_t0, phase_t1 = phase_t0[order], phase_t1[order]
phase_E, phase_dV_kt = phase_E[order], phase_dV_kt[order]
phase_dur = phase_t1 - phase_t0
phase_P_kW = np.where(phase_dur > 0, phase_E / phase_dur, 0.0) / 1e3

E_dv_taxi_cum = np.cumsum(phase_E) / 1e6
E_dv_taxi_total = float(E_dv_taxi_cum[-1]) if len(E_dv_taxi_cum) else 0.0

print(f"\n--- taxi, [{T_STOP:.0f}, {T_END:.0f}] s ---------------------------")
print(f"  G (self-calibrated)          : {G_med:.1f} N")
print(f"  torque-based energy           : {E_torque_taxi_total:.2f} MJ/brake")
print(f"  G-based energy                : {E_G_taxi_total:.2f} MJ/brake")
if E_torque_taxi_total > 0:
    print(f"    ratio (G-based/torque)      : "
         f"{E_G_taxi_total / E_torque_taxi_total:.2f}x")
print(f"  delta-V energy                : {E_dv_taxi_total:.2f} MJ/brake, "
     f"{len(phase_E)} deceleration phase(s)")
if E_torque_taxi_total > 0:
    print(f"    ratio (delta-V/torque)      : "
         f"{E_dv_taxi_total / E_torque_taxi_total:.2f}x")
print("  (the paper reports a 15-35x G-based overestimation on sustained "
     "low-power rolling -- compare against this)")
if len(phase_E):
    print("\n  deceleration phases (delta-V method):")
    print(f"    {'start [s]':>10}  {'end [s]':>10}  {'dV [kt]':>8}  "
         f"{'E [MJ/brake]':>13}")
    for k in range(len(phase_E)):
        print(f"    {phase_t0[k]:10.1f}  {phase_t1[k]:10.1f}  "
             f"{phase_dV_kt[k]:8.1f}  {phase_E[k]/1e6:13.3f}")

# =============================================================================
# FIGURES -- one per phase, power on top, cumulative energy below,
# time normalised t/t_phase within each phase (0 = start of that phase)
# =============================================================================
fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

t_land_n = t_land / T_STOP if T_STOP > 0 else t_land
t_sub_n = t_sub / T_STOP if T_STOP > 0 else t_sub

ax1.plot(t_land_n, P_torque_land_W / 1e3, 'C0-', lw=1.5,
        label='torque (ground truth)')
if len(t_sub):
    ax1.plot(t_sub_n, P_sub / 1e3, 'C3-', lw=1.2, alpha=0.85,
            label='force balance')
ax1.set_ylabel('Brake power [kW]')
ax1.set_title(f'{FLIGHT}, wheel {WHEEL_ID} -- landing roll: '
             f'torque vs. force balance')
ax1.legend(fontsize=9, framealpha=0.9)
ax1.grid(alpha=0.3)

ax2.plot(t_land_n, E_torque_land_cum, 'C0-', lw=1.6,
        label=f'torque: {E_torque_land_total:.2f} MJ')
if len(t_sub):
    ax2.plot(t_sub_n, E_fb_land_cum, 'C3-', lw=1.4,
            label=f'force balance: {E_fb_land_total:.2f} MJ')
ax2.set_xlabel('t / t_rollout (0 = touchdown, 1 = end of roll-out)')
ax2.set_ylabel('Cumulative energy per brake [MJ]')
ax2.legend(fontsize=9, framealpha=0.9)
ax2.grid(alpha=0.3)
fig1.tight_layout()

fig2, (bx1, bx2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

t_taxi_n = (t_taxi - T_STOP) / (T_END - T_STOP) if T_END > T_STOP else t_taxi
phase_t0_n = ((phase_t0 - T_STOP) / (T_END - T_STOP)
             if T_END > T_STOP else phase_t0)
phase_t1_n = ((phase_t1 - T_STOP) / (T_END - T_STOP)
             if T_END > T_STOP else phase_t1)

bx1.plot(t_taxi_n, P_torque_taxi_W / 1e3, 'C0-', lw=1.5,
        label='torque (ground truth)')
if len(phase_E):
    bx1.hlines(phase_P_kW, phase_t0_n, phase_t1_n, color='C2', lw=2.2,
              alpha=0.9, label='delta-V per phase')
bx1.set_ylabel('Brake power [kW]')
bx1.set_title(f'{FLIGHT}, wheel {WHEEL_ID} -- taxi: torque vs. delta-V')
bx1.legend(fontsize=9, framealpha=0.9)
bx1.grid(alpha=0.3)

bx2.plot(t_taxi_n, E_torque_taxi_cum, 'C0-', lw=1.6,
        label=f'torque: {E_torque_taxi_total:.2f} MJ')
if len(phase_E):
    bx2.step(phase_t1_n, E_dv_taxi_cum, where='post', color='C2', lw=1.4,
             label=f'delta-V: {E_dv_taxi_total:.2f} MJ')
bx2.set_xlabel('t / t_taxi (0 = start of taxi, 1 = end of taxi)')
bx2.set_ylabel('Cumulative energy per brake [MJ]')
bx2.legend(fontsize=9, framealpha=0.9)
bx2.grid(alpha=0.3)
fig2.tight_layout()

plt.show()