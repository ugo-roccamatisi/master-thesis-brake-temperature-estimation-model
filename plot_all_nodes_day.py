# =============================================================================
# SANITIZED PUBLIC VERSION
# All calibrated parameter values have been replaced with generic round
# placeholders, and all flight identifiers, timestamps and paths with
# examples. Calibrate on your own data before any quantitative use.
# =============================================================================
# -*- coding: utf-8 -*-
"""
Internal node temperatures across a commercial operating day.

btem_pipeline_with_eps_T.py keeps only the sensor node (Tc[m.I_SENS]) as it
walks the day, discarding the other 13 at every segment. That is all its
figures need, but it means the model's own internal state is never shown: what
the disc stack, the torque tube, the rim, the shield and the housing are doing
while the one measured node is being tracked.

This script re-runs the same day, with the same calibration and the same
handover between flights, but keeps the full 14-node state.

The whole day is still simulated, since each flight's initial condition is the
previous one's end state, but only the FIRST flight is plotted. It is drawn
four ways:

  all nodes,   absolute     how many degrees each component reaches
  all nodes,   normalised   how they rank and how they separate
  discs only,  absolute     the stack alone, without the carrier nodes
                            compressing the vertical scale
  discs only,  normalised   which end of the stack runs hottest

Absolute and normalised answer different questions: the first gives magnitude,
the second gives shape and ordering, and neither substitutes for the other.

Only the sensor node has measured data to compare against. Every other curve
is model-internal, unmeasured, and shown as a consequence of the calibration
rather than as a validated quantity.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import btem_v5_full_flight as m
import predict_day_from_csv as core_mod
import btem_pipeline_with_eps_T as pipe

# --- node display table -----------------------------------------------------
# The nine discs share one colour and are drawn thin: individually they are not
# the point, their spread is. The five carrier nodes are named and thick.
DISC_STYLE = dict(color='0.72', lw=0.8)
CARRIER = [(m.I_TUBE,   'Torque tube', 'k',        '-',  1.8),
           (m.I_WHEEL,  'Wheel/rim',   'tab:blue', '--', 1.5),
           (m.I_SHIELD, 'Heat shield', 'tab:pink', '-',  1.4),
           (m.I_PIST,   'Housing',     'tab:cyan', '-',  1.4),
           (m.I_SENS,   'BTMS sensor', 'tab:green','-',  2.2)]


def run_day_all_nodes(segments, calibration, raw_flights):
    """Same walk as btem_pipeline_with_eps_T's predict_one(gap_model='h_nat'),
    but returning the whole (14, N) state instead of just the sensor row.

    Kept deliberately close to that function: the m._t_record override placed
    AFTER apply_module_state(), the same handover temperature, the same
    _btms suppression on resumed segments. Anything that drifts from it here
    would make these curves inconsistent with the pipeline's own figures.
    """
    t_all, T_all, per_flight = [], [], []
    T_init = t_start = None

    for idx, seg in enumerate(segments):
        seg["apply_module_state"]()
        cc = calibration["core"]
        m.C_tube, m.G_ST, m.G_RW, m.eps_DT = (cc["C_tube"], cc["G_ST"],
                                              cc["G_RW"], cc["eps_DT"])
        m.G_sens = m.C_sens/cc["tau"]
        m.h_nat = cc["h_nat"]
        m.eps_T = cc["eps_T"]

        # override AFTER apply_module_state(), which resets _t_record itself
        if idx < len(segments)-1:
            _tnm = pipe._first_moving(raw_flights[idx+1])
            _tgt = (_tnm + segments[idx+1]["touchdown_time_s"]
                    - seg["touchdown_time_s"])
            _saved = m._t_record
            m._t_record = max(m._t_record or 0., _tgt)

        m.P_FLIGHT_SCALE = calibration["alpha"]
        m.C_TUBE_FLIGHT_MULT = 1.
        m.H_NAT_FLIGHT = calibration["h_nat_flight"]
        tnd, Pnd = seg["build_power_trace"](1., calibration["pulse_dur"], 1.)
        m._P_interp = interp1d(tnd, Pnd, bounds_error=False, fill_value=0.)

        if idx == 0:
            T_init = seg["T_INIT_FULL"]

        _sb = m._btms if t_start is not None else None
        if t_start is not None:
            m._btms = None
        try:
            t_loc, Tc, _ = m.run_full(T_init, t_start=t_start)
        finally:
            if _sb is not None:
                m._btms = _sb
        if idx < len(segments)-1:
            m._t_record = _saved

        t_abs = t_loc + seg["touchdown_time_s"]
        t_all.append(t_abs)
        T_all.append(Tc)                      # (14, N), not just I_SENS
        per_flight.append(dict(label=seg["label"], t=t_abs, Tc=Tc,
                              tm=seg["tm_b"]+seg["touchdown_time_s"],
                              Tm=seg["Tm_b"]))

        if idx < len(segments)-1:
            nxt = segments[idx+1]
            rec_own = float(raw_flights[idx+1]["tm"][0])
            rec_abs = rec_own + nxt["touchdown_time_s"]
            T_init = float(np.interp(rec_abs, t_abs, Tc[m.I_SENS]))
            t_start = rec_own

    order = np.argsort(np.concatenate(t_all))
    return (np.concatenate(t_all)[order],
            np.concatenate(T_all, axis=1)[:, order],
            per_flight)


def draw(t, Tc, tm, Tm, normalize, discs_only, title):
    """One window of the day.

    normalize   both axes to [0,1] -- time over the day, temperature over the
                hottest value reached by ANY node (one shared scale, never one
                per curve, which would flatten every node onto the same peak
                and erase exactly the difference being looked at).
    discs_only  the nine discs alone, which is where the heat is generated,
                without the carrier nodes compressing the vertical scale.
    """
    idxs = list(range(9)) if discs_only else None

    pool = [Tc[i] for i in (idxs if idxs else range(14))]
    T_MAX = max(float(np.max(v)) for v in pool)
    if not discs_only:
        T_MAX = max(T_MAX, float(np.max(Tm)))
    t0, t1 = float(t.min()), float(t.max())
    span = (t1 - t0) or 1.0

    if normalize:
        xf = lambda a: (np.asarray(a, float) - t0)/span
        yf = lambda a: np.asarray(a, float)/T_MAX
        xlab, ylab = 't / t_total (0 = start of the segment, 1 = end)', 'T / T_max'
    else:
        xf = lambda a: np.asarray(a, float)/60.0
        yf = lambda a: np.asarray(a, float)
        xlab, ylab = 'Time [min] (0 = touchdown)', 'Temperature [\u00b0C]'

    fig, ax = plt.subplots(figsize=(14, 7))

    if discs_only:
        # named individually here: with only nine curves they are separable,
        # and which end of the stack runs hottest is the point of this view
        for i in range(9):
            ax.plot(xf(t), yf(Tc[i]), lw=1.2, label=m.labels[i])
    else:
        for i in range(9):
            ax.plot(xf(t), yf(Tc[i]),
                    label='Discs (S1--S5, R1--R4)' if i == 0 else None,
                    **DISC_STYLE)
        for idx, lab, col, ls, lw in CARRIER:
            ax.plot(xf(t), yf(Tc[idx]), color=col, ls=ls, lw=lw, label=lab)
        ax.plot(xf(tm), yf(Tm), 'r.', ms=2.5, alpha=0.45, label='BTMS measured')

    if normalize:
        ax.axhline(1.0, color='0.5', ls='--', lw=1.0)
        ax.set_xlim(0, 1)

    ax.set(xlabel=xlab, ylabel=ylab, title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=3 if not discs_only else 5, framealpha=0.9)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    paths = [pipe.find_csv(f) for f in pipe.DAY_FLIGHTS]
    segments, _, _ = core_mod.segment_csv_inputs(paths, log=print,
                                                 wheel=pipe.WHEEL_ID)
    if len(segments) != len(pipe.DAY_FLIGHTS):
        raise SystemExit(f"segment_csv_inputs found {len(segments)} flights, "
                         f"expected {len(pipe.DAY_FLIGHTS)}")
    calibration = pipe._load_calibration_bundle_epsT(pipe.STAGE2_FILE)
    raw_flights = [pipe._load_raw(f) for f in pipe.DAY_FLIGHTS]

    t, Tc, per_flight = run_day_all_nodes(segments, calibration, raw_flights)

    # The day is still simulated end to end, because each flight's initial
    # condition is the previous one's end state -- only the plotting is
    # restricted to the first flight.
    f0 = per_flight[0]
    t0, Tc0 = f0['t'], f0['Tc']
    sel = (f0['tm'] >= t0.min()) & (f0['tm'] <= t0.max())
    tm0, Tm0 = f0['tm'][sel], f0['Tm'][sel]

    print()
    print(f"  {'node':<26}{'peak':>9}{'at end of flight':>18}")
    print(f"  {'Discs (hottest of 9)':<26}"
          f"{max(Tc0[k].max() for k in range(9)):>8.0f}C"
          f"{max(Tc0[k][-1] for k in range(9)):>17.0f}C")
    for idx, lab, *_ in CARRIER:
        print(f"  {lab:<26}{Tc0[idx].max():>8.0f}C{Tc0[idx][-1]:>17.0f}C")

    wh = f"wheel {pipe.WHEEL_ID}"
    for discs_only in (False, True):
        what = 'brake discs only' if discs_only else 'all 14 nodes'
        note = '' if discs_only else ' (only BTMS is measured)'
        for normalize in (False, True):
            scale = 'normalised' if normalize else 'absolute'
            draw(t0, Tc0, tm0, Tm0, normalize, discs_only,
                 f"{f0['label']} -- {what}, {wh}, {scale}{note}")

    plt.show()