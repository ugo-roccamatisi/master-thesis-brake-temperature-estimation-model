# =============================================================================
# SANITIZED PUBLIC VERSION
# All calibrated parameter values have been replaced with generic round
# placeholders, and all flight identifiers, timestamps and paths with
# examples. Calibrate on your own data before any quantitative use.
# =============================================================================
# -*- coding: utf-8 -*-
"""
Streamlit interface for predict_day_from_csv.py -- upload a day's CSV(s),
segment into flights, calibrate, predict, inspect the result. Every
computation reuses that script's own functions unchanged (segment_csv_
inputs, calibrate, run_prediction, build_figure, ...) so the web UI and
the CLI can never silently drift apart.

Run with:  streamlit run app.py
(predict_day_from_csv.py must sit next to this file, or on the PYTHONPATH.)
"""

import json
import streamlit as st
import predict_day_from_csv as core

st.set_page_config(page_title="BTEM day predictor", layout="wide")
st.title("Brake temperature prediction over a day of flights")

AIRCRAFT_CATEGORIES = [
    ("Aerodynamics", ['rho_air', 'a_wing', 'c_d', 'mu_rr']),
    ("Spoilers", ['n_sp', 'l_sp', 'b_sp', 'theta_max_deg', 'c_sp']),
    ("Reverse thrust", ['k_rt', 'v_th_rev_kt']),
    ("Touchdown detection", ['v_td_threshold_kt', 'v_roll_end_kt']),
    ("Segmentation & unit thresholds",
     ['n_brakes', 'n_sub_land', 'mass_lb_threshold', 'v_stop_kt',
      'lg_debounce_s', 'stop_debounce_s']),
    ("Uncertainty model (all UNVERIFIED placeholders)",
     ['climb_duration_s', 'u_bts_c', 'u_tol_c']),
]
THERMAL_CATEGORIES = [
    ("Geometry", ['r_wheel', 'R_ext', 'R_int', 'e_disc', 'R_rim', 'L_rim',
                  'A_ct', 'A_cw']),
    ("Disc material", ['rho_c', 'cp_c', 'kz_c']),
    ("Heat capacities (mass x specific heat)",
     ['mass_wheel', 'cp_wheel', 'mass_shield', 'cp_shield',
      'mass_pist', 'cp_pist', 'mass_sens', 'cp_sens']),
    ("Conductances", ['G_PH', 'G_SP', 'G_SW']),
    ("Emissivities", ['eps_DS', 'eps_SW', 'eps_end', 'eps_W', 'eps_PH',
                      'eps_S1P']),
    ("Convection", ['h_fan', 'k_air', 'nu_air', 'pr_air', 'l_conv']),
    ("Timing / thresholds", ['takeoff_accel_thresh', 't_retract_s',
                             'h_approach_ft', 'approach_fallback_s',
                             't_bay_c', 'v_taxi_kt']),
    ("'flight'-phase behaviour", ['p_brake_thresh', 'flight_forced_cooling',
                                  'fan_on', 'ground_fan_on']),
]

# Cycled per category header so each group of parameters is easy to tell
# apart at a glance. Plain HTML via unsafe_allow_html rather than
# Streamlit's :color[text] markdown syntax, which needs a recent Streamlit
# version and rendered as
# literal ":teal[...]" text on an older install.
_CATEGORY_COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728',
                    '#17a2b8', '#6c757d', '#e83e8c']


def _category_header(text, color):
    st.markdown(f"<h4 style='color:{color}; margin-bottom:0.2em'>{text}</h4>",
               unsafe_allow_html=True)

# Short help text per parameter, shown as a "?" tooltip next to each widget.
# Reuses the exact wording of the CLI --help strings (predict_day_from_csv.py's
# main()) so the two never drift apart on what a given knob does.
HELP_TEXT = {
    # -- Aircraft constants --
    'n_brakes': 'Braked wheels sharing the reconstructed energy.',
    'max_e_taxi_mj_brake': 'Hard cap on taxi-out energy per brake '
        '[MJ/brake], 0 = disabled.',
    'rho_air': 'Air density [kg/m3].',
    'a_wing': 'Wing reference area [m2], for aero drag.',
    'c_d': 'Aerodynamic drag coefficient.',
    'mu_rr': 'Rolling resistance coefficient.',
    'n_sp': 'Spoiler count.',
    'l_sp': 'Spoiler length [m].',
    'b_sp': 'Spoiler width [m].',
    'theta_max_deg': 'Max spoiler deflection [deg].',
    'c_sp': 'Spoiler drag coefficient.',
    'k_rt': 'Reverse-thrust force [N], 0 = disabled. Applies only while '
        'reverse thrust is active AND ground speed exceeds v_th_rev_kt.',
    'v_th_rev_kt': 'Ground speed below which reverse thrust cuts off [kt].',
    'disable_retraction_window': 'Skip the gear-retraction cooling window '
        '(unchecked = disabled, this project\u2019s own default).',
    'n_sub_land': 'Landing-roll sub-steps per raw interval.',
    'v_td_threshold_kt': 'rule_100kt threshold [kt]: touchdown = last '
        'sample above this speed.',
    'v_roll_end_kt': 'Ground speed [kt] below which the roll-out is '
        '"done".',
    'v_td_plausible': 'Plausible touchdown speed range [kt] (sanity '
        'check only, does not affect detection).',
    'mass_lb_threshold': 'GROSS WEIGHT above this is assumed already in '
        'lb, not kg.',
    'v_stop_kt': 'Ground speed [kt] marking a full stop, finer than '
        'v_roll_end_kt.',
    'lg_debounce_s': 'Landing-gear signal gaps shorter than this [s] are '
        'bridged rather than treated as a real state change.',
    'stop_debounce_s': 'How long [s] the aircraft must stay under '
        'v_stop_kt to count as genuinely stopped.',
    'climb_duration_s': 'UNVERIFIED assumption: how long after the start '
        'of the take-off roll the "climb" phase is assumed to last (no '
        'altitude channel, so no direct way to detect the real '
        'climb-to-cruise transition).',
    'u_bts_c': 'UNVERIFIED placeholder: BTEM sensor measurement '
        'uncertainty [\u00b0C]. Generic aerospace-thermocouple ballpark, '
        'not a sourced figure.',
    'u_tol_c': 'UNVERIFIED placeholder: the governing tolerance limit '
        'U_tol -- an engineering/policy target, not a measured quantity.',
    # -- Thermal module constants --
    'r_wheel': 'Wheel rolling radius [m].',
    'R_ext': 'Disc external radius [m].',
    'R_int': 'Disc internal radius [m].',
    'e_disc': 'Disc thickness [m].',
    'R_rim': 'Wheel rim radius [m].',
    'L_rim': 'Wheel rim length [m].',
    'A_ct': 'Tube contact area [m2].',
    'A_cw': 'Housing contact area [m2].',
    'rho_c': 'Disc material density [kg/m3].',
    'cp_c': 'Disc material specific heat [J/kg/K].',
    'kz_c': 'Disc material through-thickness conductivity [W/m/K].',
    'mass_wheel': 'Wheel/rim mass [kg].',
    'cp_wheel': 'Wheel/rim specific heat [J/kg/K].',
    'mass_shield': 'Heat shield mass [kg].',
    'cp_shield': 'Heat shield specific heat [J/kg/K].',
    'mass_pist': 'Piston housing mass [kg].',
    'cp_pist': 'Piston housing specific heat [J/kg/K].',
    'mass_sens': 'BTEM sensor mass [kg].',
    'cp_sens': 'BTEM sensor specific heat [J/kg/K].',
    'G_PH': 'Tube -> housing conductance [W/K].',
    'G_SP': 'S1 -> housing conductance [W/K].',
    'G_SW': 'Shield -> wheel conductance [W/K].',
    'eps_DS': 'Disc -> shield emissivity.',
    'eps_SW': 'Shield -> wheel emissivity.',
    'eps_end': 'Disc end face -> ambient emissivity.',
    'eps_W': 'Wheel -> ambient emissivity.',
    'eps_PH': 'Housing -> ambient emissivity.',
    'eps_S1P': 'S1 -> housing (taxi gap) emissivity.',
    'h_fan': 'Taxi brake-cooling fan coefficient [W/m2K].',
    'k_air': 'Air thermal conductivity [W/m/K].',
    'nu_air': 'Air kinematic viscosity [m2/s].',
    'pr_air': 'Air Prandtl number.',
    'l_conv': 'Convection length scale [m].',
    'takeoff_accel_thresh': 'Ground acceleration [m/s2] above which the '
        'aircraft is considered taking off.',
    't_retract_s': 'Gear-retraction cooling window [s].',
    'h_approach_ft': 'Altitude where approach cooling starts [ft].',
    'approach_fallback_s': 'Fallback approach-cooling duration if no '
        'altitude column is found [s].',
    't_bay_c': 'Gear-bay temperature, gear stowed [\u00b0C].',
    'v_taxi_kt': 'Representative taxi ground speed [kt].',
    'p_brake_thresh': 'Power [W] above which \'flight\'-phase ground '
        'contact counts as real braking (requires the 2026-08-10 module '
        'patch -- silently has no effect on an older copy).',
    'flight_forced_cooling': 'Model the gear-retraction/approach forced-'
        'convection windows during \'flight\' (unchecked = skipped).',
    'fan_on': 'Model taxi brake-cooling fans as on (default: off, this '
        'project\u2019s own convention).',
    'ground_fan_on': 'Model parking ground fans as on (default: off, '
        'this project\u2019s own convention).',
}


def constants_widget(key_prefix, defaults, categories, help_text=None,
                     exclude=()):
    """Renders a categorised set of number_input/checkbox widgets for a
    defaults dict, with a JSON-upload shortcut, a reset-to-defaults
    button, and a live count of values that differ from default. Returns
    the resulting values dict.

    exclude: keys handled manually by the caller OUTSIDE this widget
    (e.g. a tuple-valued default like v_td_plausible, which the generic
    number_input/checkbox rendering below can't handle) -- skipped
    entirely here, including from the 'leftover' safety net, so they
    don't crash or get a second, conflicting widget."""
    uploaded_json = st.file_uploader(
        "Optionally load any subset of these from a JSON file instead "
        "of typing them in", type="json", key=f"{key_prefix}_json")
    json_overrides = json.load(uploaded_json) if uploaded_json else {}
    if help_text:
        st.caption(help_text)

    if st.button("Reset to A320 defaults", key=f"{key_prefix}_reset"):
        for cat_name, keys in categories:
            for key in keys:
                st.session_state.pop(f"{key_prefix}_{key}", None)
        st.rerun()

    seen = set()
    values = {}
    n_modified = 0
    for cat_idx, (cat_name, keys) in enumerate(categories):
        color = _CATEGORY_COLORS[cat_idx % len(_CATEGORY_COLORS)]
        _category_header(cat_name, color)
        if cat_name == "'flight'-phase behaviour":
            st.caption("`p_brake_thresh` only takes effect on a "
                      "btem_v5_full_flight.py patched on 2026-08-10 "
                      "or later (promoted out of a local inside rhs()) -- "
                      "silently has no effect on an older copy.")
        cols = st.columns(4)
        for i, key in enumerate(keys):
            seen.add(key)
            default = defaults[key]
            current_default = json_overrides.get(key, default)
            label = key.replace('_', ' ')
            widget_key = f"{key_prefix}_{key}"
            col = cols[i % 4]
            tip = HELP_TEXT.get(key)
            if isinstance(default, bool):
                val = col.checkbox(label, value=current_default,
                                   key=widget_key, help=tip)
            else:
                val = col.number_input(label, value=current_default,
                                       format="%.5g", key=widget_key,
                                       help=tip)
            values[key] = val
            if val != default:
                n_modified += 1
    # safety net: any key in defaults not yet sorted into a category
    # (and not explicitly excluded -- see the exclude= docstring above)
    leftover = [k for k in defaults if k not in seen and k not in exclude]
    if leftover:
        _category_header("Other", _CATEGORY_COLORS[len(categories) % len(_CATEGORY_COLORS)])
        cols = st.columns(4)
        for i, key in enumerate(leftover):
            default = defaults[key]
            current_default = json_overrides.get(key, default)
            col = cols[i % 4]
            tip = HELP_TEXT.get(key)
            if isinstance(default, bool):
                val = col.checkbox(key, value=current_default,
                                   key=f"{key_prefix}_{key}", help=tip)
            else:
                val = col.number_input(key, value=current_default, format="%.5g",
                                       key=f"{key_prefix}_{key}", help=tip)
            values[key] = val
            if val != default:
                n_modified += 1

    if n_modified:
        st.info(f"{n_modified} value(s) differ from the A320 default.")
    else:
        st.caption("All values at their A320 default.")
    return values


# =============================================================================
# 1. LOAD + SEGMENT
# =============================================================================
with st.expander("Aircraft constants (advanced -- defaults are this "
                 "project's own A320)", expanded=False):
    aircraft_vals = constants_widget(
        "aircraft", core.AIRCRAFT_DEFAULTS, AIRCRAFT_CATEGORIES,
        help_text="v_td_threshold_kt is used to DETECT touchdown itself "
                  "(rule_100kt); v_td_plausible only checks the result "
                  "afterward and never affects detection.",
        exclude=('v_td_plausible', 'disable_retraction_window'))
    v_td_lo = st.number_input("Plausible touchdown speed, min [kt]",
                              value=core.AIRCRAFT_DEFAULTS['v_td_plausible'][0],
                              key="aircraft_v_td_lo",
                              help=HELP_TEXT['v_td_plausible'])
    v_td_hi = st.number_input("Plausible touchdown speed, max [kt]",
                              value=core.AIRCRAFT_DEFAULTS['v_td_plausible'][1],
                              key="aircraft_v_td_hi",
                              help=HELP_TEXT['v_td_plausible'])
    enable_retraction_window = st.checkbox(
        "Model gear-retraction cooling window",
        value=not core.AIRCRAFT_DEFAULTS['disable_retraction_window'],
        key="aircraft_enable_retraction_window",
        help=HELP_TEXT['disable_retraction_window'])

with st.expander("Thermal module constants (advanced -- brake/wheel "
                 "hardware geometry, defaults are this project's own "
                 "A320)", expanded=False):
    st.caption("Fixed geometry, material, conductances, emissivities, "
              "convection, and timing from btem_v5_full_flight.py "
              "itself -- distinct from the aircraft constants above, "
              "which are about the airframe's aerodynamics, not its "
              "brakes. The 7 FITTED core parameters (C_tube, G_ST, tau, "
              "eps_DT, G_RW, h_nat, eps_T) are calibrated by this "
              "pipeline and aren't here.")
    thermal_vals = constants_widget(
        "thermal", core.THERMAL_MODULE_DEFAULTS, THERMAL_CATEGORIES)

st.header("1. Load and segment")

uploaded = st.file_uploader(
    "One whole-day CSV, or several per-flight CSVs (they'll be laid end "
    "to end by their own timestamps)",
    type="csv", accept_multiple_files=True)

col1, col2, col3 = st.columns(3)
min_park_gap_min = col1.number_input(
    "Minimum parked duration to count as a gap between flights [min]",
    min_value=1.0, value=20.0, step=1.0,
    help="A parking gap is any stretch where ground speed stays below "
         "the threshold on the right for at least this long -- each "
         "stretch between two such gaps is treated as one flight.")
park_kt = col2.number_input(
    "Ground speed below which the aircraft is considered parked [kt]",
    min_value=0.0, value=0.0, step=0.5,
    help="The aircraft doesn't move while parked, so 0 is the natural "
         "choice -- raise it only if your ground speed signal has noise "
         "around 0 while genuinely parked.")
wheel = col3.selectbox(
    "Wheel to simulate", [1, 2, 3, 4], index=0,
    help="Which brake's BTEM column drives every segment's measured "
         "data, initial condition, and predicted curve. Changing this "
         "requires reloading/re-segmenting -- the whole pipeline is "
         "wheel-specific from this point on.")

if st.button("Load and segment", type="primary", disabled=not uploaded):
    core.configure_constants(dict(
        aircraft_vals, v_td_plausible=(v_td_lo, v_td_hi),
        disable_retraction_window=not enable_retraction_window))
    core.configure_thermal_module(thermal_vals)
    log_lines = []
    try:
        with st.spinner("Reading CSV(s) and detecting flights..."):
            segments, t_full, gspd_kt = core.segment_csv_inputs(
                uploaded, min_park_gap_min, park_kt, log=log_lines.append,
                wheel=wheel)
    except core.PipelineError as e:
        st.error(str(e))
        st.text("\n".join(log_lines))
        st.stop()
    st.session_state['segments'] = segments
    st.session_state['t_full'] = t_full
    st.session_state['gspd_kt'] = gspd_kt
    st.session_state['segment_log'] = log_lines
    st.session_state['data_fingerprint'] = core.data_fingerprint(uploaded)
    # any previous prediction is now stale
    st.session_state.pop('prediction', None)

if 'segment_log' in st.session_state:
    with st.expander("Segmentation log", expanded=False):
        st.text("\n".join(st.session_state['segment_log']))

if 'segments' in st.session_state:
    segments = st.session_state['segments']
    st.success(f"{len(segments)} flight(s) detected")
    st.dataframe(
        [{"flight": s['label'],
          "touchdown [h]": round(s['touchdown_time_s']/3600, 2),
          "touchdown speed [kt]": round(s['v_td_kt'], 1),
          "warning": s['warn'] or ""} for s in segments],
        width='stretch', hide_index=True)
    with st.expander("Segmentation check (ground speed + detected "
                     "windows)", expanded=False):
        st.caption("Check this BEFORE calibrating -- a flight split in "
                  "two, or two flights merged into one, will show up "
                  "here as a shading/touchdown mismatch.")
        fig_seg = core.build_segmentation_figure(
            st.session_state['t_full'], st.session_state['gspd_kt'], segments)
        st.pyplot(fig_seg)

# =============================================================================
# 2. CALIBRATE + PREDICT
# =============================================================================
def flight_selector(segments, key_prefix):
    """Renders the All/Single/Range picker and returns the selected
    0-based indices, in chronological order."""
    flight_labels = [s['label'] for s in segments]
    mode = st.radio("Flights to predict",
                    ['All flights', 'Single flight', 'Range of flights'],
                    horizontal=True, key=f"{key_prefix}_selmode")
    if mode == 'All flights':
        sel_idx = list(range(len(segments)))
    elif mode == 'Single flight':
        chosen = st.selectbox("Flight", flight_labels, key=f"{key_prefix}_single")
        sel_idx = [flight_labels.index(chosen)]
    else:
        start_lab, end_lab = st.select_slider(
            "Range (chronological order)", options=flight_labels,
            value=(flight_labels[0], flight_labels[-1]), key=f"{key_prefix}_range")
        i0, i1 = flight_labels.index(start_lab), flight_labels.index(end_lab)
        if i0 > i1:
            i0, i1 = i1, i0
        sel_idx = list(range(i0, i1+1))
    st.caption("Selected: " + ", ".join(flight_labels[i] for i in sel_idx)
              + (". Gap continuity is preserved even for a range that "
                 "doesn't start at flight 1 -- the physics still runs "
                 "through the full chain from the beginning, only the "
                 "report/plot are trimmed to the selection."
                 if sel_idx and sel_idx[0] > 0 else ""))
    return sel_idx


# Unit and one-line meaning for every calibrated parameter, so the display
# below reads as physics rather than as a bare JSON dump.
CORE_INFO = {
    'C_tube': ("J/K",     "Torque-tube heat capacity"),
    'G_ST':   ("W/K",     "Stator -> torque tube conductance"),
    'tau':    ("s",       "BTMS sensor time constant (C_sens/G_sens)"),
    'eps_DT': ("-",       "Disc -> torque tube emissivity"),
    'G_RW':   ("W/K",     "Rotor -> wheel conductance"),
    'h_nat':  ("W/m2K",   "Natural convection, touchdown/taxi/parking"),
    'eps_T':  ("-",       "Torque-tube -> ambient emissivity"),
}

PRE_TD_INFO = {
    'alpha':        ("-",     "Share of reconstructed energy reaching the pack"),
    'h_nat_flight': ("W/m2K", "Natural convection during the 'flight' phase"),
    'pulse_dur':    ("s",     "Taxi heat-input pulse duration"),
}


def calibration_display(calib, key_prefix, expanded=False):
    """Shows a calibration bundle -- whether just fitted or just imported --
    as a readable table rather than raw JSON: value, unit, meaning, and (for
    the fitted core) the search bounds it came from, so a parameter sitting on
    a bound is visible at a glance. A parameter pinned to its bound usually
    means it is compensating for something else in the model rather than
    converging on a physical value, which is worth seeing before trusting the
    numbers downstream."""
    c1, c2, c3 = st.columns(3)
    c1.metric("alpha", f"{calib['alpha']:.3f}")
    c2.metric("H_nat_flight [W/m2K]", f"{calib['h_nat_flight']:.2f}")
    c3.metric("Taxi mode", str(calib.get('taxi_mode', '-'))
              + (f" ({calib['pulse_dur']:.0f}s)" if calib.get('pulse_dur') else ""))

    at_bound = []
    rows = []
    for i, k in enumerate(core.NAMES):
        v = float(calib['core'][k])
        unit, meaning = CORE_INFO.get(k, ("", ""))
        lo, hi = float(core.LO[i]), float(core.HI[i])
        span = hi - lo
        on_edge = span > 0 and (abs(v-lo) <= 0.01*span or abs(v-hi) <= 0.01*span)
        if on_edge:
            at_bound.append(k)
        rows.append({
            "parameter": k,
            "value": f"{v:.4g}",
            "unit": unit,
            "bounds": f"{lo:.4g} .. {hi:.4g}",
            "at bound": "yes" if on_edge else "",
            "meaning": meaning,
        })

    with st.expander(f"Calibrated parameters ({len(core.NAMES)} core "
                     f"+ pre-touchdown)", expanded=expanded):
        st.caption("Fitted core parameters, with the bounds the search ran in.")
        st.dataframe(rows, width='stretch', hide_index=True)
        if at_bound:
            # tau on its LOWER bound is a known, expected result: the RMSE
            # plateaus below ~8 s, so the optimiser slides to the edge with
            # nothing to gain. Flagging it like a real problem would just
            # train the reader to ignore this warning.
            expected = [k for k in at_bound
                        if k == 'tau' and float(calib['core'][k])
                        <= float(core.LO[core.NAMES.index('tau')])*1.01]
            suspect = [k for k in at_bound if k not in expected]
            if suspect:
                st.warning(
                    "At its bound: " + ", ".join(suspect) + ". A parameter "
                    "pinned to a bound is usually compensating for something "
                    "else in the model rather than converging on a physical "
                    "value -- worth checking before relying on this "
                    "calibration.")
            if expected:
                st.info(
                    "tau sits on its lower bound. That is expected, not a "
                    "problem: the sensor time constant has a near-flat RMSE "
                    "plateau below ~8 s, so the fit slides to the edge with "
                    "nothing left to gain.")

        st.caption("Pre-touchdown parameters (fitted on the pre-takeoff "
                   "braking segment, applied to the 'flight' phase only).")
        pre_rows = []
        for k, (unit, meaning) in PRE_TD_INFO.items():
            v = calib.get(k)
            if v is None:
                continue
            pre_rows.append({"parameter": k, "value": f"{float(v):.4g}",
                             "unit": unit, "meaning": meaning})
        pre_rows.append({"parameter": "taxi_mode",
                         "value": str(calib.get('taxi_mode', '-')),
                         "unit": "", "meaning": "Shape of the taxi heat input"})
        st.dataframe(pre_rows, width='stretch', hide_index=True)

        if calib.get('source'):
            st.caption(f"Source: {calib['source']}")
        st.download_button(
            "Download this calibration (JSON)",
            data=json.dumps(calib, indent=2),
            file_name="calibration.json", mime="application/json",
            key=f"{key_prefix}_dl_calib")


def calibration_form(segments, key_prefix, label):
    """Renders the fit-first/default picker (+ either a JSON upload or the
    manual-entry form for 'default') and returns (mode, bundle_or_None)."""
    mode = st.radio(
        f"Calibration{f' -- {label}' if label else ''}",
        ['fit-first', 'default'], horizontal=True,
        key=f"{key_prefix}_calmode",
        format_func=lambda v: (
            f"Fit on {segments[0]['label']} (first flight in this CSV)"
            if v == 'fit-first' else
            "Use an already-calibrated set of parameters"))
    bundle = None
    if mode == 'default':
        source_mode = st.radio(
            "Source", ['From JSON file', 'Type in manually'],
            horizontal=True, key=f"{key_prefix}_source")
        if source_mode == 'From JSON file':
            st.caption("Any calibration JSON this pipeline's own scripts "
                      "produce (core + alpha + h_nat_flight + taxi_mode). "
                      "If you want to use an average of several "
                      "calibrations, average them yourself first -- this "
                      "doesn't average anything.")
            up = st.file_uploader("Calibration JSON", type="json",
                                  key=f"{key_prefix}_caljson")
            if up is not None:
                try:
                    bundle = core.load_calibration_bundle(up)
                    st.success(f"Loaded: alpha={bundle['alpha']:.2f}, "
                              f"H_nat_flight={bundle['h_nat_flight']:.2f} "
                              f"W/m2K, taxi_mode={bundle['taxi_mode']}")
                    calibration_display(bundle, f"{key_prefix}_imported",
                                        expanded=True)
                except core.PipelineError as e:
                    st.error(str(e))
        else:
            st.caption("Type in a calibrated set of values from a previous "
                      "run of this pipeline. If you want to use an average "
                      "of several calibrations, average them yourself and "
                      "enter the result here -- this form doesn't average "
                      "anything.")
            cols = st.columns(4)
            core_vals = {}
            for i, k in enumerate(core.NAMES):
                core_vals[k] = cols[i % 4].number_input(
                    k, value=core.BASE[k], format="%.4g", key=f"{key_prefix}_{k}")
            c1, c2, c3 = st.columns(3)
            alpha = c1.number_input("alpha", value=1.0, format="%.3f",
                                    key=f"{key_prefix}_alpha")
            h_nat_flight = c2.number_input("H_nat_flight [W/m2K]", value=8.0,
                                           format="%.2f", key=f"{key_prefix}_hnf")
            taxi_mode = c3.selectbox("taxi mode", ['honest', 'pulse'],
                                     key=f"{key_prefix}_mode")
            pulse_dur = None
            if taxi_mode == 'pulse':
                pulse_dur = st.number_input("pulse duration [s]", value=30.0,
                                            min_value=1.0, key=f"{key_prefix}_pdur")
            bundle = dict(core=core_vals, alpha=alpha, h_nat_flight=h_nat_flight,
                          taxi_mode=taxi_mode, pulse_dur=pulse_dur,
                          source="manual input")
    return mode, bundle


if 'segments' in st.session_state:
    segments = st.session_state['segments']
    st.header("2. Calibrate and predict")

    sel_idx = flight_selector(segments, "single")
    cal_mode, bundle = calibration_form(segments, "single", label=None)
    use_cache = True
    if cal_mode == 'fit-first':
        use_cache = st.checkbox("Use the calibration cache if available",
                                value=True, key="single_use_cache")
    gap_model_label = st.radio(
        "Gap model", ["h_nat (continuous physics)",
                     "exp_gap1 (exponential, fit once on the first gap)"],
        key="single_gap_model",
        help="How inter-flight gaps are filled. 'h_nat' runs "
             "continuous physics through the gap (this pipeline's "
             "original approach). 'exp_gap1' fits a Newton's-law-of-"
             "cooling exponential (k, T_amb) once on the day's first "
             "gap and reuses those two numbers, unchanged, for every "
             "later gap.")
    gap_model = ('h_nat' if gap_model_label.startswith('h_nat')
                else 'exp_gap1')

    if st.button("Predict", type="primary", disabled=not sel_idx):
        log_lines = []
        progress = st.empty()

        def log_and_show(msg):
            log_lines.append(msg)
            progress.text(msg)

        try:
            fp = st.session_state['data_fingerprint'] if cal_mode == 'fit-first' else None
            with st.spinner("Calibrating... (fit-first can take a few "
                           "minutes -- multi-start over several taxi "
                           "shapes and starting points, each needing "
                           "its own ODE solves; watch this line for "
                           "live progress)" if cal_mode == 'fit-first'
                           else "Calibrating..."):
                calib = core.calibrate(segments, cal_mode, bundle_a=bundle,
                                       data_fingerprint=fp, use_cache=use_cache,
                                       log=log_and_show)
            with st.spinner("Predicting... (also computing the combined-"
                           "uncertainty band, one extra simulation)"):
                t_all, T_all, predicted_segments, flight_report, gap_report, env = \
                    core.run_prediction(segments, sel_idx, calib,
                                        include_envelope=True,
                                        gap_model=gap_model,
                                        log=log_and_show)
        except core.PipelineError as e:
            st.error(str(e))
            st.text("\n".join(log_lines))
            st.stop()
        progress.empty()
        st.session_state['prediction'] = dict(
            calib=calib, t_all=t_all, T_all=T_all,
            predicted_segments=predicted_segments,
            flight_report=flight_report, gap_report=gap_report,
            env=env, log=log_lines)

# =============================================================================
# 3. RESULTS
# =============================================================================
if 'prediction' in st.session_state:
    res = st.session_state['prediction']
    st.header("3. Results")

    calib = res['calib']
    calibration_display(calib, "results", expanded=False)

    rows = [{"segment": r['flight'],
            "type": "prediction" if r['prediction'] else "I.C. flight",
            "n points": r['n'], "RMSE [C]": round(r['rmse'], 1),
            "bias [C]": round(r['bias'], 1)} for r in res['flight_report']]
    rows += [{"segment": g['gap'], "type": "gap", "n points": g['n'],
             "RMSE [C]": round(g['rmse'], 1), "bias [C]": round(g['bias'], 1)}
            for g in res['gap_report']]
    st.dataframe(rows, width='stretch', hide_index=True)

    # every figure/exporter below takes the same (t_all, T_all, segments) triple
    curve = (res['t_all'], res['T_all'], res['predicted_segments'])

    for norm in (True, False):
        st.pyplot(core.build_figure(*curve, env=res.get('env'), normalize=norm))

    if res.get('env') is not None:
        with st.expander("Uncertainty + relative error (diagnostic)",
                         expanded=False):
            st.caption(
                "Combined uncertainty band, tolerance limit, and the "
                "combined uncertainty u_c(t) itself against the "
                "tolerance limit. The tolerance line (and the relative-"
                "error points) turn purple wherever the combined "
                "uncertainty exceeds the tolerance limit (drift).")
            for norm in (False, True):
                st.pyplot(core.build_uncertainty_error_figure(
                    *curve, res['env'], normalize=norm))

    result_df = core.build_result_dataframe(*curve)
    st.download_button(
        "Download predicted + measured series (CSV)",
        data=result_df.to_csv(index=False),
        file_name="prediction.csv", mime="text/csv")

    with st.expander("Full log", expanded=False):
        st.text("\n".join(res['log']))