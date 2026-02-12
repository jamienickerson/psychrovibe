import streamlit as st
import plotly.graph_objects as go
import psychrolib
import numpy as np
import pandas as pd
import math
import base64

# --- 1. CONFIGURATION & ENGINE ---
st.set_page_config(page_title="PsychroVibe: Critical Cooling", layout="wide", page_icon="psychrovibe_logo.png")

# Unit system: IP (°F, Btu/lb, lb/lb, etc.) vs SI (°C, J/kg, kg/kg, etc.)
# Radio "unit_radio" is source of truth: °F (IP) = Imperial, °C (SI) = SI
if "unit_radio" not in st.session_state:
    st.session_state["unit_radio"] = "°F (IP)"
_unit_radio = st.session_state["unit_radio"]
USE_SI = (_unit_radio == "°C (SI)")  # True when user chose °C (SI)
psychrolib.SetUnitSystem(psychrolib.SI if USE_SI else psychrolib.IP)

# Unit-aware label constants (used for result dict keys and display)
if USE_SI:
    KEY_DB, KEY_WB, KEY_DP = "Dry Bulb (°C)", "Wet Bulb (°C)", "Dew Point (°C)"
    KEY_RH = "Relative Humidity (%)"
    KEY_HR, KEY_ENTH, KEY_VOL, KEY_DENS = "Humidity Ratio (kg/kg)", "Enthalpy (J/kg)", "Specific Vol (m³/kg)", "Moist Air Density (kg/m³)"
    TEMP_LABEL, ENTHALPY_LABEL = "°C", "J/kg"
else:
    KEY_DB, KEY_WB, KEY_DP = "Dry Bulb (°F)", "Wet Bulb (°F)", "Dew Point (°F)"
    KEY_RH = "Relative Humidity (%)"
    KEY_HR, KEY_ENTH, KEY_VOL, KEY_DENS = "Humidity Ratio (lb/lb)", "Enthalpy (Btu/lb)", "Specific Vol (ft³/lb)", "Moist Air Density (lb/ft³)"
    TEMP_LABEL, ENTHALPY_LABEL = "°F", "Btu/lb"

# Chart limits stored in °F; converted to °C in chart when USE_SI
DEFAULT_CHART_MIN_F, DEFAULT_CHART_MAX_F = 20.0, 115.0

# Styling constants
PRIMARY_LINE_COLOR = "black"
SECONDARY_LINE_COLOR = "lightsteelblue"

# Read persisted controls (chart limits always stored in °F)
chart_t_min = st.session_state.get("chart_t_min", DEFAULT_CHART_MIN_F)
chart_t_max = st.session_state.get("chart_t_max", DEFAULT_CHART_MAX_F)
altitude_ft = st.session_state.get("altitude_ft", 0.0)

# Clamp chart limits (°F) to reasonable range
chart_t_min = max(-20.0, min(chart_t_min, 150.0))
chart_t_max = max(chart_t_min + 5.0, min(chart_t_max, 150.0))

# Pressure: PsychroLib expects altitude in m (SI) or ft (IP)
altitude_for_pressure = (altitude_ft * 0.3048) if USE_SI else altitude_ft
PRESSURE = psychrolib.GetStandardAtmPressure(altitude_for_pressure)

# --- 2. CORE FUNCTIONS ---
def get_saturation_curve(t_min, t_max, pressure, step=1):
    """Generates the 100% RH saturation curve for the chart background."""
    temps = np.arange(t_min, t_max + step, step)
    hum_ratios = []

    for t in temps:
        try:
            hr = psychrolib.GetSatHumRatio(t, pressure)
            hum_ratios.append(hr)
        except Exception:
            hum_ratios.append(None)

    return temps, hum_ratios


def get_rh_curve(rh_percent, t_min, t_max, pressure, step=1):
    """Generates a constant RH curve for a given relative humidity percentage."""
    temps = np.arange(t_min, t_max + step, step)
    hum_ratios = []

    for t in temps:
        try:
            hr = psychrolib.GetHumRatioFromRelHum(t, rh_percent / 100.0, pressure)
            hum_ratios.append(hr)
        except Exception:
            hum_ratios.append(None)

    return temps, hum_ratios


def _solve_db_by_bisection(func, target, t_min, t_max, tol=0.01, max_iter=60):
    """Generic bisection solver for dry-bulb temperature."""
    low, high = float(t_min), float(t_max)
    try:
        f_low = func(low) - target
        f_high = func(high) - target
    except Exception:
        raise ValueError("Unable to evaluate endpoints for dry-bulb solver.")

    if math.isnan(f_low) or math.isnan(f_high) or f_low * f_high > 0:
        raise ValueError("Dry-bulb root not bracketed in given range.")

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        try:
            f_mid = func(mid) - target
        except Exception:
            raise ValueError("Dry-bulb solver failed during iteration.")

        if abs(f_mid) < tol or (high - low) < tol:
            return mid

        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    raise ValueError("Dry-bulb solver did not converge.")


def solve_state_from_two(db, rh, wb, dp, pressure, t_min, t_max):
    """
    Solve the complete psychrometric state from any two of:
    Dry Bulb (db, °F), Relative Humidity (rh, %), Wet Bulb (wb, °F), Dew Point (dp, °F).
    """
    known = {
        "db": db,
        "rh": rh,
        "wb": wb,
        "dp": dp,
    }
    present = [k for k, v in known.items() if v is not None]
    if len(present) != 2:
        raise ValueError("Exactly two state properties must be provided.")

    # Initialize variables
    TDryBulb = db
    RelHum = rh / 100.0 if rh is not None else None
    TWetBulb = wb
    TDewPoint = dp

    pair = tuple(sorted(present))

    # Pairs including dry bulb are direct
    if pair == ("db", "rh"):
        TDryBulb = db
        RelHum = rh / 100.0
        TDewPoint = psychrolib.GetTDewPointFromRelHum(TDryBulb, RelHum)
        TWetBulb = psychrolib.GetTWetBulbFromRelHum(TDryBulb, RelHum, pressure)
    elif pair == ("db", "wb"):
        TDryBulb = db
        TWetBulb = wb
        TDewPoint = psychrolib.GetTDewPointFromTWetBulb(TDryBulb, TWetBulb, pressure)
        RelHum = psychrolib.GetRelHumFromTWetBulb(TDryBulb, TWetBulb, pressure)
    elif pair == ("db", "dp"):
        TDryBulb = db
        TDewPoint = dp
        RelHum = psychrolib.GetRelHumFromTDewPoint(TDryBulb, TDewPoint)
        TWetBulb = psychrolib.GetTWetBulbFromTDewPoint(TDryBulb, TDewPoint, pressure)

    # Pairs without dry bulb require solving for TDryBulb
    elif pair == ("rh", "wb"):
        RelHum = rh / 100.0

        def f(tdb):
            return psychrolib.GetTWetBulbFromRelHum(tdb, RelHum, pressure)

        TDryBulb = _solve_db_by_bisection(f, wb, t_min, t_max)
        TWetBulb = wb
        TDewPoint = psychrolib.GetTDewPointFromRelHum(TDryBulb, RelHum)
    elif pair == ("rh", "dp"):
        RelHum = rh / 100.0

        def f(tdb):
            return psychrolib.GetTDewPointFromRelHum(tdb, RelHum)

        TDryBulb = _solve_db_by_bisection(f, dp, t_min, t_max)
        TDewPoint = dp
        TWetBulb = psychrolib.GetTWetBulbFromRelHum(TDryBulb, RelHum, pressure)
    elif pair == ("dp", "wb"):
        TDewPoint = dp

        def f(tdb):
            return psychrolib.GetTWetBulbFromTDewPoint(tdb, TDewPoint, pressure)

        TDryBulb = _solve_db_by_bisection(f, wb, t_min, t_max)
        TWetBulb = wb
        RelHum = psychrolib.GetRelHumFromTDewPoint(TDryBulb, TDewPoint)
    else:
        raise ValueError("Unsupported combination of inputs.")

    # Now compute core properties
    HumRatio = psychrolib.GetHumRatioFromTDewPoint(TDewPoint, pressure)
    Enthalpy = psychrolib.GetMoistAirEnthalpy(TDryBulb, HumRatio)
    SpecVol = psychrolib.GetMoistAirVolume(TDryBulb, HumRatio, pressure)
    MoistAirDensity = psychrolib.GetMoistAirDensity(TDryBulb, HumRatio, pressure)

    return {
        KEY_DB: TDryBulb,
        KEY_RH: RelHum * 100.0,
        KEY_WB: TWetBulb,
        KEY_DP: TDewPoint,
        KEY_HR: HumRatio,
        KEY_ENTH: Enthalpy,
        KEY_VOL: SpecVol,
        KEY_DENS: MoistAirDensity,
    }

# --- 3. THE UI LAYOUT ---
# Inline logo + title (single line, aligned)
def get_logo_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    logo_b64 = get_logo_base64("psychrovibe_logo.png")
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 0.25rem;">
            <img src="data:image/png;base64,{logo_b64}" alt="PsychroVibe" style="height: 42px; width: auto; vertical-align: middle;" />
            <span style="font-size: 2.25rem; font-weight: 700; color: #31333F;">PsychroVibe</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    st.title("PsychroVibe")
st.markdown("Interactive Psychrometric Chart Analyzing the Thermodynamic Properties of Moist Air")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Input State")
    # Mode selector: Single State Point or Process
    mode = st.radio("Mode:", ["Single State Point", "Process"], index=0)
    
    if mode == "Process":
        st.markdown("**Point 1:** Select **exactly two** of the four state inputs:")
    else:
        st.markdown("Select **exactly two** of the four state inputs:")

    # Dry Bulb
    db_default = 24.0 if USE_SI else 75.0
    db_sel_col, db_input_col = st.columns([0.25, 0.75])
    use_db = db_sel_col.checkbox("DB", value=True)
    db_input = db_input_col.number_input(f"Dry Bulb Temp ({TEMP_LABEL})", value=db_default, step=0.5)

    # Relative Humidity
    rh_sel_col, rh_input_col = st.columns([0.25, 0.75])
    use_rh = rh_sel_col.checkbox("RH", value=True)
    rh_input = rh_input_col.number_input(
        "Relative Humidity (%)", value=50.0, step=0.5, min_value=0.0, max_value=100.0
    )

    # Wet Bulb
    wb_default = 18.0 if USE_SI else 65.0
    wb_sel_col, wb_input_col = st.columns([0.25, 0.75])
    use_wb = wb_sel_col.checkbox("WB", value=False)
    wb_input = wb_input_col.number_input(f"Wet Bulb Temp ({TEMP_LABEL})", value=wb_default, step=0.5)

    # Dew Point
    dp_default = 13.0 if USE_SI else 55.0
    dp_sel_col, dp_input_col = st.columns([0.25, 0.75])
    use_dp = dp_sel_col.checkbox("DP", value=False)
    dp_input = dp_input_col.number_input(f"Dew Point Temp ({TEMP_LABEL})", value=dp_default, step=0.5)

    st.divider()

    # Helper function to get inputs dict from checkboxes and inputs
    def get_inputs_dict(use_db_val, db_val, use_rh_val, rh_val, use_wb_val, wb_val, use_dp_val, dp_val):
        selected_keys = []
        inputs = {}
        if use_db_val:
            selected_keys.append("db")
            inputs["db"] = db_val
        else:
            inputs["db"] = None
        if use_rh_val:
            selected_keys.append("rh")
            inputs["rh"] = rh_val
        else:
            inputs["rh"] = None
        if use_wb_val:
            selected_keys.append("wb")
            inputs["wb"] = wb_val
        else:
            inputs["wb"] = None
        if use_dp_val:
            selected_keys.append("dp")
            inputs["dp"] = dp_val
        else:
            inputs["dp"] = None
        return selected_keys, inputs

    # Point 1 inputs
    selected_keys_1, inputs_1 = get_inputs_dict(use_db, db_input, use_rh, rh_input, use_wb, wb_input, use_dp, dp_input)

    # Point 2 inputs (only in Process mode)
    if mode == "Process":
        st.markdown("**Point 2:** Select **exactly two** of the four state inputs:")
        
        # Point 2 inputs
        db_default_2 = 29.0 if USE_SI else 85.0
        db_sel_col_2, db_input_col_2 = st.columns([0.25, 0.75])
        use_db_2 = db_sel_col_2.checkbox("DB", value=True, key="db2")
        db_input_2 = db_input_col_2.number_input(f"Dry Bulb Temp ({TEMP_LABEL})", value=db_default_2, step=0.5, key="db_input2")

        rh_sel_col_2, rh_input_col_2 = st.columns([0.25, 0.75])
        use_rh_2 = rh_sel_col_2.checkbox("RH", value=True, key="rh2")
        rh_input_2 = rh_input_col_2.number_input(
            "Relative Humidity (%)", value=40.0, step=0.5, min_value=0.0, max_value=100.0, key="rh_input2"
        )

        wb_default_2 = 21.0 if USE_SI else 70.0
        wb_sel_col_2, wb_input_col_2 = st.columns([0.25, 0.75])
        use_wb_2 = wb_sel_col_2.checkbox("WB", value=False, key="wb2")
        wb_input_2 = wb_input_col_2.number_input(f"Wet Bulb Temp ({TEMP_LABEL})", value=wb_default_2, step=0.5, key="wb_input2")

        dp_default_2 = 15.5 if USE_SI else 60.0
        dp_sel_col_2, dp_input_col_2 = st.columns([0.25, 0.75])
        use_dp_2 = dp_sel_col_2.checkbox("DP", value=False, key="dp2")
        dp_input_2 = dp_input_col_2.number_input(f"Dew Point Temp ({TEMP_LABEL})", value=dp_default_2, step=0.5, key="dp_input2")

        st.divider()
        selected_keys_2, inputs_2 = get_inputs_dict(use_db_2, db_input_2, use_rh_2, rh_input_2, use_wb_2, wb_input_2, use_dp_2, dp_input_2)
    else:
        selected_keys_2, inputs_2 = None, None

    # Format helpers: 2 decimals default; 4 for humidity ratio and density
    def fmt(key, val):
        if key == KEY_HR or key == KEY_DENS:
            return f"{val:.4f}"
        return f"{val:.2f}"

    # Chart bounds in solver units (°C when SI, °F when IP)
    t_min_solve = (chart_t_min - 32.0) * 5.0 / 9.0 if USE_SI else chart_t_min
    t_max_solve = (chart_t_max - 32.0) * 5.0 / 9.0 if USE_SI else chart_t_max

    # Solve Point 1
    if len(selected_keys_1) != 2:
        st.warning("Point 1: Please select exactly two inputs (DB, RH, WB, DP).")
        results_1 = None
    else:
        try:
            results_1 = solve_state_from_two(
                db=inputs_1["db"],
                rh=inputs_1["rh"],
                wb=inputs_1["wb"],
                dp=inputs_1["dp"],
                pressure=PRESSURE,
                t_min=t_min_solve,
                t_max=t_max_solve,
            )
            if mode == "Single State Point":
                st.success("Point 1 Calculated")
                st.markdown("**Point 1 Properties:**")
                use_grains = st.session_state.get("hr_grains_toggle", False)
                for key, value in results_1.items():
                    if key == KEY_HR:
                        if not USE_SI and use_grains:
                            st.metric("Humidity Ratio (gr/lb)", fmt(key, value * 7000.0))
                        else:
                            st.metric(key, fmt(key, value))
                    else:
                        st.metric(key, fmt(key, value))
        except Exception as e:
            st.error(f"Point 1: Unable to solve state from selected inputs: {e}")
            results_1 = None

    # Solve Point 2 (only in Process mode)
    results_2 = None
    if mode == "Process":
        if len(selected_keys_2) != 2:
            st.warning("Point 2: Please select exactly two inputs (DB, RH, WB, DP).")
            results_2 = None
        else:
            try:
                results_2 = solve_state_from_two(
                    db=inputs_2["db"],
                    rh=inputs_2["rh"],
                    wb=inputs_2["wb"],
                    dp=inputs_2["dp"],
                    pressure=PRESSURE,
                    t_min=t_min_solve,
                    t_max=t_max_solve,
                )
            except Exception as e:
                st.error(f"Point 2: Unable to solve state from selected inputs: {e}")
                results_2 = None

    # Store mode and results in session state for chart and full-width output
    st.session_state["mode"] = mode
    st.session_state["results_1"] = results_1
    st.session_state["results_2"] = results_2
    
    # For backward compatibility, set results to results_1 for single state point mode
    if mode == "Single State Point":
        results = results_1
    else:
        results = None  # Will handle both points separately in chart section

with col2:
    st.header("Psychrometric Chart")
    # Use global HR unit toggle for chart axis and hovers
    use_grains = st.session_state.get("hr_grains_toggle", False)
    
    # Get mode and results from col1 scope (need to access them)
    # We'll handle this by checking if results_1 and results_2 exist in session state or pass them differently
    # For now, we'll check mode and results variables - but they're in col1 scope
    # Actually, we can't directly access variables from col1 in col2. We need to use session state or restructure.
    # Let me use session state to pass the mode and results

    # Generate Chart Data (chart limits in °C when USE_SI, °F when IP)
    if USE_SI:
        chart_T_min = (chart_t_min - 32.0) * 5.0 / 9.0
        chart_T_max = (chart_t_max - 32.0) * 5.0 / 9.0
    else:
        chart_T_min, chart_T_max = chart_t_min, chart_t_max
    sat_temps, sat_hr = get_saturation_curve(chart_T_min, chart_T_max, PRESSURE)
    T_MIN, T_MAX = sat_temps[0], sat_temps[-1]
    P = PRESSURE
    max_hr = max([hr for hr in sat_hr if hr is not None])
    # Lookup table for saturation humidity ratio by dry-bulb (to clip lines inside envelope)
    sat_lookup = {int(t): hr for t, hr in zip(sat_temps, sat_hr) if hr is not None}

    # Cap chart top: max HR = humidity ratio at (DB=chart_t_max, WB=nearest 5°F >= 90% of chart_t_max)
    wb_cap = math.ceil(0.9 * T_MAX / 5.0) * 5.0
    try:
        cap_hr = psychrolib.GetHumRatioFromTWetBulb(T_MAX, wb_cap, P)
        max_hr = min(max_hr, cap_hr)
    except Exception:
        pass  # keep max_hr from saturation if WB cap fails

    # Scale factor for y-axis (grains only in IP)
    y_scale = (7000.0 if use_grains else 1.0) if not USE_SI else 1.0

    # Create Plotly Figure
    fig = go.Figure()

    # 1. Dry Bulb minor lines (every 2°F) - vertical (primary)
    for db in range(int(T_MIN), int(T_MAX) + 1, 2):
        hr_top = sat_lookup.get(int(db))
        if hr_top is None:
            continue
        hr_capped = min(hr_top, max_hr)
        fig.add_trace(go.Scatter(
            x=[db, db],
            y=[0, hr_capped * y_scale],
            mode='lines',
            line=dict(color=PRIMARY_LINE_COLOR, width=1),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))

    # 2. Wet Bulb minor lines (every 5°F) - sloping
    wb_labels = []  # Store labels for wet bulb lines
    WB_LINE_COLOR = SECONDARY_LINE_COLOR  # lightsteelblue
    for twb in range(int(T_MIN), int(T_MAX) - 10, 5):
        tdb_vals = []
        w_vals = []
        for tdb in np.arange(max(twb, T_MIN), T_MAX + 1, 1):
            try:
                w = psychrolib.GetHumRatioFromTWetBulb(tdb, twb, P)
            except:
                continue
            ws = sat_lookup.get(int(tdb))
            if ws is None or w is None or w < 0 or w > max_hr or w > ws:
                continue
            tdb_vals.append(tdb)
            w_vals.append(w)
        if len(tdb_vals) > 1:
            w_vals_scaled = [w * y_scale for w in w_vals]
            fig.add_trace(go.Scatter(
                x=tdb_vals,
                y=w_vals_scaled,
                mode='lines',
                line=dict(color=WB_LINE_COLOR, width=1),
                opacity=0.35,
                showlegend=False,
                hoverinfo='skip'
            ))
            # Find intersection with saturation curve (where WB = DB, i.e., at saturation)
            # Label position: bottom-right corner of box will touch the line at saturation curve
            if twb >= T_MIN and twb <= T_MAX:
                sat_w = sat_lookup.get(int(twb))
                if sat_w is not None:
                    label_x = twb
                    label_y = sat_w * y_scale
                    wb_labels.append({
                        'x': label_x,
                        'y': label_y,
                        'text': f'{twb}',
                        'twb': twb
                    })

    # 3. Dew Point minor lines (every 5°F) - horizontal
    dp_labels = []  # Store labels for dew point lines
    DP_LINE_COLOR = "lightgreen"
    for tdp in range(int(T_MIN), int(T_MAX) - 10, 5):
        try:
            w_dp = psychrolib.GetHumRatioFromTDewPoint(tdp, P)
        except:
            continue
        if w_dp <= 0 or w_dp >= max_hr:
            continue
        t_vals = []
        w_vals = []
        for t in np.arange(T_MIN, T_MAX + 1, 1):
            ws = sat_lookup.get(int(t))
            if ws is None or w_dp > ws:
                continue
            t_vals.append(t)
            w_vals.append(w_dp)
        if len(t_vals) > 1:
            w_vals_scaled = [w * y_scale for w in w_vals]
            fig.add_trace(go.Scatter(
                x=t_vals,
                y=w_vals_scaled,
                mode='lines',
                line=dict(color=DP_LINE_COLOR, width=1),
                opacity=0.25,
                showlegend=False,
                hoverinfo='skip'
            ))
            # Label position: on the right side inside chart (so visible on y-axis side)
            if w_dp <= max_hr:
                label_x = T_MAX - (T_MAX - T_MIN) * 0.02  # just inside right edge
                label_y = w_dp * y_scale
                dp_labels.append({
                    'x': label_x,
                    'y': label_y,
                    'text': f'{tdp}',
                    'tdp': tdp
                })

    # 4. Enthalpy minor lines - sloping (5 Btu/lb IP, 10000 J/kg SI)
    try:
        h_min = psychrolib.GetMoistAirEnthalpy(T_MIN, 0.0)
    except:
        h_min = 0.0
    try:
        h_max = psychrolib.GetMoistAirEnthalpy(T_MAX, max_hr)
    except:
        h_max = h_min + (60000.0 if USE_SI else 60.0)

    if USE_SI:
        h_step = 10000.0
        h_start = int(np.ceil(h_min / h_step) * h_step)
        h_end = int(np.floor(h_max / h_step) * h_step)
    else:
        h_step = 5.0
        h_start = int(np.ceil(h_min / h_step) * h_step)
        h_end = int(np.floor(h_max / h_step) * h_step)

    for h in range(int(h_start), int(h_end) + 1, int(h_step)):
        t_vals = []
        w_vals = []
        for t in np.arange(T_MIN, T_MAX + 1, 1):
            try:
                w = psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(h, t)
            except:
                continue
            ws = sat_lookup.get(int(t))
            if ws is None or w is None or w < 0 or w > max_hr or w > ws:
                continue
            t_vals.append(t)
            w_vals.append(w)
        if len(t_vals) > 1:
            w_vals_scaled = [w * y_scale for w in w_vals]
            fig.add_trace(go.Scatter(
                x=t_vals,
                y=w_vals_scaled,
                mode='lines',
                line=dict(color=SECONDARY_LINE_COLOR, width=1, dash='dash'),
                opacity=0.25,
                showlegend=False,
                hoverinfo='skip'
            ))

    # 5. Humidity Ratio minor lines - horizontal (10 gr/lb IP, 0.001 kg/kg SI)
    if USE_SI:
        hr_step = 0.001
        n_max = int(max_hr / hr_step)
    else:
        hr_step = 10.0 / 7000.0  # 10 grains per lb
        n_max = int(max_hr / hr_step)
    for n in range(1, n_max):
        w = n * hr_step
        t_vals = []
        w_vals = []
        for t in np.arange(T_MIN, T_MAX + 1, 1):
            ws = sat_lookup.get(int(t))
            if ws is None or w > ws:
                continue
            t_vals.append(t)
            w_vals.append(w)
        if len(t_vals) > 1:
            w_vals_scaled = [w * y_scale for w in w_vals]
            fig.add_trace(go.Scatter(
                x=t_vals,
                y=w_vals_scaled,
                mode='lines',
                line=dict(color=SECONDARY_LINE_COLOR, width=0.7, dash='dot'),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))

    # 6. Plot minor RH lines (10% increments) as primary guides
    rh_annotations = []
    for rh in range(10, 100, 10):  # 10%, 20%, ..., 90%
        rh_temps, rh_hr = get_rh_curve(rh, T_MIN, T_MAX, P)
        # Scale y-values and cap at max_hr
        rh_hr_scaled = [min(hr, max_hr) * y_scale if hr is not None else None for hr in rh_hr]
        fig.add_trace(go.Scatter(
            x=rh_temps,
            y=rh_hr_scaled,
            mode='lines',
            name=f'{rh}% RH',
            line=dict(color=PRIMARY_LINE_COLOR, width=1),
            opacity=0.35,
            showlegend=False,
            hoverinfo='skip'
        ))
        # Label ~3/4 along curve from left (higher temp side), lighter text to match curve, no box
        n_pts = len(rh_temps)
        idx = min(int(n_pts * 0.75), n_pts - 1)
        while idx >= 0 and (rh_hr_scaled[idx] is None or rh_hr_scaled[idx] <= 0):
            idx -= 1
        if idx >= 0 and rh_hr_scaled[idx] is not None:
            rh_annotations.append(dict(
                x=rh_temps[idx],
                y=rh_hr_scaled[idx],
                text=f"{rh}%",
                showarrow=False,
                xref="x",
                yref="y",
                font=dict(size=9, color="#666666"),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
                borderpad=0,
            ))

    # 7. Plot Saturation Curve (100% RH) on top of minor lines (primary, bolder)
    sat_hr_scaled = [min(hr, max_hr) * y_scale if hr is not None else None for hr in sat_hr]
    fig.add_trace(go.Scatter(
        x=sat_temps,
        y=sat_hr_scaled,
        mode='lines',
        name='Saturation Line (100% RH)',
        line=dict(color=PRIMARY_LINE_COLOR, width=2)
    ))

    # Legend-only traces for Wet Bulb and Dew Point (with matching line colors)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color=WB_LINE_COLOR, width=1),
        name='Wet Bulb',
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color=DP_LINE_COLOR, width=1),
        name='Dew Point',
        showlegend=True,
    ))

    # Add Wet Bulb labels: bottom-right corner of box touches WB line at saturation curve
    wb_annotations = []
    for label in wb_labels:
        wb_annotations.append(dict(
            x=label['x'],
            y=label['y'],
            text=label['text'],
            showarrow=False,
            xref="x",
            yref="y",
            xanchor="right",
            yanchor="bottom",
            font=dict(size=9, color=WB_LINE_COLOR),
            bgcolor="white",
            bordercolor=WB_LINE_COLOR,
            borderwidth=1,
            borderpad=2,
        ))

    # Add Dew Point value labels on the right (like WB labels: box, green to match DP lines)
    dp_annotations = []
    for label in dp_labels:
        dp_annotations.append(dict(
            x=label['x'],
            y=label['y'],
            text=label['text'],
            showarrow=False,
            xref="x",
            yref="y",
            font=dict(size=9, color=DP_LINE_COLOR),
            bgcolor="white",
            bordercolor=DP_LINE_COLOR,
            borderwidth=1,
            borderpad=2,
        ))

    # 8. Plot User State Point(s) (if solved)
    # Single State Point mode
    if mode == "Single State Point" and results_1 is not None:
        results = results_1
        hr_value = results[KEY_HR]
        hr_display_value = (hr_value * 7000.0 if use_grains else hr_value) if not USE_SI else hr_value
        hr_unit = "gr/lb" if (not USE_SI and use_grains) else ("kg/kg" if USE_SI else "lb/lb")
        
        # Custom data order: DB, HR (in display units), RH, WB, DP
        state_customdata = [[
            results[KEY_DB],
            hr_display_value,
            results[KEY_RH],
            results[KEY_WB],
            results[KEY_DP],
        ]]
        hover_tmpl = (
            f"DB: %{{customdata[0]:.2f}} {TEMP_LABEL}<br>"
            f"HR: %{{customdata[1]:.2f}} {hr_unit}<br>"
            "RH: %{customdata[2]:.2f} %<br>"
            f"WB: %{{customdata[3]:.2f}} {TEMP_LABEL}<br>"
            f"DP: %{{customdata[4]:.2f}} {TEMP_LABEL}<br>"
            "<extra></extra>"
        )

        # Visible red state point (scale y-coordinate based on unit selection)
        fig.add_trace(go.Scatter(
            x=[results[KEY_DB]],
            y=[hr_value * y_scale],
            mode='markers+text',
            name='Current State',
            showlegend=False,
            text=['State Point'],
            textposition="top center",
            marker=dict(size=12, color='red', symbol='x'),
            customdata=state_customdata,
            hovertemplate=hover_tmpl,
            hoverlabel=dict(
                bgcolor="#ffcccc",  # light red
                font_color="black",
            ),
        ))

        # Invisible hover grid so state details appear anywhere on the chart
        grid_x = np.linspace(T_MIN, T_MAX, 40)
        grid_y = np.linspace(0, max_hr, 20) * y_scale
        gx, gy = np.meshgrid(grid_x, grid_y)
        gx = gx.ravel()
        gy = gy.ravel()
        grid_customdata = np.repeat(state_customdata, len(gx), axis=0)

        fig.add_trace(go.Scatter(
            x=gx,
            y=gy,
            mode='markers',
            marker=dict(size=4, opacity=0.0),
            showlegend=False,
            hovertemplate=hover_tmpl,
            customdata=grid_customdata,
        ))
    
    # Process mode: Plot both points
    elif mode == "Process" and results_1 is not None and results_2 is not None:
        hr_unit = "gr/lb" if (not USE_SI and use_grains) else ("kg/kg" if USE_SI else "lb/lb")
        
        # Point 1
        hr_value_1 = results_1[KEY_HR]
        hr_display_value_1 = (hr_value_1 * 7000.0 if use_grains else hr_value_1) if not USE_SI else hr_value_1
        state_customdata_1 = [[
            results_1[KEY_DB],
            hr_display_value_1,
            results_1[KEY_RH],
            results_1[KEY_WB],
            results_1[KEY_DP],
        ]]
        hover_tmpl_1 = (
            "Point 1<br>"
            f"DB: %{{customdata[0]:.2f}} {TEMP_LABEL}<br>"
            f"HR: %{{customdata[1]:.2f}} {hr_unit}<br>"
            "RH: %{customdata[2]:.2f} %<br>"
            f"WB: %{{customdata[3]:.2f}} {TEMP_LABEL}<br>"
            f"DP: %{{customdata[4]:.2f}} {TEMP_LABEL}<br>"
            "<extra></extra>"
        )
        
        fig.add_trace(go.Scatter(
            x=[results_1[KEY_DB]],
            y=[hr_value_1 * y_scale],
            mode='markers+text',
            name='Point 1',
            showlegend=False,
            text=['Point 1'],
            textposition="top center",
            marker=dict(size=12, color='blue', symbol='circle'),
            customdata=state_customdata_1,
            hovertemplate=hover_tmpl_1,
            hoverlabel=dict(
                bgcolor="#ccccff",
                font_color="black",
            ),
        ))
        
        # Point 2
        hr_value_2 = results_2[KEY_HR]
        hr_display_value_2 = (hr_value_2 * 7000.0 if use_grains else hr_value_2) if not USE_SI else hr_value_2
        state_customdata_2 = [[
            results_2[KEY_DB],
            hr_display_value_2,
            results_2[KEY_RH],
            results_2[KEY_WB],
            results_2[KEY_DP],
        ]]
        hover_tmpl_2 = (
            "Point 2<br>"
            f"DB: %{{customdata[0]:.2f}} {TEMP_LABEL}<br>"
            f"HR: %{{customdata[1]:.2f}} {hr_unit}<br>"
            "RH: %{customdata[2]:.2f} %<br>"
            f"WB: %{{customdata[3]:.2f}} {TEMP_LABEL}<br>"
            f"DP: %{{customdata[4]:.2f}} {TEMP_LABEL}<br>"
            "<extra></extra>"
        )
        
        fig.add_trace(go.Scatter(
            x=[results_2[KEY_DB]],
            y=[hr_value_2 * y_scale],
            mode='markers+text',
            name='Point 2',
            showlegend=False,
            text=['Point 2'],
            textposition="top center",
            marker=dict(size=12, color='green', symbol='square'),
            customdata=state_customdata_2,
            hovertemplate=hover_tmpl_2,
            hoverlabel=dict(
                bgcolor="#ccffcc",
                font_color="black",
            ),
        ))
        
        # Draw line connecting the two points
        fig.add_trace(go.Scatter(
            x=[results_1[KEY_DB], results_2[KEY_DB]],
            y=[hr_value_1 * y_scale, hr_value_2 * y_scale],
            mode='lines',
            name='Process Line',
            showlegend=False,
            line=dict(color='gray', width=2, dash='dash'),
            hoverinfo='skip',
        ))
        
        # Invisible hover grid showing Point 1 data (for consistency)
        grid_x = np.linspace(T_MIN, T_MAX, 40)
        grid_y = np.linspace(0, max_hr, 20) * y_scale
        gx, gy = np.meshgrid(grid_x, grid_y)
        gx = gx.ravel()
        gy = gy.ravel()
        grid_customdata_1 = np.repeat(state_customdata_1, len(gx), axis=0)
        
        fig.add_trace(go.Scatter(
            x=gx,
            y=gy,
            mode='markers',
            marker=dict(size=4, opacity=0.0),
            showlegend=False,
            hovertemplate=hover_tmpl_1,
            customdata=grid_customdata_1,
        ))
        
        results = None  # Not used in process mode

    # Chart Cosmetics simulating a real Psych Chart
    # Determine y-axis title and range (flat top at capped max_hr)
    if USE_SI:
        yaxis_title = "Humidity Ratio (kg w / kg da)"
        yaxis_range = [0, max_hr]
    elif use_grains:
        yaxis_title = "Humidity Ratio (gr w / lb da)"
        yaxis_range = [0, max_hr * 7000.0]
    else:
        yaxis_title = "Humidity Ratio (lb w / lb da)"
        yaxis_range = [0, max_hr]
    
    fig.update_layout(
        xaxis_title=f"Dry Bulb Temperature ({TEMP_LABEL})",
        yaxis_title=yaxis_title,
        yaxis=dict(
            side="right",
            showgrid=False,
            zeroline=False,
            range=yaxis_range,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            range=[T_MIN, T_MAX],
        ),
        height=600,
        plot_bgcolor='white',
        hovermode="closest",
        font=dict(family="Arial", size=12, color="black"),
        margin=dict(l=60, r=60, t=40, b=60),
        annotations=wb_annotations + dp_annotations + rh_annotations,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Chart & Altitude Settings")
    limits_col, altitude_col, toggles_col = st.columns([2, 2, 1])

    with limits_col:
        if USE_SI:
            chart_min_disp = (chart_t_min - 32.0) * 5.0 / 9.0
            chart_max_disp = (chart_t_max - 32.0) * 5.0 / 9.0
            new_c_min = st.number_input(
                "Chart Min Dry Bulb (°C)",
                key="chart_t_min_c",
                value=round(chart_min_disp, 1),
                step=0.5,
                min_value=-30.0,
                max_value=65.0,
            )
            new_c_max = st.number_input(
                "Chart Max Dry Bulb (°C)",
                key="chart_t_max_c",
                value=round(chart_max_disp, 1),
                step=0.5,
                min_value=new_c_min + 3.0,
                max_value=65.0,
            )
            st.session_state["chart_t_min"] = new_c_min * 9.0 / 5.0 + 32.0
            st.session_state["chart_t_max"] = new_c_max * 9.0 / 5.0 + 32.0
        else:
            new_chart_t_min = st.number_input(
                "Chart Min Dry Bulb (°F)",
                key="chart_t_min",
                value=chart_t_min,
                step=1.0,
                min_value=-20.0,
                max_value=150.0,
            )
            new_chart_t_max = st.number_input(
                "Chart Max Dry Bulb (°F)",
                key="chart_t_max",
                value=chart_t_max,
                step=1.0,
                min_value=new_chart_t_min + 5.0,
                max_value=150.0,
            )

    with altitude_col:
        if USE_SI:
            alt_m = st.number_input(
                "Altitude (m)",
                key="altitude_m",
                value=round(altitude_ft * 0.3048, 0),
                step=100.0,
                min_value=-300.0,
                max_value=6000.0,
            )
            st.session_state["altitude_ft"] = alt_m / 0.3048
        else:
            new_altitude = st.number_input(
                "Altitude (ft)",
                key="altitude_ft",
                value=altitude_ft,
                step=100.0,
                min_value=-1000.0,
                max_value=20000.0,
            )

        # Recompute density at current state for display (single state point mode only)
        if mode == "Single State Point" and results_1 is not None:
            st.metric(KEY_DENS, f"{results_1[KEY_DENS]:.4f}")

    with toggles_col:
        st.markdown("**Display**")
        st.radio(
            "Units",
            ["°F (IP)", "°C (SI)"],
            index=0,  # default first time; after that key="unit_radio" holds selection
            key="unit_radio",
        )
        st.session_state["use_si_units"] = (st.session_state["unit_radio"] == "°C (SI)")
        if not USE_SI:
            st.checkbox("HR in gr/lb", value=st.session_state.get("hr_grains_toggle", False), key="hr_grains_toggle")

# Full-width section: Process mode — Point 1 | Point 2 | Deltas in three separate columns
if st.session_state.get("mode") == "Process":
    r1 = st.session_state.get("results_1")
    r2 = st.session_state.get("results_2")
    use_grains_hr = st.session_state.get("hr_grains_toggle", False)
    if r1 is not None and r2 is not None:
        use_si_deltas = st.session_state.get("use_si_units", False)
        def _fmt(key, val):
            if key == KEY_HR or key == KEY_DENS:
                return f"{val:.4f}"
            return f"{val:.2f}"
        st.success("Point 1 & Point 2 Calculated")
        col_p1, col_p2, col_deltas = st.columns(3)
        with col_p1:
            st.markdown("**Point 1 Properties:**")
            for key, value in r1.items():
                if key == KEY_HR:
                    if not use_si_deltas and use_grains_hr:
                        st.metric("Humidity Ratio (gr/lb)", _fmt(key, value * 7000.0))
                    else:
                        st.metric(key, _fmt(key, value))
                else:
                    st.metric(key, _fmt(key, value))
        with col_p2:
            st.markdown("**Point 2 Properties:**")
            for key, value in r2.items():
                if key == KEY_HR:
                    if not use_si_deltas and use_grains_hr:
                        st.metric("Humidity Ratio (gr/lb)", _fmt(key, value * 7000.0))
                    else:
                        st.metric(key, _fmt(key, value))
                else:
                    st.metric(key, _fmt(key, value))
        with col_deltas:
            st.markdown("**Process Deltas (Point 1 → Point 2):**")
            delta_enthalpy = r2[KEY_ENTH] - r1[KEY_ENTH]
            delta_hr = r2[KEY_HR] - r1[KEY_HR]
            st.metric(f"Δ {KEY_ENTH}", _fmt(KEY_ENTH, delta_enthalpy))
            if not use_si_deltas and use_grains_hr:
                st.metric("Δ Humidity Ratio (gr/lb)", _fmt(KEY_HR, delta_hr * 7000.0))
            else:
                st.metric(f"Δ {KEY_HR}", _fmt(KEY_HR, delta_hr))