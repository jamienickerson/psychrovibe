import streamlit as st
import plotly.graph_objects as go
import psychrolib
import numpy as np
import pandas as pd
import math
import base64

# --- 1. CONFIGURATION & ENGINE ---
st.set_page_config(page_title="PsychroVibe: Critical Cooling", layout="wide", page_icon="psychrovibe_logo.png")

# Top padding: enough so header title is not clipped, still compact for chart
st.markdown("""
    <style>
    .block-container { padding-top: 2.5rem !important; }
    </style>
    """, unsafe_allow_html=True)

# Unit system: IP (°F, Btu/lb, lb/lb, etc.) vs SI (°C, J/kg, kg/kg, etc.)
# Radio "unit_radio" is source of truth: °F (IP) = Imperial, °C (SI) = SI
if "unit_radio" not in st.session_state:
    st.session_state["unit_radio"] = "°F (IP)"
if "prev_unit_system" not in st.session_state:
    st.session_state["prev_unit_system"] = "IP"

_unit_radio = st.session_state["unit_radio"]
USE_SI = (_unit_radio == "°C (SI)")  # True when user chose °C (SI)
current_unit_system = "SI" if USE_SI else "IP"
unit_system_changed = (st.session_state["prev_unit_system"] != current_unit_system)
st.session_state["prev_unit_system"] = current_unit_system

psychrolib.SetUnitSystem(psychrolib.SI if USE_SI else psychrolib.IP)

# Conversion functions
def f_to_c(temp_f):
    return (temp_f - 32.0) * 5.0 / 9.0

def c_to_f(temp_c):
    return temp_c * 9.0 / 5.0 + 32.0

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


def smoothstep(edge0, edge1, x):
    """Smooth interpolation function (smoothstep) for gradual transitions."""
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def find_saturation_intersection(t_start, t_end, w_target, sat_lookup, T_MIN, T_MAX):
    """
    Find the exact temperature where a horizontal line (constant w_target) intersects the saturation curve.
    Returns the intersection temperature, or None if no intersection found.
    """
    # Binary search for intersection
    t_low = max(t_start, T_MIN)
    t_high = min(t_end, T_MAX)
    
    # Check if intersection exists
    sat_low = sat_lookup.get(int(t_low))
    sat_high = sat_lookup.get(int(t_high))
    
    if sat_low is None or sat_high is None:
        return None
    
    # If w_target is above saturation at both ends, no intersection
    if w_target > sat_low and w_target > sat_high:
        return None
    # If w_target is below saturation at both ends, intersection is at t_high
    if w_target < sat_low and w_target < sat_high:
        return t_high
    
    # Binary search for exact intersection
    for _ in range(20):  # 20 iterations should be enough for 0.01° precision
        t_mid = (t_low + t_high) / 2.0
        sat_mid = sat_lookup.get(int(t_mid))
        if sat_mid is None:
            break
        if abs(w_target - sat_mid) < 0.00001:  # Close enough
            return t_mid
        if w_target > sat_mid:
            t_low = t_mid
        else:
            t_high = t_mid
    
    return (t_low + t_high) / 2.0


def find_saturation_intersection_sloping(t_start, t_end, w_func, sat_lookup, T_MIN, T_MAX):
    """
    Find the exact point where a sloping line (w_func(t)) intersects the saturation curve.
    w_func should be a function that takes temperature and returns humidity ratio.
    Returns (t_intersect, w_intersect) or (None, None) if no intersection.
    """
    # Check points along the line
    t_low = max(t_start, T_MIN)
    t_high = min(t_end, T_MAX)
    
    # Find where line crosses saturation curve
    last_valid_t = None
    last_valid_w = None
    
    for t in np.arange(t_low, t_high + 0.1, 0.1):
        try:
            w = w_func(t)
            sat_w = sat_lookup.get(int(t))
            if sat_w is None:
                continue
            
            if w <= sat_w:
                last_valid_t = t
                last_valid_w = w
            else:
                # We've crossed the saturation curve
                # Find exact intersection between last valid point and current point
                if last_valid_t is not None:
                    # Binary search for exact intersection
                    t1, t2 = last_valid_t, t
                    for _ in range(15):
                        t_mid = (t1 + t2) / 2.0
                        w_mid = w_func(t_mid)
                        sat_mid = sat_lookup.get(int(t_mid))
                        if sat_mid is None:
                            break
                        if abs(w_mid - sat_mid) < 0.00001:
                            return t_mid, w_mid
                        if w_mid > sat_mid:
                            t2 = t_mid
                        else:
                            t1 = t_mid
                    return (t1 + t2) / 2.0, w_func((t1 + t2) / 2.0)
                break
        except:
            continue
    
    # If we never crossed, return the last valid point (at saturation)
    if last_valid_t is not None:
        sat_w = sat_lookup.get(int(last_valid_t))
        if sat_w is not None and abs(last_valid_w - sat_w) < 0.0001:
            return last_valid_t, last_valid_w
    
    return None, None


def adjust_saturation_curve_shape(temps, hum_ratios, t_min, t_max):
    """
    Adjusts the saturation curve shape to match reference chart style with smooth transitions:
    - Slightly higher at low temperatures (20-50°F equivalent range)
    - Similar in mid-range (50-85°F equivalent range)
    - Higher, flatter plateau at high temperatures (85°F+ equivalent) with smooth cutoff
    Works for both IP and SI by using relative temperature positions.
    Uses smooth interpolation to maintain curve smoothness.
    """
    # Check if inputs are valid (handle arrays/lists properly)
    if temps is None or hum_ratios is None or len(temps) == 0 or len(hum_ratios) == 0:
        return temps, hum_ratios
    
    adjusted_hrs = []
    temp_range = t_max - t_min
    
    for t, hr in zip(temps, hum_ratios):
        if hr is None:
            adjusted_hrs.append(None)
            continue
        
        # Normalize temperature position (0.0 = t_min, 1.0 = t_max)
        t_norm = (t - t_min) / temp_range if temp_range > 0 else 0.0
        
        # Define smooth adjustment factors using smoothstep for gradual transitions
        # Low temp adjustment: peaks at start, smoothly decreases to ~32% of range
        low_adj = smoothstep(0.32, 0.0, t_norm) * 0.04  # 4% max at start, smooth to 0 at 32%
        
        # Mid-range: minimal adjustment (~1%) centered around 50% of range
        mid_adj = 0.01 * (1.0 - abs(t_norm - 0.5) * 2.0)  # Peaks at 50%, smooth falloff
        mid_adj = max(0, mid_adj)  # Don't go negative
        
        # High temp adjustment: starts increasing around 68%, peaks around 90%, then smooth drop
        # Rising phase (68% to 90%)
        rise_adj = smoothstep(0.68, 0.90, t_norm) * 0.19  # Smooth rise from 0 to 19%
        # Plateau and drop phase (90% to 100%)
        if t_norm > 0.90:
            plateau_factor = smoothstep(0.90, 0.98, t_norm)  # 1.0 at 90%, 1.0 at 98%
            drop_factor = smoothstep(0.98, 1.0, t_norm)  # 0.0 at 98%, 1.0 at 100%
            # Maintain plateau, then smooth drop
            high_adj = 0.19 * (1.0 - drop_factor * 0.75)  # Smooth drop from 19% to ~5%
        else:
            high_adj = rise_adj
        
        # Combine adjustments smoothly
        # At low temps: use low_adj
        # At mid temps: blend low_adj and mid_adj
        # At high temps: use high_adj
        if t_norm < 0.32:
            adjustment_factor = 1.0 + low_adj
        elif t_norm < 0.68:
            # Smooth blend between low and mid adjustments
            blend = smoothstep(0.32, 0.68, t_norm)
            adjustment_factor = 1.0 + low_adj * (1.0 - blend) + mid_adj * blend
        else:
            # Smooth blend between mid and high adjustments
            blend = smoothstep(0.68, 0.84, t_norm)
            adjustment_factor = 1.0 + mid_adj * (1.0 - blend) + high_adj * blend
        
        adjusted_hr = hr * adjustment_factor
        adjusted_hrs.append(adjusted_hr)
    
    return temps, adjusted_hrs


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


def solve_state_from_enthalpy_humratio(enthalpy, hum_ratio, pressure, t_min, t_max):
    """
    Solve the complete psychrometric state from enthalpy and humidity ratio.
    Used for mixed air calculations.
    """
    try:
        TDryBulb = psychrolib.GetTDryBulbFromEnthalpyAndHumRatio(enthalpy, hum_ratio)
        # Clamp to reasonable range
        TDryBulb = max(t_min, min(t_max, TDryBulb))
    except Exception:
        # Fallback: iterate to find DB that gives correct enthalpy
        def f(tdb):
            try:
                hr_calc = psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(enthalpy, tdb)
                return hr_calc - hum_ratio
            except:
                return float('inf')
        TDryBulb = _solve_db_by_bisection(f, (t_min + t_max) / 2.0, t_min, t_max)
    
    # Now compute all other properties
    RelHum = psychrolib.GetRelHumFromHumRatio(TDryBulb, hum_ratio, pressure)
    TWetBulb = psychrolib.GetTWetBulbFromHumRatio(TDryBulb, hum_ratio, pressure)
    TDewPoint = psychrolib.GetTDewPointFromHumRatio(TDryBulb, hum_ratio, pressure)
    SpecVol = psychrolib.GetMoistAirVolume(TDryBulb, hum_ratio, pressure)
    MoistAirDensity = psychrolib.GetMoistAirDensity(TDryBulb, hum_ratio, pressure)
    
    return {
        KEY_DB: TDryBulb,
        KEY_RH: RelHum * 100.0,
        KEY_WB: TWetBulb,
        KEY_DP: TDewPoint,
        KEY_HR: hum_ratio,
        KEY_ENTH: enthalpy,
        KEY_VOL: SpecVol,
        KEY_DENS: MoistAirDensity,
    }


def calculate_properties_from_db_hr(db, hr, pressure, sat_lookup):
    """
    Calculate psychrometric properties from dry bulb temperature and humidity ratio.
    Returns None if the point is invalid (above saturation curve or out of bounds).
    Used for dynamic hover tooltips on the chart.
    """
    try:
        # Check if point is valid (below saturation curve)
        db_int = int(db)
        sat_hr = sat_lookup.get(db_int)
        if sat_hr is None or hr < 0 or hr > sat_hr:
            return None
        
        # Calculate all properties
        RelHum = psychrolib.GetRelHumFromHumRatio(db, hr, pressure)
        TWetBulb = psychrolib.GetTWetBulbFromHumRatio(db, hr, pressure)
        TDewPoint = psychrolib.GetTDewPointFromHumRatio(db, hr, pressure)
        Enthalpy = psychrolib.GetMoistAirEnthalpy(db, hr, pressure)
        
        return {
            KEY_DB: db,
            KEY_HR: hr,
            KEY_RH: RelHum * 100.0,
            KEY_WB: TWetBulb,
            KEY_DP: TDewPoint,
            KEY_ENTH: Enthalpy,
        }
    except Exception:
        return None

# --- 3. THE UI LAYOUT ---
def format_airflow(val):
    """Format airflow for display: commas for thousands, no decimals."""
    return f"{int(round(val)):,}"

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
    # Mode selector: Single State Point, Process, Air Mixing, or Cycle Analysis
    mode = st.radio("Mode:", ["Single State Point", "Process", "Air Mixing", "Cycle Analysis"], index=0)
    
    if mode == "Process":
        st.markdown("**Point 1:** Select **exactly two** of the four state inputs:")
    elif mode == "Air Mixing":
        st.markdown("**Point 1:** Select **exactly two** of the four state inputs:")
    elif mode == "Cycle Analysis":
        # --- Cycle Analysis (AHU Builder) ---
        if "cycle_points" not in st.session_state:
            st.session_state["cycle_points"] = []
        if "cycle_total_airflow" not in st.session_state:
            st.session_state["cycle_total_airflow"] = 10000.0
        if "cycle_pct_oa" not in st.session_state:
            st.session_state["cycle_pct_oa"] = 20.0
        if "cycle_ra_db" not in st.session_state:
            st.session_state["cycle_ra_db"] = 75.0 if not USE_SI else 24.0
        if "cycle_ra_rh" not in st.session_state:
            st.session_state["cycle_ra_rh"] = 50.0
        if "cycle_oa_db" not in st.session_state:
            st.session_state["cycle_oa_db"] = 95.0 if not USE_SI else 35.0
        if "cycle_oa_rh" not in st.session_state:
            st.session_state["cycle_oa_rh"] = 40.0

        st.markdown("**Mixing Box**")
        if USE_SI:
            total_airflow = st.number_input("Total System Airflow (m³/hr)", value=int(st.session_state.get("cycle_total_airflow", 5000)), step=500, min_value=100, key="cycle_total_airflow", format="%d")
        else:
            total_airflow = st.number_input("Total System Airflow (CFM)", value=int(st.session_state.get("cycle_total_airflow", 10000)), step=500, min_value=100, key="cycle_total_airflow", format="%d")
        pct_oa = st.slider("% Outside Air", 0, 100, int(st.session_state.get("cycle_pct_oa", 20)), key="cycle_pct_oa")
        st.markdown("**Return Air (RA)**")
        ra_db = st.number_input("RA Dry Bulb (" + (TEMP_LABEL + ")"), value=float(st.session_state.get("cycle_ra_db", 75.0 if not USE_SI else 24.0)), step=0.5, key="cycle_ra_db")
        ra_rh = st.number_input("RA Relative Humidity (%)", value=float(st.session_state.get("cycle_ra_rh", 50.0)), step=0.5, min_value=0.0, max_value=100.0, key="cycle_ra_rh")
        st.markdown("**Outside Air (OA)**")
        oa_db = st.number_input("OA Dry Bulb (" + (TEMP_LABEL + ")"), value=float(st.session_state.get("cycle_oa_db", 95.0 if not USE_SI else 35.0)), step=0.5, key="cycle_oa_db")
        oa_rh = st.number_input("OA Relative Humidity (%)", value=float(st.session_state.get("cycle_oa_rh", 40.0)), step=0.5, min_value=0.0, max_value=100.0, key="cycle_oa_rh")

        t_min_solve = (chart_t_min - 32.0) * 5.0 / 9.0 if USE_SI else chart_t_min
        t_max_solve = (chart_t_max - 32.0) * 5.0 / 9.0 if USE_SI else chart_t_max
        results_ra = None
        results_oa = None
        results_ma = None
        try:
            results_ra = solve_state_from_two(db=ra_db, rh=ra_rh, wb=None, dp=None, pressure=PRESSURE, t_min=t_min_solve, t_max=t_max_solve)
            results_oa = solve_state_from_two(db=oa_db, rh=oa_rh, wb=None, dp=None, pressure=PRESSURE, t_min=t_min_solve, t_max=t_max_solve)
        except Exception as e:
            st.error(f"RA or OA state invalid: {e}")
        if results_ra is not None and results_oa is not None:
            if pct_oa >= 99.5:
                results_ma = results_oa
                st.success("100% Outside Air - no mixing")
                st.caption(f"DB: {results_ma[KEY_DB]:.1f} {TEMP_LABEL}  |  WB: {results_ma[KEY_WB]:.1f} {TEMP_LABEL}  |  h: {results_ma[KEY_ENTH]:.1f} {ENTHALPY_LABEL}")
            elif pct_oa <= 0.5:
                results_ma = results_ra
                st.success("100% Return Air - no mixing")
                st.caption(f"DB: {results_ma[KEY_DB]:.1f} {TEMP_LABEL}  |  WB: {results_ma[KEY_WB]:.1f} {TEMP_LABEL}  |  h: {results_ma[KEY_ENTH]:.1f} {ENTHALPY_LABEL}")
            else:
                airflow_oa = total_airflow * (pct_oa / 100.0)
                airflow_ra = total_airflow * (1.0 - pct_oa / 100.0)
                if USE_SI:
                    mass_oa = airflow_oa * results_oa[KEY_DENS] / 3600.0
                    mass_ra = airflow_ra * results_ra[KEY_DENS] / 3600.0
                else:
                    mass_oa = airflow_oa * results_oa[KEY_DENS]
                    mass_ra = airflow_ra * results_ra[KEY_DENS]
                tot_mass = mass_oa + mass_ra
                if tot_mass > 0:
                    h_mix = (mass_oa * results_oa[KEY_ENTH] + mass_ra * results_ra[KEY_ENTH]) / tot_mass
                    w_mix = (mass_oa * results_oa[KEY_HR] + mass_ra * results_ra[KEY_HR]) / tot_mass
                    try:
                        results_ma = solve_state_from_enthalpy_humratio(h_mix, w_mix, PRESSURE, t_min_solve, t_max_solve)
                    except Exception:
                        results_ma = None
                if results_ma is not None:
                    st.success("Mixed Air Result")
                    st.caption(f"DB: {results_ma[KEY_DB]:.1f} {TEMP_LABEL}  |  WB: {results_ma[KEY_WB]:.1f} {TEMP_LABEL}  |  h: {results_ma[KEY_ENTH]:.1f} {ENTHALPY_LABEL}")

        st.divider()
        st.markdown("**Process Chain**")
        cycle_points = st.session_state.get("cycle_points", [])
        if len(cycle_points) < 3:
            st.markdown("Add next component (leaving state):")
            st.caption("Enter DB and RH for the leaving air state.")
            c_db = st.number_input("DB (" + TEMP_LABEL + ")", value=55.0 if not USE_SI else 13.0, step=0.5, key="cycle_new_db")
            c_rh = st.number_input("RH (%)", value=90.0, step=0.5, min_value=0.0, max_value=100.0, key="cycle_new_rh")
            c_name = st.text_input("Point name (e.g. Cooling Coil Leaving)", value="", key="cycle_new_name", placeholder="e.g. Coil Leaving")
            if st.button("Add Point", key="cycle_add_btn"):
                try:
                    pt_state = solve_state_from_two(db=c_db, rh=c_rh, wb=None, dp=None, pressure=PRESSURE, t_min=t_min_solve, t_max=t_max_solve)
                    name = c_name.strip() or f"Point {len(cycle_points)+1}"
                    cycle_points = st.session_state.get("cycle_points", []) + [{"name": name, "state": pt_state}]
                    st.session_state["cycle_points"] = cycle_points[:3]
                    st.rerun()
                except Exception as ex:
                    st.error(f"Invalid state: {ex}. Select exactly two of DB/RH/WB/DP.")
        for i, pt in enumerate(cycle_points):
            cols = st.columns([1, 0.15])
            with cols[0]:
                st.caption(f"{pt['name']}: DB={pt['state'][KEY_DB]:.1f}, WB={pt['state'][KEY_WB]:.1f}")
            with cols[1]:
                if st.button("X", key=f"cycle_del_{i}"):
                    cycle_points = st.session_state.get("cycle_points", [])
                    cycle_points.pop(i)
                    st.session_state["cycle_points"] = cycle_points
                    st.rerun()
        results_1 = None
        results_2 = None
        st.session_state["cycle_ra"] = results_ra
        st.session_state["cycle_oa"] = results_oa
        st.session_state["cycle_ma"] = results_ma
    else:
        st.markdown("Select **exactly two** of the four state inputs:")

    if mode != "Cycle Analysis":
        # Temperature inputs with unit conversion on system change
        # Get stored values or defaults
        db_stored = st.session_state.get("db_input_1", None)
        wb_stored = st.session_state.get("wb_input_1", None)
        dp_stored = st.session_state.get("dp_input_1", None)
        
        # Convert if unit system changed and update session state
        if unit_system_changed and db_stored is not None:
            if USE_SI:  # Switching to SI: convert °F to °C
                st.session_state["db_input_1"] = f_to_c(db_stored)
                db_stored = st.session_state["db_input_1"]
                if wb_stored is not None:
                    st.session_state["wb_input_1"] = f_to_c(wb_stored)
                    wb_stored = st.session_state["wb_input_1"]
                if dp_stored is not None:
                    st.session_state["dp_input_1"] = f_to_c(dp_stored)
                    dp_stored = st.session_state["dp_input_1"]
            else:  # Switching to IP: convert °C to °F
                st.session_state["db_input_1"] = c_to_f(db_stored)
                db_stored = st.session_state["db_input_1"]
                if wb_stored is not None:
                    st.session_state["wb_input_1"] = c_to_f(wb_stored)
                    wb_stored = st.session_state["wb_input_1"]
                if dp_stored is not None:
                    st.session_state["dp_input_1"] = c_to_f(dp_stored)
                    dp_stored = st.session_state["dp_input_1"]
        
        # Dry Bulb
        db_default = 24.0 if USE_SI else 75.0
        db_value = db_stored if db_stored is not None else db_default
        db_sel_col, db_input_col = st.columns([0.25, 0.75])
        use_db = db_sel_col.checkbox("DB", value=True)
        db_input = db_input_col.number_input(f"Dry Bulb Temp ({TEMP_LABEL})", value=db_value, step=0.5, key="db_input_1")

        # Relative Humidity (no conversion needed)
        rh_sel_col, rh_input_col = st.columns([0.25, 0.75])
        use_rh = rh_sel_col.checkbox("RH", value=True)
        rh_input = rh_input_col.number_input(
            "Relative Humidity (%)", value=st.session_state.get("rh_input_1", 50.0), step=0.5, min_value=0.0, max_value=100.0, key="rh_input_1"
        )

        # Wet Bulb
        wb_default = 18.0 if USE_SI else 65.0
        wb_value = wb_stored if wb_stored is not None else wb_default
        wb_sel_col, wb_input_col = st.columns([0.25, 0.75])
        use_wb = wb_sel_col.checkbox("WB", value=False)
        wb_input = wb_input_col.number_input(f"Wet Bulb Temp ({TEMP_LABEL})", value=wb_value, step=0.5, key="wb_input_1")

        # Dew Point
        dp_default = 13.0 if USE_SI else 55.0
        dp_value = dp_stored if dp_stored is not None else dp_default
        dp_sel_col, dp_input_col = st.columns([0.25, 0.75])
        use_dp = dp_sel_col.checkbox("DP", value=False)
        dp_input = dp_input_col.number_input(f"Dew Point Temp ({TEMP_LABEL})", value=dp_value, step=0.5, key="dp_input_1")

        # Airflow Input for Point 1
        st.markdown("**Airflow:**")
        if USE_SI:
            airflow_1 = st.number_input("Airflow (m³/hr)", value=1000, step=100, min_value=0, key="airflow_1", format="%d")
            airflow_type_1 = None  # No radio needed for SI
        else:
            airflow_col, type_col = st.columns([0.7, 0.3])
            with airflow_col:
                airflow_1 = st.number_input("Airflow", value=1000, step=100, min_value=0, key="airflow_1", format="%d")
            with type_col:
                airflow_type_1 = st.radio("", ["ACFM", "SCFM"], index=0, key="airflow_type_1", horizontal=True, label_visibility="collapsed")
    
        # Display mass flow below airflow input (updated after Point 1 is solved)
        mass_flow_1_display = st.session_state.get("mass_flow_1_display", None)
        if mass_flow_1_display is not None:
            if USE_SI:
                st.caption(f":gray[Mass Flow: {mass_flow_1_display:.3f} kg/s]")
            else:
                st.caption(f":gray[Mass Flow: {mass_flow_1_display:.2f} lb/min]")

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

        # Point 2 inputs (only in Process or Air Mixing mode)
        if mode == "Process" or mode == "Air Mixing":
            st.markdown("**Point 2:** Select **exactly two** of the four state inputs:")
        
            # Temperature inputs with unit conversion on system change
            db_stored_2 = st.session_state.get("db_input2", None)
            wb_stored_2 = st.session_state.get("wb_input2", None)
            dp_stored_2 = st.session_state.get("dp_input2", None)
        
            # Convert if unit system changed and update session state
            if unit_system_changed and db_stored_2 is not None:
                if USE_SI:  # Switching to SI: convert °F to °C
                    st.session_state["db_input2"] = f_to_c(db_stored_2)
                    db_stored_2 = st.session_state["db_input2"]
                    if wb_stored_2 is not None:
                        st.session_state["wb_input2"] = f_to_c(wb_stored_2)
                        wb_stored_2 = st.session_state["wb_input2"]
                    if dp_stored_2 is not None:
                        st.session_state["dp_input2"] = f_to_c(dp_stored_2)
                        dp_stored_2 = st.session_state["dp_input2"]
                else:  # Switching to IP: convert °C to °F
                    st.session_state["db_input2"] = c_to_f(db_stored_2)
                    db_stored_2 = st.session_state["db_input2"]
                    if wb_stored_2 is not None:
                        st.session_state["wb_input2"] = c_to_f(wb_stored_2)
                        wb_stored_2 = st.session_state["wb_input2"]
                    if dp_stored_2 is not None:
                        st.session_state["dp_input2"] = c_to_f(dp_stored_2)
                        dp_stored_2 = st.session_state["dp_input2"]
        
            # Point 2 inputs
            db_default_2 = 29.0 if USE_SI else 85.0
            db_value_2 = db_stored_2 if db_stored_2 is not None else db_default_2
            db_sel_col_2, db_input_col_2 = st.columns([0.25, 0.75])
            use_db_2 = db_sel_col_2.checkbox("DB", value=True, key="db2")
            db_input_2 = db_input_col_2.number_input(f"Dry Bulb Temp ({TEMP_LABEL})", value=db_value_2, step=0.5, key="db_input2")

            rh_sel_col_2, rh_input_col_2 = st.columns([0.25, 0.75])
            use_rh_2 = rh_sel_col_2.checkbox("RH", value=True, key="rh2")
            rh_input_2 = rh_input_col_2.number_input(
                "Relative Humidity (%)", value=st.session_state.get("rh_input2", 40.0), step=0.5, min_value=0.0, max_value=100.0, key="rh_input2"
            )

            wb_default_2 = 21.0 if USE_SI else 70.0
            wb_value_2 = wb_stored_2 if wb_stored_2 is not None else wb_default_2
            wb_sel_col_2, wb_input_col_2 = st.columns([0.25, 0.75])
            use_wb_2 = wb_sel_col_2.checkbox("WB", value=False, key="wb2")
            wb_input_2 = wb_input_col_2.number_input(f"Wet Bulb Temp ({TEMP_LABEL})", value=wb_value_2, step=0.5, key="wb_input2")

            dp_default_2 = 15.5 if USE_SI else 60.0
            dp_value_2 = dp_stored_2 if dp_stored_2 is not None else dp_default_2
            dp_sel_col_2, dp_input_col_2 = st.columns([0.25, 0.75])
            use_dp_2 = dp_sel_col_2.checkbox("DP", value=False, key="dp2")
            dp_input_2 = dp_input_col_2.number_input(f"Dew Point Temp ({TEMP_LABEL})", value=dp_value_2, step=0.5, key="dp_input2")

            # Airflow Input for Point 2
            st.markdown("**Airflow:**")
            if USE_SI:
                airflow_2 = st.number_input("Airflow (m³/hr)", value=1000, step=100, min_value=0, key="airflow_2", format="%d")
                airflow_type_2 = None
            else:
                airflow_col_2, type_col_2 = st.columns([0.7, 0.3])
                with airflow_col_2:
                    airflow_2 = st.number_input("Airflow", value=1000, step=100, min_value=0, key="airflow_2", format="%d")
                with type_col_2:
                    airflow_type_2 = st.radio("", ["ACFM", "SCFM"], index=0, key="airflow_type_2", horizontal=True, label_visibility="collapsed")
        
            # Display mass flow below airflow input (updated after Point 2 is solved)
            mass_flow_2_display = st.session_state.get("mass_flow_2_display", None)
            if mass_flow_2_display is not None:
                if USE_SI:
                    st.caption(f":gray[Mass Flow: {mass_flow_2_display:.3f} kg/s]")
                else:
                    st.caption(f":gray[Mass Flow: {mass_flow_2_display:.2f} lb/min]")

            st.divider()
            selected_keys_2, inputs_2 = get_inputs_dict(use_db_2, db_input_2, use_rh_2, rh_input_2, use_wb_2, wb_input_2, use_dp_2, dp_input_2)
        else:
            selected_keys_2, inputs_2 = None, None
            airflow_2 = 0.0
            airflow_type_2 = "ACFM"
            mass_flow_2 = None

        # Mass flow calculation helper
        def calculate_mass_flow(airflow_val, airflow_type, results_dict=None):
            """Calculate mass flow rate from airflow.
            Returns mass flow in lb/min (IP) or kg/s (SI)"""
            if results_dict is None or KEY_DENS not in results_dict:
                return None
            if USE_SI:
                # SI: airflow in m³/hr, density in kg/m³
                # Mass flow = airflow (m³/hr) * density (kg/m³) / 3600 (s/hr) = kg/s
                mass_flow = airflow_val * results_dict[KEY_DENS] / 3600.0
            else:
                # IP: airflow in CFM
                if airflow_type == "ACFM":
                    # Actual CFM: use calculated density
                    # Density in lb/ft³, airflow in CFM
                    # Mass flow = CFM * density (lb/ft³) = lb/min
                    mass_flow = airflow_val * results_dict[KEY_DENS]
                else:  # SCFM
                    # Standard CFM: use standard density 0.075 lb/ft³
                    mass_flow = airflow_val * 0.075
            return mass_flow

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
                    # Display airflow as a property
                    if USE_SI:
                        st.metric("Airflow", f"{format_airflow(airflow_1)} m³/hr")
                else:
                    airflow_label = f"{format_airflow(airflow_1)} {airflow_type_1}" if airflow_type_1 else f"{format_airflow(airflow_1)} CFM"
                    st.metric("Airflow", airflow_label)
                
                # Display mass flow for Point 1 (below airflow)
                mass_flow_1 = calculate_mass_flow(airflow_1, airflow_type_1, results_1)
                if mass_flow_1 is not None:
                    if USE_SI:
                        st.caption(f":gray[Mass Flow: {mass_flow_1:.3f} kg/s]")
                    else:
                        st.caption(f":gray[Mass Flow: {mass_flow_1:.2f} lb/min]")
                
                # Store mass flow for display below airflow input
                st.session_state["mass_flow_1_display"] = mass_flow_1
            except Exception as e:
                st.error(f"Point 1: Unable to solve state from selected inputs: {e}")
                results_1 = None

        # Solve Point 2 (only in Process or Air Mixing mode)
    results_2 = None
    if mode == "Process" or mode == "Air Mixing":
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
                # Store mass flow for display below airflow input
                if results_2 is not None:
                    mass_flow_2_calc = calculate_mass_flow(airflow_2, airflow_type_2, results_2)
                    st.session_state["mass_flow_2_display"] = mass_flow_2_calc
            except Exception as e:
                st.error(f"Point 2: Unable to solve state from selected inputs: {e}")
                results_2 = None
                st.session_state["mass_flow_2_display"] = None

    # Store mode and results in session state for chart and full-width output
    # Note: airflow values are automatically stored by widgets with key= parameters
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
    # Chart line visibility toggles (read from session state, set in Chart Settings section)
    show_wb_lines = st.session_state.get("show_wb_lines", False)
    show_dp_lines = st.session_state.get("show_dp_lines", False)
    show_enthalpy_lines = st.session_state.get("show_enthalpy_lines", True)
    show_hr_lines = st.session_state.get("show_hr_lines", True)
    
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
    # Generate saturation curve with finer step for smoother appearance
    sat_temps, sat_hr = get_saturation_curve(chart_T_min, chart_T_max, PRESSURE, step=0.5)
    T_MIN, T_MAX = sat_temps[0], sat_temps[-1]
    P = PRESSURE
    
    # Use natural saturation curve from psychrolib (already smooth and thermodynamically correct)
    # Removed shape adjustment to maintain natural smooth curve appearance
    max_hr = max([hr for hr in sat_hr if hr is not None])
    # Lookup table for saturation humidity ratio by dry-bulb (to clip lines inside envelope)
    sat_lookup = {int(t): hr for t, hr in zip(sat_temps, sat_hr) if hr is not None}

    def get_sat_hr(t, sat_lookup, pressure):
        """Saturation humidity ratio at t; use lookup first, then psychrolib so no point is skipped."""
        v = sat_lookup.get(int(t))
        if v is not None:
            return v
        try:
            return psychrolib.GetSatHumRatio(t, pressure)
        except Exception:
            return None

    def find_sat_intersection(t_lo, t_hi, w_at_t_func, tol=1e-6, max_iter=25):
        """Find t in (t_lo, t_hi) where w_at_t_func(t) == get_sat_hr(t). Returns (t, w) or None."""
        for _ in range(max_iter):
            t_mid = (t_lo + t_hi) / 2.0
            if t_hi - t_lo < tol:
                ws = get_sat_hr(t_mid, sat_lookup, P)
                if ws is not None:
                    try:
                        w_mid = w_at_t_func(t_mid)
                        if w_mid is not None:
                            return (t_mid, ws)
                    except Exception:
                        pass
                return (t_mid, get_sat_hr(t_mid, sat_lookup, P)) if get_sat_hr(t_mid, sat_lookup, P) is not None else None
            try:
                w_mid = w_at_t_func(t_mid)
            except Exception:
                t_lo = t_mid
                continue
            ws_mid = get_sat_hr(t_mid, sat_lookup, P)
            if ws_mid is None or w_mid is None:
                t_lo = t_mid
                continue
            if abs(w_mid - ws_mid) < 1e-9:
                return (t_mid, ws_mid)
            if w_mid > ws_mid:
                t_hi = t_mid
            else:
                t_lo = t_mid
        return None

    def find_sat_t_for_w(w_target, t_lo, t_hi, tol=1e-6, max_iter=25):
        """Find t in (t_lo, t_hi) where get_sat_hr(t) == w_target. Returns t or None."""
        for _ in range(max_iter):
            t_mid = (t_lo + t_hi) / 2.0
            if t_hi - t_lo < tol:
                return t_mid
            ws = get_sat_hr(t_mid, sat_lookup, P)
            if ws is None:
                t_lo = t_mid
                continue
            if abs(ws - w_target) < 1e-9:
                return t_mid
            if ws < w_target:
                t_lo = t_mid
            else:
                t_hi = t_mid
        return (t_lo + t_hi) / 2.0

    # Cap chart top: max HR = humidity ratio at (DB=chart_t_max, WB=nearest 5°F >= 90% of chart_t_max)
    wb_cap = math.ceil(0.9 * T_MAX / 5.0) * 5.0
    try:
        cap_hr = psychrolib.GetHumRatioFromTWetBulb(T_MAX, wb_cap, P)
        max_hr = min(max_hr, cap_hr)
    except Exception:
        pass  # keep max_hr from saturation if WB cap fails

    # Scale factor for y-axis (grains only in IP)
    y_scale = (7000.0 if use_grains else 1.0) if not USE_SI else 1.0

    # Create Plotly Figure (visuals match reference: order and styles)
    fig = go.Figure()

    # 1. Dry Bulb grid lines - darker every 5°F, lighter every 1°F
    for db in range(int(T_MIN), int(T_MAX) + 1, 1):
        hr_top = sat_lookup.get(db)
        if hr_top is None:
            continue
        hr_capped = min(hr_top, max_hr)
        # Darker lines every 5°F, lighter lines every 1°F
        if db % 5 == 0:
            fig.add_trace(go.Scatter(
                x=[db, db],
                y=[0, hr_capped * y_scale],
                mode='lines',
                line=dict(color=PRIMARY_LINE_COLOR, width=1),
                opacity=0.4,
                showlegend=False,
                hoverinfo='skip'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[db, db],
                y=[0, hr_capped * y_scale],
                mode='lines',
                line=dict(color=PRIMARY_LINE_COLOR, width=0.5),
                opacity=0.2,
                showlegend=False,
                hoverinfo='skip'
            ))

    # 2. Wet Bulb lines (every 5°F in both IP and SI) - solid royal blue, finer than DB/RH
    wb_labels = []
    WB_LINE_COLOR = "#4169E1"  # Royal blue
    if show_wb_lines:
        # 5°F increments in both modes: iterate in °F, convert to chart temp when SI
        step_degF = 5
        t_min_F = chart_t_min if not USE_SI else T_MIN * 9.0 / 5.0 + 32.0
        t_max_F = chart_t_max if not USE_SI else T_MAX * 9.0 / 5.0 + 32.0
        twb_F_start = int(t_min_F // step_degF) * step_degF
        twb_F_end = int(t_max_F) - 10
        for twb_F in range(twb_F_start, twb_F_end + 1, step_degF):
            if USE_SI:
                twb = (twb_F - 32.0) * 5.0 / 9.0  # chart temp in °C
                # Round to nearest 5 °C for clean labels (15, 20, 25, 30) like IP
                label_text = f'{int(round(twb / 5) * 5)}'
            else:
                twb = float(twb_F)
                label_text = f'{twb_F}'
            tdb_vals = []
            w_vals = []
            # WB line starts ON the saturation curve at dry-bulb = wet-bulb (100% RH), then extends right
            t_start_wb = max(twb, T_MIN)
            if t_start_wb >= T_MAX:
                continue
            ws_start = get_sat_hr(twb, sat_lookup, P)
            if ws_start is None:
                continue
            # Start at saturation curve (twb, ws) then draw to T_MAX
            for tdb in np.arange(t_start_wb, T_MAX + 0.5, 0.5):
                try:
                    w = psychrolib.GetHumRatioFromTWetBulb(tdb, twb, P)
                except Exception:
                    continue
                ws = get_sat_hr(tdb, sat_lookup, P)
                if ws is None or w is None or w < 0 or w > max_hr or w > ws:
                    continue
                tdb_vals.append(tdb)
                w_vals.append(w)
            # Ensure we include the curve start point so line begins exactly on saturation
            if len(tdb_vals) == 0 or tdb_vals[0] > twb:
                tdb_vals.insert(0, twb)
                w_vals.insert(0, ws_start)
            if len(tdb_vals) > 1:
                w_vals_scaled = [w * y_scale for w in w_vals]
                fig.add_trace(go.Scatter(
                    x=tdb_vals,
                    y=w_vals_scaled,
                    mode='lines',
                    line=dict(color=WB_LINE_COLOR, width=0.7),
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo='skip'
                ))
                # Label just left of saturation curve (small offset so label sits near the line)
                twb_int = int(round(twb))
                if twb_int >= int(T_MIN) and twb_int <= int(T_MAX):
                    sat_w = sat_lookup.get(twb_int)
                    if sat_w is not None:
                        # Offset by ~1 degree so label is close to the line
                        offset = 1.0 if (T_MAX - T_MIN) > 50 else 0.5
                        wb_labels.append({
                            'x': twb - offset,
                            'y': sat_w * y_scale,
                            'text': label_text,
                            'twb': twb
                        })

    # 3. Dew Point lines (every 5°, same grid as WB) - solid green, finer than DB/RH
    dp_labels = []
    DP_LINE_COLOR = "green"
    if show_dp_lines:
        # Match WB: 5° increments in both IP and SI, aligned to 0s and 5s
        step_deg = 5
        if USE_SI:
            t_min_C = T_MIN
            t_max_C = T_MAX
            tdp_start = int(t_min_C // step_deg) * step_deg
            # Extend to full chart range so DP lines reach top of y-axis (e.g. 40 when chart goes to 40°C)
            tdp_end = int(t_max_C)
            for tdp in range(tdp_start, tdp_end + 1, step_deg):
                if tdp < 0:
                    continue
                try:
                    w_dp = psychrolib.GetHumRatioFromTDewPoint(tdp, P)
                except Exception:
                    continue
                if w_dp <= 0 or w_dp >= max_hr:
                    continue
                t_left = find_sat_t_for_w(w_dp, T_MIN, T_MAX)
                if t_left is None or t_left >= T_MAX:
                    continue
                t_vals = []
                w_vals = []
                for t in np.arange(t_left, T_MAX + 0.5, 0.5):
                    t_vals.append(t)
                    w_vals.append(w_dp)
                if len(t_vals) == 0 or t_vals[0] > t_left:
                    t_vals.insert(0, t_left)
                    w_vals.insert(0, w_dp)
                if len(t_vals) > 1:
                    w_vals_scaled = [w * y_scale for w in w_vals]
                    fig.add_trace(go.Scatter(
                        x=t_vals,
                        y=w_vals_scaled,
                        mode='lines',
                        line=dict(color=DP_LINE_COLOR, width=0.7),
                        opacity=0.6,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    label_x = T_MAX - (T_MAX - T_MIN) * 0.02
                    label_y = w_dp * y_scale
                    # Round to nearest 5 °C for label (15, 20, 25, 30) like WB
                    dp_labels.append({
                        'x': label_x,
                        'y': label_y,
                        'text': f'{int(round(tdp / 5) * 5)}',
                        'tdp': tdp
                    })
        else:
            # IP: use °F grid aligned to 5s (same as WB)
            t_min_F = chart_t_min
            t_max_F = chart_t_max
            tdp_F_start = int(t_min_F // step_deg) * step_deg
            tdp_F_end = int(t_max_F) - 10
            for tdp_F in range(tdp_F_start, tdp_F_end + 1, step_deg):
                tdp = float(tdp_F)
                try:
                    w_dp = psychrolib.GetHumRatioFromTDewPoint(tdp, P)
                except Exception:
                    continue
                if w_dp <= 0 or w_dp >= max_hr:
                    continue
                t_left = find_sat_t_for_w(w_dp, T_MIN, T_MAX)
                if t_left is None or t_left >= T_MAX:
                    continue
                t_vals = []
                w_vals = []
                for t in np.arange(t_left, T_MAX + 0.5, 0.5):
                    t_vals.append(t)
                    w_vals.append(w_dp)
                if len(t_vals) == 0 or t_vals[0] > t_left:
                    t_vals.insert(0, t_left)
                    w_vals.insert(0, w_dp)
                if len(t_vals) > 1:
                    w_vals_scaled = [w * y_scale for w in w_vals]
                    fig.add_trace(go.Scatter(
                        x=t_vals,
                        y=w_vals_scaled,
                        mode='lines',
                        line=dict(color=DP_LINE_COLOR, width=0.7),
                        opacity=0.6,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    label_x = T_MAX - (T_MAX - T_MIN) * 0.02
                    label_y = w_dp * y_scale
                    dp_labels.append({
                        'x': label_x,
                        'y': label_y,
                        'text': f'{tdp_F}',
                        'tdp': tdp
                    })

    # 4. Enthalpy lines - red/maroon, increments of 10 Btu/lb (IP) or 20 kJ/kg (SI)
    enthalpy_labels = []
    if show_enthalpy_lines:
        try:
            h_min = psychrolib.GetMoistAirEnthalpy(T_MIN, 0.0)
        except Exception:
            h_min = 0.0
        try:
            h_max = psychrolib.GetMoistAirEnthalpy(T_MAX, max_hr)
        except Exception:
            h_max = h_min + (60000.0 if USE_SI else 60.0)
        if USE_SI:
            h_step = 20000.0  # 20 kJ/kg increments
            h_start = int(np.ceil(h_min / h_step) * h_step)
            h_end = int(np.floor(h_max / h_step) * h_step)
        else:
            h_step = 10.0  # 10 Btu/lb increments
            h_start = int(np.ceil(h_min / h_step) * h_step)
            h_end = int(np.floor(h_max / h_step) * h_step)
        for h in range(int(h_start), int(h_end) + 1, int(h_step)):
            t_vals = []
            w_vals = []
            def w_at_t(t):
                try:
                    return psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(h, t)
                except Exception:
                    return None
            # Enthalpy line: at low T it's above saturation, at high T below. Find where it crosses into valid region (left boundary).
            t_intersect = None
            t_prev_above = None  # last t where line was above saturation
            for t_test in np.arange(T_MIN, T_MAX + 0.5, 0.5):
                try:
                    w_test = psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(h, t_test)
                except Exception:
                    continue
                ws_test = get_sat_hr(t_test, sat_lookup, P)
                if ws_test is None or w_test is None:
                    continue
                if w_test > ws_test:
                    # Above saturation (invalid), keep as candidate for left boundary
                    t_prev_above = t_test
                    continue
                else:
                    # First point at or below saturation: intersection is between t_prev_above and t_test
                    if t_prev_above is not None:
                        hit = find_sat_intersection(t_prev_above, t_test, w_at_t)
                        if hit:
                            t_intersect = hit[0]
                    else:
                        # Line already valid at T_MIN, start from T_MIN
                        t_intersect = T_MIN
                    break
            if t_intersect is None:
                continue
            # Draw from intersection point to T_MAX (right side of saturation curve)
            for t in np.arange(t_intersect, T_MAX + 0.5, 0.5):
                try:
                    w = psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(h, t)
                except Exception:
                    continue
                ws = get_sat_hr(t, sat_lookup, P)
                if ws is None or w is None or w < 0 or w > max_hr or w > ws:
                    continue
                t_vals.append(t)
                w_vals.append(w)
            # Add intersection point at start if not already included
            if len(t_vals) == 0 or t_vals[0] > t_intersect:
                ws_int = get_sat_hr(t_intersect, sat_lookup, P)
                if ws_int is not None:
                    t_vals.insert(0, t_intersect)
                    w_vals.insert(0, ws_int)
            if len(t_vals) > 1:
                w_vals_scaled = [w * y_scale for w in w_vals]
                fig.add_trace(go.Scatter(
                    x=t_vals,
                    y=w_vals_scaled,
                    mode='lines',
                    line=dict(color='#8B0000', width=1),  # Dark red/maroon
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo='skip'
                ))
                # Label near left third of line
                label_idx = max(0, len(t_vals) // 3)
                if label_idx < len(t_vals) and label_idx < len(w_vals_scaled):
                    h_label = f"{int(h/1000)}" if USE_SI else f"{int(h)}"
                    enthalpy_labels.append(dict(
                        x=t_vals[label_idx],
                        y=w_vals_scaled[label_idx],
                        text=h_label,
                        h_val=h
                    ))

    # 5. Humidity Ratio lines - light grey, at 20 gr/lb or 0.002 then 0.004 increments for lb/lb/kg/kg
    hr_labels = []
    if show_hr_lines:
        if USE_SI:
            # Start at 0.002, then increment by 0.004
            hr_start = 0.002
            hr_step = 0.004
        else:
            if use_grains:
                hr_start = 20.0 / 7000.0  # Start at 20 grains/lb
                hr_step = 20.0 / 7000.0  # Then increment by 20 grains/lb
            else:
                # Start at 0.002, then increment by 0.004
                hr_start = 0.002
                hr_step = 0.004
        # First line at hr_start, then add hr_step for each subsequent line
        w = hr_start
        while w <= max_hr:
            t_left = find_sat_t_for_w(w, T_MIN, T_MAX)
            if t_left is None or t_left >= T_MAX:
                w += hr_step
                continue
            t_vals = []
            w_vals = []
            for t in np.arange(max(T_MIN, t_left), T_MAX + 0.5, 0.5):
                ws = get_sat_hr(t, sat_lookup, P)
                if ws is None or w > ws:
                    continue
                t_vals.append(t)
                w_vals.append(w)
            if t_left >= T_MIN and (not t_vals or t_vals[0] > t_left):
                t_vals.insert(0, t_left)
                w_vals.insert(0, w)
            if len(t_vals) > 1:
                w_vals_scaled = [w * y_scale for w in w_vals]
                fig.add_trace(go.Scatter(
                    x=t_vals,
                    y=w_vals_scaled,
                    mode='lines',
                    line=dict(color='lightgray', width=0.8),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                # Store HR value for Y-axis tick labels (not annotations in chart)
                hr_labels.append({
                    'w': w,
                    'y_scaled': w * y_scale
                })
            # Increment for next line
            w += hr_step

    # 6. Plot minor RH lines (10% increments) as primary guides (trim at saturation)
    rh_annotations = []
    for rh in range(10, 100, 10):
        rh_temps, rh_hr = get_rh_curve(rh, T_MIN, T_MAX, P)
        def w_at_t_rh(t):
            try:
                return psychrolib.GetHumRatioFromRelHum(t, rh / 100.0, P)
            except Exception:
                return None
        # Find intersection point where RH curve meets saturation (left boundary)
        trim_temps = []
        trim_hr = []
        t_intersect = None
        for i in range(len(rh_temps)):
            if rh_hr[i] is None:
                continue
            ws_i = get_sat_hr(rh_temps[i], sat_lookup, P)
            if ws_i is None:
                continue
            if rh_hr[i] <= ws_i:
                # Still below/at saturation, check next point
                continue
            else:
                # Crossed saturation, find exact intersection
                t_lo = rh_temps[i - 1] if i > 0 else T_MIN
                hit = find_sat_intersection(t_lo, rh_temps[i], w_at_t_rh)
                if hit:
                    t_intersect = hit[0]
                break
        if t_intersect is None:
            # No intersection found, use all points (shouldn't happen normally)
            trim_temps = list(rh_temps)
            trim_hr = list(rh_hr)
        else:
            # Keep only points from intersection to T_MAX (right side of saturation curve)
            for i in range(len(rh_temps)):
                if rh_temps[i] >= t_intersect and rh_hr[i] is not None:
                    ws_i = get_sat_hr(rh_temps[i], sat_lookup, P)
                    if ws_i is not None and rh_hr[i] <= ws_i:
                        trim_temps.append(rh_temps[i])
                        trim_hr.append(rh_hr[i])
            # Add intersection point at start if not already included
            if len(trim_temps) == 0 or trim_temps[0] > t_intersect:
                ws_int = get_sat_hr(t_intersect, sat_lookup, P)
                if ws_int is not None:
                    trim_temps.insert(0, t_intersect)
                    trim_hr.insert(0, ws_int)
        rh_hr_scaled = [min(hr, max_hr) * y_scale if hr is not None else None for hr in trim_hr]
        fig.add_trace(go.Scatter(
            x=trim_temps,
            y=rh_hr_scaled,
            mode='lines',
            name=f'{rh}% RH',
            line=dict(color=PRIMARY_LINE_COLOR, width=1),
            opacity=0.55,
            showlegend=False,
            hoverinfo='skip'
        ))
        n_pts = len(trim_temps)
        idx = min(int(n_pts * 0.75), n_pts - 1)
        while idx >= 0 and (rh_hr_scaled[idx] is None or rh_hr_scaled[idx] <= 0):
            idx -= 1
        if idx >= 0 and rh_hr_scaled[idx] is not None:
            rh_annotations.append(dict(
                x=trim_temps[idx],
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

    # 7. Plot Saturation Curve (100% RH) on top of minor lines
    sat_hr_scaled = [min(hr, max_hr) * y_scale if hr is not None else None for hr in sat_hr]
    fig.add_trace(go.Scatter(
        x=sat_temps,
        y=sat_hr_scaled,
        mode='lines',
        name='Saturation Line (100% RH)',
        line=dict(color=PRIMARY_LINE_COLOR, width=2)
    ))

    # Legend-only traces for Wet Bulb, Dew Point, Enthalpy, and HR Grid
    if show_wb_lines:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color=WB_LINE_COLOR, width=0.7),
            name='Wet Bulb',
            showlegend=True,
        ))
    if show_dp_lines:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color=DP_LINE_COLOR, width=0.7),
            name='Dew Point',
            showlegend=True,
        ))
    if show_enthalpy_lines:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='#8B0000', width=1),
            name='Enthalpy' + (" (kJ/kg)" if USE_SI else " (Btu/lb)"),
            showlegend=True,
        ))
    if show_hr_lines:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='lightgray', width=0.8),
            name='Humidity Ratio',
            showlegend=True,
        ))

    # Add Wet Bulb labels: to the left of saturation curve
    wb_annotations = []
    if show_wb_lines:
        for label in wb_labels:
            wb_annotations.append(dict(
                x=label['x'],
                y=label['y'],
                text=label['text'],
                showarrow=False,
                xref="x",
                yref="y",
                xanchor="right",
                yanchor="middle",
                font=dict(size=9, color=WB_LINE_COLOR),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
                borderpad=2,
            ))

    # Add Dew Point labels: to the right of y-axis
    dp_annotations = []
    if show_dp_lines:
        for label in dp_labels:
            dp_annotations.append(dict(
                x=label['x'],
                y=label['y'],
                text=label['text'],
                showarrow=False,
                xref="x",
                yref="y",
                xanchor="left",
                yanchor="middle",
                font=dict(size=9, color=DP_LINE_COLOR),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
                borderpad=2,
            ))

    # Add Enthalpy labels
    enthalpy_annotations = []
    if show_enthalpy_lines and len(enthalpy_labels) > 0:
        for label in enthalpy_labels:
            enthalpy_annotations.append(dict(
                x=label['x'],
                y=label['y'],
                text=label['text'],
                showarrow=False,
                xref="x",
                yref="y",
                xanchor="right",
                yanchor="middle",
                font=dict(size=9, color='#8B0000'),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
                borderpad=2,
            ))

    # Add property labels as traces (not layout annotations) so state points can be drawn on top
    def _textposition_from_anchor(xanchor, yanchor):
        ypos = "top" if yanchor == "bottom" else ("bottom" if yanchor == "top" else "middle")
        xpos = "left" if xanchor == "left" else ("right" if xanchor == "right" else "center")
        return f"{ypos} {xpos}" if ypos != "middle" else f"middle {xpos}"
    all_label_annotations = wb_annotations + dp_annotations + enthalpy_annotations + rh_annotations
    for ann in all_label_annotations:
        textpos = _textposition_from_anchor(ann.get("xanchor", "center"), ann.get("yanchor", "middle"))
        fig.add_trace(go.Scatter(
            x=[ann["x"]],
            y=[ann["y"]],
            mode="text",
            text=[ann["text"]],
            textposition=textpos,
            textfont=ann.get("font", dict(size=9, color="#333")),
            showlegend=False,
            hoverinfo="skip",
        ))

    # 8. Hover tooltip grid: build data and add trace so hovering anywhere shows DB, HR, RH, WB, DP, h
    grid_x = np.linspace(T_MIN, T_MAX, 150)
    grid_y = np.linspace(0, max_hr, 100)
    gx, gy = np.meshgrid(grid_x, grid_y)
    gx = gx.ravel()
    gy = gy.ravel()
    grid_customdata = []
    valid_x = []
    valid_y = []
    for x, y_hr in zip(gx, gy):
        props = calculate_properties_from_db_hr(x, y_hr, P, sat_lookup)
        if props is not None:
            hr_display_val = (props[KEY_HR] * 7000.0 if use_grains else props[KEY_HR]) if not USE_SI else props[KEY_HR]
            grid_customdata.append([
                props[KEY_DB],
                hr_display_val,
                props[KEY_RH],
                props[KEY_WB],
                props[KEY_DP],
                props[KEY_ENTH],
            ])
            valid_x.append(x)
            valid_y.append(y_hr * y_scale)

    cycle_arrow_annotations = []
    # 9. Plot User State Point(s) (if solved)
    if mode == "Single State Point" and results_1 is not None:
        results = results_1
        hr_value = results[KEY_HR]
        hr_display_value = (hr_value * 7000.0 if use_grains else hr_value) if not USE_SI else hr_value
        hr_unit = "gr/lb" if (not USE_SI and use_grains) else ("kg/kg" if USE_SI else "lb/lb")
        state_customdata = [[
            results[KEY_DB],
            hr_display_value,
            results[KEY_RH],
            results[KEY_WB],
            results[KEY_DP],
            results[KEY_ENTH],
        ]]
        hover_tmpl = (
            f"DB: %{{customdata[0]:.2f}} {TEMP_LABEL}<br>"
            f"HR: %{{customdata[1]:.2f}} {hr_unit}<br>"
            "RH: %{customdata[2]:.2f} %<br>"
            f"WB: %{{customdata[3]:.2f}} {TEMP_LABEL}<br>"
            f"DP: %{{customdata[4]:.2f}} {TEMP_LABEL}<br>"
            f"h: %{{customdata[5]:.2f}} {ENTHALPY_LABEL}<br>"
            "<extra></extra>"
        )
        # Visible red state point
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
            results_1[KEY_ENTH],
        ]]
        hover_tmpl_1 = (
            "Point 1<br>"
            f"DB: %{{customdata[0]:.2f}} {TEMP_LABEL}<br>"
            f"HR: %{{customdata[1]:.2f}} {hr_unit}<br>"
            "RH: %{customdata[2]:.2f} %<br>"
            f"WB: %{{customdata[3]:.2f}} {TEMP_LABEL}<br>"
            f"DP: %{{customdata[4]:.2f}} {TEMP_LABEL}<br>"
            f"h: %{{customdata[5]:.2f}} {ENTHALPY_LABEL}<br>"
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
            results_2[KEY_ENTH],
        ]]
        hover_tmpl_2 = (
            "Point 2<br>"
            f"DB: %{{customdata[0]:.2f}} {TEMP_LABEL}<br>"
            f"HR: %{{customdata[1]:.2f}} {hr_unit}<br>"
            "RH: %{customdata[2]:.2f} %<br>"
            f"WB: %{{customdata[3]:.2f}} {TEMP_LABEL}<br>"
            f"DP: %{{customdata[4]:.2f}} {TEMP_LABEL}<br>"
            f"h: %{{customdata[5]:.2f}} {ENTHALPY_LABEL}<br>"
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
        
        results = None  # Not used in process mode
    
    # Air Mixing mode: Plot Point 1, Point 2, and Mixed Air
    elif mode == "Air Mixing" and results_1 is not None and results_2 is not None:
        # Recalculate mixed air based on current unit system (don't use stored results_mix)
        airflow_1_chart = st.session_state.get("airflow_1", 0.0)
        airflow_type_1_chart = st.session_state.get("airflow_type_1", "ACFM")
        airflow_2_chart = st.session_state.get("airflow_2", 0.0)
        airflow_type_2_chart = st.session_state.get("airflow_type_2", "ACFM")
        
        # Calculate mass flows
        def _calc_mass_flow_chart(airflow_val, airflow_type, results_dict):
            if results_dict is None or KEY_DENS not in results_dict:
                return None
            if USE_SI:
                return airflow_val * results_dict[KEY_DENS] / 3600.0  # kg/s
            else:
                if airflow_type == "ACFM":
                    return airflow_val * results_dict[KEY_DENS]  # lb/min
                else:  # SCFM
                    return airflow_val * 0.075  # lb/min
        
        mass_flow_1_chart = _calc_mass_flow_chart(airflow_1_chart, airflow_type_1_chart, results_1)
        mass_flow_2_chart = _calc_mass_flow_chart(airflow_2_chart, airflow_type_2_chart, results_2)
        
        # Calculate Mixed Air state
        results_mix = None
        if mass_flow_1_chart is not None and mass_flow_2_chart is not None and (mass_flow_1_chart + mass_flow_2_chart) > 0:
            # Mass-weighted mixing
            h_mix = (mass_flow_1_chart * results_1[KEY_ENTH] + mass_flow_2_chart * results_2[KEY_ENTH]) / (mass_flow_1_chart + mass_flow_2_chart)
            w_mix = (mass_flow_1_chart * results_1[KEY_HR] + mass_flow_2_chart * results_2[KEY_HR]) / (mass_flow_1_chart + mass_flow_2_chart)
            
            # Solve for mixed air state from h_mix and w_mix
            chart_t_min_val = st.session_state.get("chart_t_min", DEFAULT_CHART_MIN_F)
            chart_t_max_val = st.session_state.get("chart_t_max", DEFAULT_CHART_MAX_F)
            t_min_solve_mix = (chart_t_min_val - 32.0) * 5.0 / 9.0 if USE_SI else chart_t_min_val
            t_max_solve_mix = (chart_t_max_val - 32.0) * 5.0 / 9.0 if USE_SI else chart_t_max_val
            try:
                results_mix = solve_state_from_enthalpy_humratio(
                    h_mix, w_mix, PRESSURE, t_min_solve_mix, t_max_solve_mix
                )
            except Exception:
                results_mix = None
        
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
            results_1[KEY_ENTH],
        ]]
        hover_tmpl_1 = (
            "Point 1<br>"
            f"DB: %{{customdata[0]:.2f}} {TEMP_LABEL}<br>"
            f"HR: %{{customdata[1]:.2f}} {hr_unit}<br>"
            "RH: %{customdata[2]:.2f} %<br>"
            f"WB: %{{customdata[3]:.2f}} {TEMP_LABEL}<br>"
            f"DP: %{{customdata[4]:.2f}} {TEMP_LABEL}<br>"
            f"h: %{{customdata[5]:.2f}} {ENTHALPY_LABEL}<br>"
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
            results_2[KEY_ENTH],
        ]]
        hover_tmpl_2 = (
            "Point 2<br>"
            f"DB: %{{customdata[0]:.2f}} {TEMP_LABEL}<br>"
            f"HR: %{{customdata[1]:.2f}} {hr_unit}<br>"
            "RH: %{customdata[2]:.2f} %<br>"
            f"WB: %{{customdata[3]:.2f}} {TEMP_LABEL}<br>"
            f"DP: %{{customdata[4]:.2f}} {TEMP_LABEL}<br>"
            f"h: %{{customdata[5]:.2f}} {ENTHALPY_LABEL}<br>"
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
        
        # Mixed Air Point (if calculated)
        if results_mix is not None:
            hr_value_mix = results_mix[KEY_HR]
            hr_display_value_mix = (hr_value_mix * 7000.0 if use_grains else hr_value_mix) if not USE_SI else hr_value_mix
            state_customdata_mix = [[
                results_mix[KEY_DB],
                hr_display_value_mix,
                results_mix[KEY_RH],
                results_mix[KEY_WB],
                results_mix[KEY_DP],
                results_mix[KEY_ENTH],
            ]]
            hover_tmpl_mix = (
                "Mixed Air<br>"
                f"DB: %{{customdata[0]:.2f}} {TEMP_LABEL}<br>"
                f"HR: %{{customdata[1]:.2f}} {hr_unit}<br>"
                "RH: %{customdata[2]:.2f} %<br>"
                f"WB: %{{customdata[3]:.2f}} {TEMP_LABEL}<br>"
                f"DP: %{{customdata[4]:.2f}} {TEMP_LABEL}<br>"
                f"h: %{{customdata[5]:.2f}} {ENTHALPY_LABEL}<br>"
                "<extra></extra>"
            )
            
            fig.add_trace(go.Scatter(
                x=[results_mix[KEY_DB]],
                y=[hr_value_mix * y_scale],
                mode='markers+text',
                name='Mixed Air',
                showlegend=False,
                text=['Mixed Air'],
                textposition="top center",
                marker=dict(size=12, color='orange', symbol='diamond'),
                customdata=state_customdata_mix,
                hovertemplate=hover_tmpl_mix,
                hoverlabel=dict(
                    bgcolor="#ffddcc",
                    font_color="black",
                ),
            ))
        
        # Draw line connecting Point 1 and Point 2 (Mixed Air should lie on this line)
        fig.add_trace(go.Scatter(
            x=[results_1[KEY_DB], results_2[KEY_DB]],
            y=[hr_value_1 * y_scale, hr_value_2 * y_scale],
            mode='lines',
            name='Mixing Line',
            showlegend=False,
            line=dict(color='gray', width=2, dash='dash'),
            hoverinfo='skip',
        ))
        
        results = None  # Not used in mixing mode

    elif mode == "Cycle Analysis":
        results_ra = st.session_state.get("cycle_ra")
        results_oa = st.session_state.get("cycle_oa")
        results_ma = st.session_state.get("cycle_ma")
        cycle_points = st.session_state.get("cycle_points", [])
        pct_oa = st.session_state.get("cycle_pct_oa", 20.0)
        if results_ma is not None:
            hr_ra = (results_ra[KEY_HR] * y_scale) if results_ra is not None else None
            hr_oa = (results_oa[KEY_HR] * y_scale) if results_oa is not None else None
            hr_ma = results_ma[KEY_HR] * y_scale
            # 100% OA: only OA visible; OA is the start of the chain. 0% OA: only RA visible; RA is the start.
            if pct_oa >= 99.5:
                pts_x = [results_oa[KEY_DB]]
                pts_y = [hr_oa]
                for pt in cycle_points:
                    pts_x.append(pt["state"][KEY_DB])
                    pts_y.append(pt["state"][KEY_HR] * y_scale)
                if len(pts_x) > 1:
                    fig.add_trace(go.Scatter(x=pts_x, y=pts_y, mode='lines', line=dict(color='#333', width=2), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(
                    x=[results_oa[KEY_DB]], y=[hr_oa],
                    mode='markers+text', text=['Outside Air'], textposition='top center',
                    marker=dict(size=12, color='green', symbol='square'), showlegend=False, hoverinfo='skip',
                ))
            elif pct_oa <= 0.5:
                pts_x = [results_ra[KEY_DB]]
                pts_y = [hr_ra]
                for pt in cycle_points:
                    pts_x.append(pt["state"][KEY_DB])
                    pts_y.append(pt["state"][KEY_HR] * y_scale)
                if len(pts_x) > 1:
                    fig.add_trace(go.Scatter(x=pts_x, y=pts_y, mode='lines', line=dict(color='#333', width=2), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(
                    x=[results_ra[KEY_DB]], y=[hr_ra],
                    mode='markers+text', text=['Return Air'], textposition='top center',
                    marker=dict(size=12, color='blue', symbol='circle'), showlegend=False, hoverinfo='skip',
                ))
            else:
                # Mixed: show RA, OA, MA and mixing lines
                fig.add_trace(go.Scatter(
                    x=[results_oa[KEY_DB], results_ma[KEY_DB]], y=[hr_oa, hr_ma],
                    mode='lines', line=dict(color='gray', width=2, dash='dash'), showlegend=False, hoverinfo='skip',
                ))
                fig.add_trace(go.Scatter(
                    x=[results_ra[KEY_DB], results_ma[KEY_DB]], y=[hr_ra, hr_ma],
                    mode='lines', line=dict(color='gray', width=2, dash='dash'), showlegend=False, hoverinfo='skip',
                ))
                pts_x = [results_ma[KEY_DB]]
                pts_y = [hr_ma]
                for pt in cycle_points:
                    pts_x.append(pt["state"][KEY_DB])
                    pts_y.append(pt["state"][KEY_HR] * y_scale)
                if len(pts_x) > 1:
                    fig.add_trace(go.Scatter(x=pts_x, y=pts_y, mode='lines', line=dict(color='#333', width=2), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(
                    x=[results_ra[KEY_DB]], y=[hr_ra],
                    mode='markers+text', text=['Return'], textposition='top center',
                    marker=dict(size=12, color='blue', symbol='circle'), showlegend=False, hoverinfo='skip',
                ))
                fig.add_trace(go.Scatter(
                    x=[results_oa[KEY_DB]], y=[hr_oa],
                    mode='markers+text', text=['Outside'], textposition='top center',
                    marker=dict(size=12, color='green', symbol='square'), showlegend=False, hoverinfo='skip',
                ))
                fig.add_trace(go.Scatter(
                    x=[results_ma[KEY_DB]], y=[hr_ma],
                    mode='markers+text', text=['Mixed Air'], textposition='top center',
                    marker=dict(size=12, color='orange', symbol='diamond'), showlegend=False, hoverinfo='skip',
                ))
            for i, pt in enumerate(cycle_points):
                fig.add_trace(go.Scatter(
                    x=[pt["state"][KEY_DB]], y=[pt["state"][KEY_HR] * y_scale],
                    mode='markers+text', text=[pt["name"]], textposition='top center',
                    marker=dict(size=10, color='purple', symbol='diamond-open'), showlegend=False, hoverinfo='skip',
                ))
            def _arrow(ax, ay, x, y):
                cycle_arrow_annotations.append(dict(
                    x=x, y=y, ax=ax, ay=ay,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                ))
            if pct_oa >= 99.5:
                pts_x = [results_oa[KEY_DB]]
                pts_y = [hr_oa]
                for pt in cycle_points:
                    pts_x.append(pt["state"][KEY_DB])
                    pts_y.append(pt["state"][KEY_HR] * y_scale)
                for i in range(len(pts_x) - 1):
                    _arrow(pts_x[i], pts_y[i], pts_x[i + 1], pts_y[i + 1])
            elif pct_oa <= 0.5:
                pts_x = [results_ra[KEY_DB]]
                pts_y = [hr_ra]
                for pt in cycle_points:
                    pts_x.append(pt["state"][KEY_DB])
                    pts_y.append(pt["state"][KEY_HR] * y_scale)
                for i in range(len(pts_x) - 1):
                    _arrow(pts_x[i], pts_y[i], pts_x[i + 1], pts_y[i + 1])
            else:
                _arrow(results_oa[KEY_DB], hr_oa, results_ma[KEY_DB], hr_ma)
                _arrow(results_ra[KEY_DB], hr_ra, results_ma[KEY_DB], hr_ma)
                for i in range(len(pts_x) - 1):
                    _arrow(pts_x[i], pts_y[i], pts_x[i + 1], pts_y[i + 1])

    # Add hover grid trace LAST (after all state points) so it's on top and receives hover everywhere
    # Critical: opacity must be > 0 for Plotly to register hover events; using 0.05 for reliability
    if len(grid_customdata) > 0:
        hr_unit = "gr/lb" if (not USE_SI and use_grains) else ("kg/kg" if USE_SI else "lb/lb")
        dynamic_hover_tmpl = (
            f"DB: %{{customdata[0]:.2f}} {TEMP_LABEL}<br>"
            f"HR: %{{customdata[1]:.2f}} {hr_unit}<br>"
            "RH: %{customdata[2]:.2f} %<br>"
            f"WB: %{{customdata[3]:.2f}} {TEMP_LABEL}<br>"
            f"DP: %{{customdata[4]:.2f}} {TEMP_LABEL}<br>"
            f"h: %{{customdata[5]:.2f}} {ENTHALPY_LABEL}<br>"
            "<extra></extra>"
        )
        # Large marker size (100) with overlapping hit areas ensures full chart coverage
        # Opacity 0.1 is faint but visible enough for Plotly/Streamlit to reliably register hover
        fig.add_trace(go.Scatter(
            x=valid_x,
            y=valid_y,
            mode='markers',
            marker=dict(size=100, opacity=0.1, color='lightgray', line=dict(width=0)),
            showlegend=False,
            hovertemplate=dynamic_hover_tmpl,
            customdata=grid_customdata,
            name='_hover_grid',
            visible=True,
            hoverlabel=dict(bgcolor="white", font_size=11),
        ))

    # Chart cosmetics (reference style)
    if USE_SI:
        yaxis_title = "Humidity Ratio (kg w / kg da)"
        yaxis_range = [0, max_hr]
    elif use_grains:
        yaxis_title = "Humidity Ratio (gr w / lb da)"
        yaxis_range = [0, max_hr * 7000.0]
    else:
        yaxis_title = "Humidity Ratio (lb w / lb da)"
        yaxis_range = [0, max_hr]

    # Prepare Y-axis tick labels from HR lines (0.002 increments)
    yaxis_tickvals = []
    yaxis_ticktext = []
    if show_hr_lines and len(hr_labels) > 0:
        hr_labels_sorted = sorted(hr_labels, key=lambda x: x['y_scaled'])
        for label in hr_labels_sorted:
            yaxis_tickvals.append(label['y_scaled'])
            if use_grains:
                yaxis_ticktext.append(f"{int(label['w'] * 7000.0)}")
            else:
                yaxis_ticktext.append(f"{label['w']:.3f}")
    
    fig.update_layout(
        xaxis_title=f"Dry Bulb Temperature ({TEMP_LABEL})",
        yaxis_title=yaxis_title,
        yaxis=dict(
            side="right",
            showgrid=False,
            zeroline=False,
            range=yaxis_range,
            tickmode='array' if len(yaxis_tickvals) > 0 else 'linear',
            tickvals=yaxis_tickvals if len(yaxis_tickvals) > 0 else None,
            ticktext=yaxis_ticktext if len(yaxis_ticktext) > 0 else None,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            range=[T_MIN, T_MAX],
        ),
        height=800,
        autosize=True,
        plot_bgcolor='white',
        hovermode="closest",
        font=dict(family="Arial", size=12, color="black"),
        margin=dict(l=10, r=10, t=30, b=10),
        annotations=cycle_arrow_annotations if mode == "Cycle Analysis" else [],  # Cycle arrows, or property labels as traces
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

    if mode == "Cycle Analysis":
        results_ra_t = st.session_state.get("cycle_ra")
        results_oa_t = st.session_state.get("cycle_oa")
        results_ma_t = st.session_state.get("cycle_ma")
        cycle_points_t = st.session_state.get("cycle_points", [])
        if results_ra_t is not None and results_oa_t is not None and results_ma_t is not None:
            rows = [
                ("Return", results_ra_t[KEY_DB], results_ra_t[KEY_WB], results_ra_t[KEY_RH], results_ra_t[KEY_ENTH], results_ra_t[KEY_DP]),
                ("Outside", results_oa_t[KEY_DB], results_oa_t[KEY_WB], results_oa_t[KEY_RH], results_oa_t[KEY_ENTH], results_oa_t[KEY_DP]),
                ("Mixed", results_ma_t[KEY_DB], results_ma_t[KEY_WB], results_ma_t[KEY_RH], results_ma_t[KEY_ENTH], results_ma_t[KEY_DP]),
            ]
            for pt in cycle_points_t:
                s = pt["state"]
                rows.append((pt["name"], s[KEY_DB], s[KEY_WB], s[KEY_RH], s[KEY_ENTH], s[KEY_DP]))
            st.subheader("Cycle Data")
            df_cycle = pd.DataFrame(rows, columns=["Point Name", "DB", "WB", "RH", "Enthalpy", "Dew Point"])
            st.dataframe(df_cycle, use_container_width=True, hide_index=True)
            if len(cycle_points_t) > 0:
                st.caption("To remove a user-added point, select it below and click Remove.")
                rm_col1, rm_col2 = st.columns([2, 1])
                with rm_col1:
                    point_names = ["-- Select point to remove --"] + [pt["name"] for pt in cycle_points_t]
                    selected_to_remove = st.selectbox(
                        "User-added point to remove",
                        options=point_names,
                        key="cycle_select_remove",
                        label_visibility="collapsed",
                    )
                with rm_col2:
                    if st.button("Remove selected point", key="cycle_remove_btn"):
                        if selected_to_remove and selected_to_remove != "-- Select point to remove --":
                            updated = [p for p in cycle_points_t if p["name"] != selected_to_remove]
                            st.session_state["cycle_points"] = updated
                            st.rerun()

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
        st.markdown("**Chart Lines**")
        st.checkbox("Wet Bulb", value=st.session_state.get("show_wb_lines", False), key="show_wb_lines")
        st.checkbox("Dew Point", value=st.session_state.get("show_dp_lines", False), key="show_dp_lines")
        st.checkbox("Enthalpy", value=st.session_state.get("show_enthalpy_lines", True), key="show_enthalpy_lines")
        st.checkbox("HR Grid", value=st.session_state.get("show_hr_lines", True), key="show_hr_lines")

# Full-width section: Process mode - Point 1 | Point 2 | Process Summary
if st.session_state.get("mode") == "Process":
    r1 = st.session_state.get("results_1")
    r2 = st.session_state.get("results_2")
    airflow_1 = st.session_state.get("airflow_1", 0.0)
    airflow_type_1 = st.session_state.get("airflow_type_1", "ACFM")
    airflow_2 = st.session_state.get("airflow_2", 0.0)
    airflow_type_2 = st.session_state.get("airflow_type_2", "ACFM")
    use_grains_hr = st.session_state.get("hr_grains_toggle", False)
    if r1 is not None and r2 is not None:
        use_si_deltas = st.session_state.get("use_si_units", False)
        def _fmt(key, val):
            if key == KEY_HR or key == KEY_DENS:
                return f"{val:.4f}"
            return f"{val:.2f}"
        
        # Calculate mass flows
        def _calc_mass_flow(airflow_val, airflow_type, results_dict):
            if results_dict is None or KEY_DENS not in results_dict:
                return None
            if use_si_deltas:
                return airflow_val * results_dict[KEY_DENS] / 3600.0  # kg/s
            else:
                if airflow_type == "ACFM":
                    return airflow_val * results_dict[KEY_DENS]  # lb/min
                else:  # SCFM
                    return airflow_val * 0.075  # lb/min
        
        mass_flow_1 = _calc_mass_flow(airflow_1, airflow_type_1, r1)
        mass_flow_2 = _calc_mass_flow(airflow_2, airflow_type_2, r2)
        
        st.success("Point 1 & Point 2 Calculated")
        col_p1, col_p2, col_summary = st.columns(3)
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
            # Display airflow as a property
            if use_si_deltas:
                st.metric("Airflow", f"{format_airflow(airflow_1)} m³/hr")
            else:
                airflow_label_1 = f"{format_airflow(airflow_1)} {airflow_type_1}" if airflow_type_1 else f"{format_airflow(airflow_1)} CFM"
                st.metric("Airflow", airflow_label_1)
            # Display mass flow (below airflow)
            if mass_flow_1 is not None:
                if use_si_deltas:
                    st.caption(f":gray[Mass Flow: {mass_flow_1:.3f} kg/s]")
                else:
                    st.caption(f":gray[Mass Flow: {mass_flow_1:.2f} lb/min]")
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
            # Display airflow as a property
            if use_si_deltas:
                st.metric("Airflow", f"{format_airflow(airflow_2)} m³/hr")
            else:
                airflow_label_2 = f"{format_airflow(airflow_2)} {airflow_type_2}" if airflow_type_2 else f"{format_airflow(airflow_2)} CFM"
                st.metric("Airflow", airflow_label_2)
            # Display mass flow (below airflow)
            if mass_flow_2 is not None:
                if use_si_deltas:
                    st.caption(f":gray[Mass Flow: {mass_flow_2:.3f} kg/s]")
                else:
                    st.caption(f":gray[Mass Flow: {mass_flow_2:.2f} lb/min]")
        with col_summary:
            st.markdown("**Process Summary (Point 1 → Point 2):**")
            delta_enthalpy = r2[KEY_ENTH] - r1[KEY_ENTH]
            delta_hr = r2[KEY_HR] - r1[KEY_HR]
            st.metric(f"Δ {KEY_ENTH}", _fmt(KEY_ENTH, delta_enthalpy))
            if not use_si_deltas and use_grains_hr:
                st.metric("Δ Humidity Ratio (gr/lb)", _fmt(KEY_HR, delta_hr * 7000.0))
            else:
                st.metric(f"Δ {KEY_HR}", _fmt(KEY_HR, delta_hr))
            
            # Thermal Energy: ΔEnthalpy * Mass Flow
            if mass_flow_1 is not None:
                if use_si_deltas:
                    # SI: enthalpy in J/kg, mass flow in kg/s, energy in W = J/s
                    # Convert to kW: W / 1000
                    thermal_energy = delta_enthalpy * mass_flow_1 / 1000.0
                    st.metric("Total Energy Transfer", f"{thermal_energy:.2f} kW")
                else:
                    # IP: enthalpy in Btu/lb, mass flow in lb/min
                    # Convert to BTU/hr: Btu/lb * lb/min * 60 min/hr = BTU/hr
                    thermal_energy = delta_enthalpy * mass_flow_1 * 60.0
                    st.metric("Total Energy Transfer", f"{thermal_energy:.0f} BTU/hr")
            
            # Condensate: if W1 > W2, calculate water removal rate
            if delta_hr < 0:  # W1 > W2 (humidity ratio decreased)
                if mass_flow_1 is not None:
                    # Water removal = mass flow * |ΔW|
                    water_removal = mass_flow_1 * abs(delta_hr)
                    if use_si_deltas:
                        # SI: mass flow in kg/s, ΔW in kg/kg, result in kg/s
                        # Convert to L/min: kg/s * 1000 g/kg * 1 L/kg (water) * 60 s/min = L/min
                        condensate_rate = water_removal * 60.0  # L/min (since 1 kg water ≈ 1 L)
                        st.metric("Condensate Rate", f"{condensate_rate:.3f} L/min")
                    else:
                        # IP: mass flow in lb/min, ΔW in lb/lb, result in lb/min
                        # Convert to GPM: lb/min / 8.34 lb/gal = gal/min
                        condensate_rate = water_removal / 8.34  # GPM
                        st.metric("Condensate Rate", f"{condensate_rate:.3f} GPM")

# Full-width section: Air Mixing mode - Point 1 | Point 2 | Mixed Air Properties
if st.session_state.get("mode") == "Air Mixing":
    r1 = st.session_state.get("results_1")
    r2 = st.session_state.get("results_2")
    airflow_1 = st.session_state.get("airflow_1", 0.0)
    airflow_type_1 = st.session_state.get("airflow_type_1", "ACFM")
    airflow_2 = st.session_state.get("airflow_2", 0.0)
    airflow_type_2 = st.session_state.get("airflow_type_2", "ACFM")
    use_grains_hr = st.session_state.get("hr_grains_toggle", False)
    if r1 is not None and r2 is not None:
        use_si_mix = st.session_state.get("use_si_units", False)
        def _fmt(key, val):
            if key == KEY_HR or key == KEY_DENS:
                return f"{val:.4f}"
            return f"{val:.2f}"
        
        # Calculate mass flows
        def _calc_mass_flow(airflow_val, airflow_type, results_dict):
            if results_dict is None or KEY_DENS not in results_dict:
                return None
            if use_si_mix:
                return airflow_val * results_dict[KEY_DENS] / 3600.0  # kg/s
            else:
                if airflow_type == "ACFM":
                    return airflow_val * results_dict[KEY_DENS]  # lb/min
                else:  # SCFM
                    return airflow_val * 0.075  # lb/min
        
        mass_flow_1 = _calc_mass_flow(airflow_1, airflow_type_1, r1)
        mass_flow_2 = _calc_mass_flow(airflow_2, airflow_type_2, r2)
        
        # Calculate Mixed Air state
        if mass_flow_1 is not None and mass_flow_2 is not None and (mass_flow_1 + mass_flow_2) > 0:
            # Mass-weighted mixing
            h_mix = (mass_flow_1 * r1[KEY_ENTH] + mass_flow_2 * r2[KEY_ENTH]) / (mass_flow_1 + mass_flow_2)
            w_mix = (mass_flow_1 * r1[KEY_HR] + mass_flow_2 * r2[KEY_HR]) / (mass_flow_1 + mass_flow_2)
            
            # Solve for mixed air state from h_mix and w_mix
            # Chart bounds in solver units (°C when SI, °F when IP)
            chart_t_min_val = st.session_state.get("chart_t_min", DEFAULT_CHART_MIN_F)
            chart_t_max_val = st.session_state.get("chart_t_max", DEFAULT_CHART_MAX_F)
            t_min_solve_mix = (chart_t_min_val - 32.0) * 5.0 / 9.0 if use_si_mix else chart_t_min_val
            t_max_solve_mix = (chart_t_max_val - 32.0) * 5.0 / 9.0 if use_si_mix else chart_t_max_val
            try:
                results_mix = solve_state_from_enthalpy_humratio(
                    h_mix, w_mix, PRESSURE, t_min_solve_mix, t_max_solve_mix
                )
                # Store for potential use (though we recalculate in chart section)
                st.session_state["results_mix"] = results_mix
            except Exception as e:
                st.error(f"Unable to solve mixed air state: {e}")
                results_mix = None
                st.session_state["results_mix"] = None
        else:
            results_mix = None
        
        if results_mix is not None:
            st.success("Point 1, Point 2 & Mixed Air Calculated")
            col_p1, col_p2, col_mix = st.columns(3)
            with col_p1:
                st.markdown("**Point 1 Properties:**")
                for key, value in r1.items():
                    if key == KEY_HR:
                        if not use_si_mix and use_grains_hr:
                            st.metric("Humidity Ratio (gr/lb)", _fmt(key, value * 7000.0))
                        else:
                            st.metric(key, _fmt(key, value))
                    else:
                        st.metric(key, _fmt(key, value))
                # Display airflow as a property
                if use_si_mix:
                    st.metric("Airflow", f"{format_airflow(airflow_1)} m³/hr")
                else:
                    airflow_label_1_mix = f"{format_airflow(airflow_1)} {airflow_type_1}" if airflow_type_1 else f"{format_airflow(airflow_1)} CFM"
                    st.metric("Airflow", airflow_label_1_mix)
                # Display mass flow (below airflow)
                if mass_flow_1 is not None:
                    if use_si_mix:
                        st.caption(f":gray[Mass Flow: {mass_flow_1:.3f} kg/s]")
                    else:
                        st.caption(f":gray[Mass Flow: {mass_flow_1:.2f} lb/min]")
            with col_p2:
                st.markdown("**Point 2 Properties:**")
                for key, value in r2.items():
                    if key == KEY_HR:
                        if not use_si_mix and use_grains_hr:
                            st.metric("Humidity Ratio (gr/lb)", _fmt(key, value * 7000.0))
                        else:
                            st.metric(key, _fmt(key, value))
                    else:
                        st.metric(key, _fmt(key, value))
                # Display airflow as a property
                if use_si_mix:
                    st.metric("Airflow", f"{format_airflow(airflow_2)} m³/hr")
                else:
                    airflow_label_2_mix = f"{format_airflow(airflow_2)} {airflow_type_2}" if airflow_type_2 else f"{format_airflow(airflow_2)} CFM"
                    st.metric("Airflow", airflow_label_2_mix)
                # Display mass flow (below airflow)
                if mass_flow_2 is not None:
                    if use_si_mix:
                        st.caption(f":gray[Mass Flow: {mass_flow_2:.3f} kg/s]")
                    else:
                        st.caption(f":gray[Mass Flow: {mass_flow_2:.2f} lb/min]")
            with col_mix:
                st.markdown("**Mixed Air Properties:**")
                for key, value in results_mix.items():
                    if key == KEY_HR:
                        if not use_si_mix and use_grains_hr:
                            st.metric("Humidity Ratio (gr/lb)", _fmt(key, value * 7000.0))
                        else:
                            st.metric(key, _fmt(key, value))
                    else:
                        st.metric(key, _fmt(key, value))
                # Display total airflow as a property
                total_airflow = airflow_1 + airflow_2
                if use_si_mix:
                    st.metric("Total Airflow", f"{format_airflow(total_airflow)} m³/hr")
                else:
                    # For mixed air, show total CFM (assuming same type)
                    airflow_type_mix = airflow_type_1 if airflow_type_1 else "CFM"
                    st.metric("Total Airflow", f"{format_airflow(total_airflow)} {airflow_type_mix}")
                # Display total mass flow (below total airflow)
                if mass_flow_1 is not None and mass_flow_2 is not None:
                    total_mass_flow = mass_flow_1 + mass_flow_2
                    if use_si_mix:
                        st.caption(f":gray[Total Mass Flow: {total_mass_flow:.3f} kg/s]")
                    else:
                        st.caption(f":gray[Total Mass Flow: {total_mass_flow:.2f} lb/min]")
            
            # Store mixed air results for chart plotting
            st.session_state["results_mix"] = results_mix
        else:
            st.warning("Mixed air calculation requires valid mass flows for both points.")
            st.session_state["results_mix"] = None
