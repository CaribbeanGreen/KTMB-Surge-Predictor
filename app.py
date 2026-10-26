import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import shap
import holidays
import matplotlib.pyplot as plt
from datetime import datetime
import os
import requests

# ---------------------------------------------------------------------------
# The model is trained on exactly these 5 features. hour/day_of_week/
# is_holiday are computed locally. temperature/rainfall come from a live
# weather API -- the model was trained on REAL historical weather for the
# same location (see preprocess_data.py), so this live value is comparing
# like with like, unlike the old version's hardcoded 30.0/0.0 fallback.
# ---------------------------------------------------------------------------
FEATURE_ORDER = ["hour", "day_of_week", "is_holiday", "temperature", "rainfall"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# NOTE: move this to an environment variable / st.secrets before sharing or
# committing this file anywhere -- a hardcoded API key in source code is a
# real credential leak risk (anyone with the file can use your quota/account).
OWM_API_KEY = os.environ.get("OWM_API_KEY", "4b16a66a0561a93b6682705be78483a0")
KL_CITY_ID = 1734705  # Kuala Lumpur -- matches the station used in preprocess_data.py

@st.cache_data(ttl=300)
def get_current_busiest_lane():
    """
    Returns the busiest Origin → Destination pair for the current hour
    based on historical average ridership.
    """

    route_df = load_route_data()

    if route_df is None:
        return None

    current_hour = datetime.now().hour

    hourly_df = route_df[
        route_df["hour"] == current_hour
    ]

    if hourly_df.empty:
        return None

    busiest = (
        hourly_df
        .groupby(["origin", "destination"], as_index=False)["ridership"]
        .mean()
        .sort_values("ridership", ascending=False)
        .iloc[0]
    )

    return busiest

@st.cache_data(ttl=600)
def get_live_weather():
    """Live weather for the same location used to build historical training
    data. If this call fails, we return None rather than a silent fallback
    constant -- see build_live_feature_vector() for why that matters."""
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?id={KL_CITY_ID}&appid={OWM_API_KEY}&units=metric"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        temperature = data["main"]["temp"]
        rainfall = data.get("rain", {}).get("1h", 0.0)
        return temperature, rainfall
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_holiday_status():
    today = datetime.now().date()
    malaysia = holidays.Malaysia(years=today.year)

    if today in malaysia:
        return 1, malaysia.get(today)
    return 0, None


FRIENDLY_NAMES = {
    "hour": "Time of day",
    "day_of_week": "Day of week",
    "is_holiday": "Public holiday",
    "temperature": "Temperature",
    "rainfall": "Rainfall",
}


def format_feature_detail(feature, value):
    if feature == "hour":
        return f"{int(value):02d}:00"
    if feature == "day_of_week":
        return DAY_NAMES[int(value)]
    if feature == "is_holiday":
        return "Yes" if value else "No"
    if feature == "temperature":
        return f"{value:.1f} °C"
    if feature == "rainfall":
        return f"{value:.1f} mm"
    return str(value)


def render_friendly_shap(shap_values, live_data, prediction):
    """Plain-language replacement for shap.plots.waterfall. Same underlying
    SHAP contributions, but: real feature names instead of raw column
    names, sorted by impact size, value labels placed outside each bar so
    they never collide, and an on-chart legend/caption so it's readable
    without a separate explanation."""
    exp = shap_values[0]
    base = float(exp.base_values)

    items = []
    for feature, value, contrib in zip(FEATURE_ORDER, live_data.iloc[0], exp.values):
        label = FRIENDLY_NAMES.get(feature, feature)
        detail = format_feature_detail(feature, value)
        items.append((label, detail, float(contrib)))

    items.sort(key=lambda x: abs(x[2]), reverse=True)

    labels = [f"{lab} ({det})" for lab, det, _ in items]
    contribs = [c for _, _, c in items]
    colors = ["#e91e63" if c >= 0 else "#2196f3" for c in contribs]

    fig, ax = plt.subplots(figsize=(9, 5.8))
    y_pos = list(range(len(items)))
    bars = ax.barh(y_pos, contribs, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()  # biggest impact on top
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Passengers added or removed vs. the baseline")
    ax.set_title(
        f"Baseline average: {base:.0f} pax   →   This forecast: {prediction} pax",
        fontsize=13, pad=12,
    )

    span = max(abs(min(contribs, default=0)), abs(max(contribs, default=0)), 1)
    ax.set_xlim(-span * 1.35, span * 1.35)

    for bar, c in zip(bars, contribs):
        offset = span * 0.03
        x = bar.get_width() + (offset if c >= 0 else -offset)
        ha = "left" if c >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{c:+.0f}",
                 va="center", ha=ha, fontsize=10, color="#333333")

    # On-chart legend so pink/blue don't need to be explained elsewhere
    increase_patch = plt.Rectangle((0, 0), 1, 1, fc="#e91e63")
    decrease_patch = plt.Rectangle((0, 0), 1, 1, fc="#2196f3")
    ax.legend(
        [increase_patch, decrease_patch],
        ["Pushes forecast up", "Pushes forecast down"],
        loc="lower right", fontsize=9, frameon=False,
    )

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.text(
        0.02, 0.02,
        f"How to read this: {base:.0f} pax is what the model predicts for a "
        f"typical hour. Each bar shows how much that specific condition "
        f"(time, day, weather...) shifts this hour's forecast up or down. "
        f"Add them all together and you get {prediction} pax.",
        fontsize=9, color="#555555", ha="left",
    )
    return fig


@st.cache_data(ttl=3600)
def load_route_data():
    """Loads the RAW per-route ticketing data (not the aggregated training
    CSV) so the Route Explorer can look up a specific origin->destination
    pair. Place ets_2026.parquet in the same folder as this app, or update
    the path below to match where it actually lives."""
    try:
        df = pd.read_parquet("ets_2026.parquet")
        df["date"] = pd.to_datetime(df["date"])
        df["hour"] = pd.to_datetime(df["time"], format="%H:%M").dt.hour
        return df
    except FileNotFoundError:
        return None


@st.cache_resource
def load_engine():
    model = joblib.load("ktmb_surge_model.pkl")
    explainer = shap.TreeExplainer(model)
    return model, explainer


def render_live_clock(start_dt):
    """Ticks a live HH:MM:SS clock in the browser via JS, seeded from the
    server's timestamp at prediction time. This is separate from the model
    -- the model only ever sees whole hours (see the feature table below),
    this is purely so the page doesn't look frozen at the moment it loaded."""
    iso = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
    components.html(
        f"""
        <div style="font-family: monospace; font-size: 1.6rem; color:#fafafa;">
          <span id='clock'>{start_dt.strftime('%H:%M:%S')}</span>
        </div>
        <script>
          let current = new Date("{iso}");
          function tick() {{
            current.setSeconds(current.getSeconds() + 1);
            const h = String(current.getHours()).padStart(2, '0');
            const m = String(current.getMinutes()).padStart(2, '0');
            const s = String(current.getSeconds()).padStart(2, '0');
            document.getElementById('clock').innerText = h + ':' + m + ':' + s;
          }}
          setInterval(tick, 1000);
        </script>
        """,
        height=45,
    )


def build_display_table(live_data, ctx):
    """Human-readable version of the feature vector for display. The model
    itself still gets the raw encoded values (0-6, 0/1, etc.) -- this is
    only for what the person sees on screen."""
    temperature = live_data["temperature"].iloc[0]
    rainfall = live_data["rainfall"].iloc[0]

    rows = [
        ("Time", ctx["now"].strftime("%H:%M")),
        ("Day", ctx["day_name"]),
        ("Public holiday", ctx["holiday_name"] if ctx["holiday_flag"] else "No"),
        ("Temperature", f"{temperature:.1f} °C"),
        ("Rainfall (past hour)", f"{rainfall:.1f} mm"),
    ]
    return pd.DataFrame(rows, columns=["Feature", "Value"]).set_index("Feature")


def build_live_feature_vector():
    now = datetime.now()
    holiday_flag, holiday_name = get_holiday_status()
    weather = get_live_weather()

    weather_ok = weather is not None
    temperature, rainfall = weather if weather_ok else (None, None)

    row = {
        "hour": now.hour,
        "day_of_week": now.weekday(),
        "is_holiday": holiday_flag,
        "temperature": temperature,
        "rainfall": rainfall,
    }

    live_data = pd.DataFrame([row], columns=FEATURE_ORDER)

    context = {
        "now": now,
        "day_name": DAY_NAMES[now.weekday()],
        "holiday_flag": holiday_flag,
        "holiday_name": holiday_name,
        "weather_ok": weather_ok,
    }

    return live_data, context


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
st.set_page_config(page_title="KTMB Predictor", layout="wide")
st.title("KTMB ETS Passenger Surge Predictor")

rf_model, explainer = load_engine()

tab1, tab2 = st.tabs(["Transparency Eval (SHAP XAI)", "Passanger Surge Analysis"])

# TAB 1: Transparency Evaluation
with tab1:
    st.sidebar.header("Live Feature Stream")

    with st.spinner("Building live feature vector..."):
        live_data, ctx = build_live_feature_vector()

    if not ctx["weather_ok"]:
        st.error(
            "Live weather fetch failed, so no prediction was made. "
            "The model needs a real temperature/rainfall reading -- "
            "silently substituting a guessed value would reproduce the "
            "exact bug this app used to have."
        )
        st.stop()

    prediction = int(rf_model.predict(live_data)[0])
    shap_values = explainer(live_data)

    col1, col2 = st.columns([1, 2])

    with col1:
       st.metric("Forecasted Surge", f"{prediction} Pax")

    lane = get_current_busiest_lane()

    if lane is not None:
        st.metric(
            "Current Busiest Lane",
            f"{lane['origin']} → {lane['destination']}",
            f"{lane['ridership']:.1f} avg passengers"
        )

        render_live_clock(ctx["now"])

        if ctx["holiday_flag"]:
            st.success(f"Holiday: {ctx['holiday_name']}")
        else:
            st.info("Today is not a public holiday.")

        st.write("### Live Feature Vector")
        st.dataframe(build_display_table(live_data, ctx), use_container_width=True)

        if st.button("Log Prediction"):
            new_log = pd.DataFrame([{
                "Date": ctx["now"].strftime("%Y-%m-%d %H:%M"),
                "DayOfWeek": ctx["now"].weekday(),
                "DayName": ctx["day_name"],
                "Predicted": prediction,
                "Actual": None,
            }])

            file_exists = os.path.exists("prediction_logs.csv")
            new_log.to_csv(
                "prediction_logs.csv",
                mode="a",
                header=not file_exists,
                index=False,
            )
            st.success("Prediction logged successfully.")

    with col2:
        st.subheader("Feature Contribution Analysis")
        fig = render_friendly_shap(shap_values, live_data, prediction)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# TAB 2: Route Explorer + Performance Evaluation
with tab2:
    st.header("Route Explorer")
    st.caption(
        "This looks up real historical ridership for a specific route. "
        "It's a lookup of actual past data, not an AI prediction -- the "
        "forecast model itself only knows network-wide totals, not "
        "individual routes."
    )

    route_df = load_route_data()

    if route_df is None:
        st.warning(
            "Couldn't find `ets_2026.parquet` in the app's folder. "
            "Place it next to app.py to enable route-level lookup."
        )
    else:
        origins = sorted(route_df["origin"].unique())
        col_o, col_d = st.columns(2)
        with col_o:
            origin = st.selectbox("Origin", origins)
        dests = sorted(route_df.loc[route_df["origin"] == origin, "destination"].unique())
        with col_d:
            destination = st.selectbox("Destination", dests)

        filtered = route_df[
            (route_df["origin"] == origin) & (route_df["destination"] == destination)
        ]

        if filtered.empty:
            st.info("No historical records for this route.")
        else:
            busiest_hour = int(filtered.groupby("hour")["ridership"].mean().idxmax())

            col1, col2, col3 = st.columns(3)
            col1.metric("Historical Records", len(filtered))
            col2.metric("Avg Ridership / Trip", f"{filtered['ridership'].mean():.1f}")
            col3.metric("Busiest Hour", f"{busiest_hour:02d}:00")

            st.write(f"### Average Ridership by Hour — {origin} → {destination}")
            hourly = filtered.groupby("hour")["ridership"].mean().reindex(range(24), fill_value=0)

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(hourly.index, hourly.values, color="#e91e63")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Average ridership")
            ax.set_xticks(range(0, 24, 2))
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.write("### Matching Records")
            search = st.text_input("Filter records (optional)", placeholder="e.g. 2026-07")
            display_records = filtered.sort_values("date", ascending=False)
            if search:
                display_records = display_records[
                    display_records["date"].astype(str).str.contains(search)
                    | display_records["time"].astype(str).str.contains(search)
                ]
            st.dataframe(
                display_records[["date", "time", "origin", "destination", "ridership"]],
                use_container_width=True,
                hide_index=True,
            )
    try:
        logs_df = pd.read_csv("prediction_logs.csv")
        completed = logs_df.dropna(subset=["Actual"])

        if not completed.empty:
            mae = (abs(completed["Predicted"] - completed["Actual"])).mean()

            col_a, col_b = st.columns(2)
            col_a.metric("Model MAE (Avg Error)", f"± {int(mae)} Pax")
            col_b.metric("Evaluated Trips", len(completed))

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(completed["Date"], completed["Predicted"], label="AI Prediction", marker="o")
            ax2.plot(completed["Date"], completed["Actual"], label="Ground Truth", marker="x", linestyle="--")
            ax2.legend()
            st.pyplot(fig2)

            st.write("### Raw Validation Logs")
            st.dataframe(logs_df)
        else:
            st.warning("No completed logs yet. Add 'Actual' values to prediction_logs.csv.")
    except FileNotFoundError:
        st.warning("No tracking logs found yet. Run a prediction in Tab 1 and save it.")