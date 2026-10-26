import pandas as pd
import holidays
import requests
import os

print("Current directory:", os.getcwd())
print("Loading KTMB dataset...")

# --- Load the raw parquet -------------------------------------------------
df = pd.read_parquet("C:/Users/aidil/Documents/PythonProjects/ets_2026.parquet")
print("Rows loaded (per route/hour):", len(df))

df["hour"] = pd.to_datetime(df["time"], format="%H:%M").dt.hour
df["day_of_week"] = df["date"].dt.dayofweek

malaysia_holidays = holidays.Malaysia(years=df["date"].dt.year.unique().tolist())
df["is_holiday"] = df["date"].dt.date.apply(lambda d: 1 if d in malaysia_holidays else 0)

# ==========================================================================
# REAL historical weather via Open-Meteo's free archive API (no key needed).
#
# This is a network-wide "surge" model, and KTMB ETS routes span a long
# west-coast corridor (Gemas <-> Padang Besar), so no single weather station
# is perfectly representative. Kuala Lumpur is used here as a practical
# proxy for the corridor's general climate -- good enough to capture
# "is it a rainy/hot day" patterns, not station-exact conditions. If you
# want to get fancier later, you could fetch a few stations along the
# corridor and average them, but start with one and see if it helps at all.
#
# NOTE: the archive API only has data for dates that have already happened.
# Any date in your parquet that's still in the future (relative to when you
# run this script) will come back as NaN -- that's expected, not a bug.
# ==========================================================================
KL_LAT, KL_LON = 3.139, 101.6869

start_date = df["date"].min().strftime("%Y-%m-%d")
end_date = df["date"].max().strftime("%Y-%m-%d")

print(f"Fetching historical weather for {start_date} to {end_date}...")

weather_url = (
    "https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={KL_LAT}&longitude={KL_LON}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=temperature_2m,precipitation"
    "&timezone=Asia%2FKuala_Lumpur"
)

resp = requests.get(weather_url, timeout=30)
resp.raise_for_status()
weather_json = resp.json()["hourly"]

weather_df = pd.DataFrame({
    "datetime": pd.to_datetime(weather_json["time"]),
    "temperature": weather_json["temperature_2m"],
    "rainfall": weather_json["precipitation"],
})
weather_df["date"] = weather_df["datetime"].dt.date
weather_df["hour"] = weather_df["datetime"].dt.hour

df["date_only"] = df["date"].dt.date
df = df.merge(
    weather_df[["date", "hour", "temperature", "rainfall"]],
    left_on=["date_only", "hour"],
    right_on=["date", "hour"],
    how="left",
    suffixes=("", "_weather"),
)

missing_weather = df["temperature"].isna().sum()
if missing_weather:
    print(f"WARNING: {missing_weather} rows have no matching historical weather "
          f"(likely future-dated rows or an API gap). These will need to be "
          f"dropped or imputed before training -- do not fill with a constant, "
          f"that recreates the original bug.")

# ==========================================================================
# Aggregate to TOTAL NETWORK RIDERSHIP PER HOUR
# Weather is already the same for every route in a given date+hour, so
# 'first' is correct (not an average-losing-information issue).
# ==========================================================================
training_df = (
    df.groupby(["date_only", "hour"], as_index=False)
    .agg(
        day_of_week=("day_of_week", "first"),
        is_holiday=("is_holiday", "first"),
        temperature=("temperature", "first"),
        rainfall=("rainfall", "first"),
        passenger_surge_count=("ridership", "sum"),
    )
    .rename(columns={"date_only": "date"})
)

print(training_df.head())
print("Rows after aggregation:", len(training_df))
print("Target stats:\n", training_df["passenger_surge_count"].describe())

training_df.to_csv(
    "C:/Users/aidil/Documents/PythonProjects/historical_ktmb_data.csv", index=False
)

print("Done! historical_ktmb_data.csv created.")