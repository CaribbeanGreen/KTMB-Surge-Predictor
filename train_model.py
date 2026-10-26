# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

print("Current working directory:", os.getcwd())
print("1) Loading aggregated KTMB dataset...")

df = pd.read_csv("historical_ktmb_data.csv", parse_dates=["date"])
print("Rows loaded:", len(df))

# ==========================================================================
# Feature / Target Separation
#
# temperature and rainfall are now REAL historical values (from Open-Meteo,
# see preprocess_data.py) instead of fabricated constants, so they're back
# in as legitimate features. population_density / train_frequency /
# historical-ridership-as-a-feature are still gone -- no real historical
# source exists for those, so faking them would just reintroduce noise.
# ==========================================================================
FEATURE_ORDER = ["hour", "day_of_week", "is_holiday", "temperature", "rainfall"]

before_drop = len(df)
df = df.dropna(subset=FEATURE_ORDER + ["passenger_surge_count"])
if len(df) < before_drop:
    print(f"Dropped {before_drop - len(df)} rows with missing weather "
          f"(e.g. future-dated rows the archive API had no data for).")

X = df[FEATURE_ORDER]
y = df["passenger_surge_count"]

# ==========================================================================
# Time-based split
#
# This is time-series data (one row per date+hour). A random 80/20 split
# would let the model "peek" at hours from dates surrounding its test
# examples. Splitting chronologically -- train on earlier dates, test on
# the most recent slice -- gives a more honest read of real-world accuracy.
# ==========================================================================
df_sorted = df.sort_values("date")
split_idx = int(len(df_sorted) * 0.8)

train_idx = df_sorted.index[:split_idx]
test_idx = df_sorted.index[split_idx:]

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

print(f"2) Training on {len(X_train)} rows, testing on {len(X_test)} rows "
      f"(chronological split)...")

rf_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)

preds = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"3) Test MAE: {mae:.1f} pax  |  Test target mean: {y_test.mean():.1f} pax")

print("4) Saving model to ktmb_surge_model.pkl ...")
joblib.dump(rf_model, "ktmb_surge_model.pkl")

print("Success! 'ktmb_surge_model.pkl' has been created.")
print("Model trained with features:", list(rf_model.feature_names_in_))
print("Feature importances:", dict(zip(rf_model.feature_names_in_, rf_model.feature_importances_)))