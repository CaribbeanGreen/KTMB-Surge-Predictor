import joblib

model = joblib.load("ktmb_surge_model.pkl")

print(model.feature_names_in_)