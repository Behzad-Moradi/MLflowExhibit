import mlflow
import mlflow.xgboost
import pandas as pd

df = pd.read_csv(
    "data/processed/xgboost_features.csv",
    parse_dates=["SETTLEMENTDATE"],
)

df = df.set_index("SETTLEMENTDATE")

TARGET = "TOTALDEMAND"

X = df.drop(
    columns=[TARGET]
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

run_id = "1acf55104bbe454985ccc0536d4d33a6"

model_uri = f"runs:/{run_id}/XGBoost"
model_alias = "models:/EnergyDemandXGBoost@champion"

loaded_model = mlflow.xgboost.load_model(model_alias)

predictions = loaded_model.predict(X)

print(predictions)

