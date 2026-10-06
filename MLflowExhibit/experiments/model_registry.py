import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_name = "EnergyDemandXGBoost"
run_id = "1acf55104bbe454985ccc0536d4d33a6"

model_rui = f"runs:/{run_id}/XGBoost" 

result = mlflow.register_model(
    model_rui,
    model_name,
)

print("Registered version:", result.version)