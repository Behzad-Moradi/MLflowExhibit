from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/Energy_Demand.csv")
PROCESSED_PATH = Path("data/processed/hourly_demand.csv")
XGB_OUTPUT_PATH = Path("data/processed/xgboost_features.csv")


df = pd.read_csv(RAW_PATH)
df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
df = df[["SETTLEMENTDATE", "TOTALDEMAND"]]
df = df.set_index("SETTLEMENTDATE")
hourly_df = df.resample("1h").mean()
hourly_df.to_csv(PROCESSED_PATH)


hourly_df["lag_1"] = hourly_df["TOTALDEMAND"].shift(1)
hourly_df["lag_2"] = hourly_df["TOTALDEMAND"].shift(2)
hourly_df["lag_24"] = hourly_df["TOTALDEMAND"].shift(24)
hourly_df["lag_48"] = hourly_df["TOTALDEMAND"].shift(48)
hourly_df["lag_168"] = hourly_df["TOTALDEMAND"].shift(168)
hourly_df["rolling_mean_24"] = hourly_df["TOTALDEMAND"].shift(1).rolling(24).mean()
hourly_df["rolling_mean_168"] = hourly_df["TOTALDEMAND"].shift(1).rolling(168).mean()
hourly_df["rolling_std_24"] = hourly_df["TOTALDEMAND"].shift(1).rolling(24).std()
hourly_df["rolling_std_168"] = hourly_df["TOTALDEMAND"].shift(1).rolling(168).std()
hourly_df["hour"] = hourly_df.index.hour
hourly_df["day_of_week"] = hourly_df.index.dayofweek
hourly_df["is_weekend"] = (hourly_df.index.dayofweek >= 5).astype(int)
hourly_df["month"] = hourly_df.index.month


hourly_df = hourly_df.dropna()

hourly_df.to_csv(XGB_OUTPUT_PATH)
