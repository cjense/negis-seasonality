import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import s3fs
from functools import reduce
import os

# ─────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────
GLACIER_NAME  = "zach"           # change to glacier_B for second run
S3_BUCKET     = "s3://gaia"
RESOLUTION    = "6D"                  # "6D" or "1D" — start with 6D
TRAIN_CUTOFF  = "2022-01-01"          # everything before this is train
TEST_START    = "2022-01-01"          # everything from here is test
# CACHE_PARQUET = True                  # write flat df to S3 after extraction
RANDOM_SEED   = 42

fs = s3fs.S3FileSystem(
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    client_kwargs={"endpoint_url": os.environ["S3_ENDPOINT_URL"]},
    config_kwargs={
        "request_checksum_calculation": "when_required",
        "response_checksum_validation": "when_required",
                }
)

storage_options = {
    "client_kwargs": {"endpoint_url": os.environ["S3_ENDPOINT_URL"]},
    "config_kwargs": {
        "request_checksum_calculation": "when_required",
        "response_checksum_validation": "when_required",
    },
}

FEATURE_COLS = [
    # Spatial vars
    # "ice_elevation",
    "meltwater",
    # Non-spatial vars (broadcast)
    "t_u", "masked_mel_velocity", "non_nan_area", "ocean_EN4_TFc",
    # Lag features
    "vel_lag_1step", "vel_lag_30d", "vel_lag_60d", "vel_lag_90d",
    "vel_roll_30d_mean", "vel_roll_30d_std",
    # Time
    "season_sin", "season_cos", "year_norm", "time_days",
    # Space
    "x_norm", "y_norm",
]
TARGET_COL = "ice_velocity"

def engineer_features(df: pd.DataFrame, resolution_days: int = 6) -> pd.DataFrame:
    """
    Add lag, rolling, and time-encoding features.
    All lags are in timesteps, not days — adjust shift() values if changing resolution.

    Lag features are critical with one glacier: the model learns entirely from
    space × time variation, so velocity history at each pixel is a strong signal.
    """
    df = df.sort_values(["x", "y", "time"]).reset_index(drop=True)
    px = df.groupby(["x", "y"])

    # ── Velocity lags (in timesteps) ──
    steps_30d  = max(1, round(30  / resolution_days))
    steps_60d  = max(1, round(60  / resolution_days))
    steps_90d  = max(1, round(90  / resolution_days))

    df["vel_lag_1step"] = px["ice_velocity"].shift(1)          # 1 timestep ago
    df["vel_lag_30d"]   = px["ice_velocity"].shift(steps_30d)
    df["vel_lag_60d"]   = px["ice_velocity"].shift(steps_60d)
    df["vel_lag_90d"]   = px["ice_velocity"].shift(steps_90d)

    # Rolling mean over past ~30 days (excludes current timestep via shift first)
    df["vel_roll_30d_mean"] = (
        px["ice_velocity"]
        .transform(lambda s: s.shift(1).rolling(steps_30d, min_periods=1).mean())
    )
    df["vel_roll_30d_std"] = (
        px["ice_velocity"]
        .transform(lambda s: s.shift(1).rolling(steps_30d, min_periods=1).std())
    )

    # ── Time features ──
    t = pd.to_datetime(df["time"])

    # Cyclic seasonality — prevents Dec/Jan discontinuity
    df["season_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365.25).astype("float32")
    df["season_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365.25).astype("float32")

    # Long-term trend (0.0 = year 2000, 1.0 = year 2025)
    df["year_norm"] = ((t.dt.year - 2000) / 25).astype("float32")

    # Integer time (days since 2000-01-01) — useful as raw feature too
    df["time_days"] = (t - pd.Timestamp("2000-01-01")).dt.days.astype("int16")

    # ── Spatial position features ──
    # Normalize x/y so model can learn position-dependent patterns
    df["x_norm"] = ((df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())).astype("float32")
    df["y_norm"] = ((df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min())).astype("float32")

    return df

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Halve memory usage by downcasting float64 → float32.
    At daily resolution this takes ~51 GB → ~25 GB.
    """
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    df["x"] = df["x"].astype("int32")
    df["y"] = df["y"].astype("int32")
    return df

def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> xgb.Booster:
    # Drop rows with NaN in any feature (from lag windows at start of timeseries)
    dtrain = xgb.DMatrix(train_df[FEATURE_COLS], label=train_df[TARGET_COL])
    dtest  = xgb.DMatrix(test_df[FEATURE_COLS],  label=test_df[TARGET_COL])

    params = {
        "tree_method":      "hist",
        "device":           "cuda",       # GPU on Tillicum
        "objective":        "reg:squarederror",
        "eval_metric":      ["rmse", "mae"],
        "max_depth":        6,
        "eta":              0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,           # regularize — 140k pixels, don't overfit spatially
        "seed":             RANDOM_SEED,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=50,
        verbose_eval=25,
    )

    return model

def evaluate(model: xgb.Booster, test_df: pd.DataFrame):
    dtest = xgb.DMatrix(test_df[FEATURE_COLS])
    preds = model.predict(dtest)
    truth = test_df[TARGET_COL].values

    # Mask NaN in truth only — can't compute metrics on unknown ground truth
    valid = ~np.isnan(truth)
    preds = preds[valid]
    truth = truth[valid]

    rmse = np.sqrt(np.mean((preds - truth) ** 2))
    mae  = np.mean(np.abs(preds - truth))
    truth_var = np.sum((truth - truth.mean()) ** 2)
    r2 = 1 - np.sum((truth - preds) ** 2) / truth_var if truth_var > 0 else float("nan")
   
    # Guard against zero-variance truth
    # truth_var = np.sum((truth - truth.mean()) ** 2)
    # if truth_var == 0:
    #     print("  WARNING: truth has zero variance — R² undefined")
    #     r2 = float("nan")
    # else:
    #     r2 = 1 - np.sum((truth - preds) ** 2) / truth_var

    print(f"\n── Test metrics ──────────────────")
    print(f"  RMSE : {rmse:.4f} m/day")
    print(f"  MAE  : {mae:.4f} m/day")
    print(f"  R²   : {r2:.4f}")

    importance = model.get_score(importance_type="gain")
    importance = pd.Series(importance).sort_values(ascending=False)
    print(f"\n── Top 10 features by gain ───────")
    print(importance.head(10).to_string())

    return {"rmse": rmse, "mae": mae, "r2": r2, "feature_importance": importance}


def main():
    # df = pd.read_parquet('s3://gaia/cjense/data/testmodel/flat_6D.parquet', storage_options=storage_options)

    # ns = pd.read_parquet(
    #     f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_non_spatial.parquet",
    #     storage_options=storage_options
    # )
    # ns["time"] = pd.to_datetime(ns["time"]).dt.normalize()
    # df["time"] = pd.to_datetime(df["time"])

    # df = df.merge(ns, on="time", how="left")

    # print("Merged spatial and non-spatial dataframes.")
    # print(df.head())

    # resolution_days = int(RESOLUTION.replace("D", ""))
    # df = engineer_features(df, resolution_days=resolution_days)
    # df = optimize_dtypes(df)
    # print("Datatypes optimized")
    
    # cache_path = f"{S3_BUCKET}/cjense/data/testmodel/flat_2{RESOLUTION}.parquet"
    # print(f"Writing flat parquet to {cache_path} ...")
    # df.to_parquet(cache_path, storage_options=storage_options, index=False)
    # print("Cached.")
    
    df = pd.read_parquet(f"{S3_BUCKET}/cjense/data/testmodel/flat_2{RESOLUTION}.parquet", storage_options=storage_options)
    
    train = df[df["time"] <  TRAIN_CUTOFF].copy()
    test  = df[df["time"] >= TEST_START].copy()
    print(f"Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    
    del df
    
    model = train_model(train, test)
    
    metrics = evaluate(model, test)
    
    model_path = f"/gpfs/scrubbed/jensencc/negis_seasonality/{GLACIER_NAME}_xgb_{RESOLUTION}.json"
    model.save_model(model_path)
    fs.put(model_path, f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_xgb_{RESOLUTION}.json")
    print(f"\nModel saved to S3.")
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = main()