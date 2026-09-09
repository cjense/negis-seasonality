#!/usr/bin/env python3

"""
Glacier Ice Velocity ML Pipeline — XGBoost on Tillicum/Kopah
=============================================================
Summary of decisions from design session:

DATA LAYOUT
-----------
- One glacier at a time (run script twice for two glaciers separately)
- Spatial variables (ice_velocity, meltwater, ice_elevation): stay as Zarr on S3
  → Do NOT bulk-convert Zarr to Parquet — only convert at the flat extraction stage
- Non-spatial variables (melange, terminus_change, OTF): Parquet/CSV, tiny, keep in RAM
- Non-spatial vars repeat per pixel per timestep (XGBoost requires it) but are
  broadcast lazily at merge time — never stored redundantly on disk

ACTUAL DATA SIZE (140,151 pixels, one glacier)
-----------------------------------------------
  6-day resolution  (~1,522 timesteps): ~8-9 GB flat in RAM  ← recommended starting point
  Daily resolution  (~9,132 timesteps): ~25 GB flat in RAM   ← fine for Tillicum, use float32

PIPELINE
--------
1. [CPU SLURM job] Extract Zarr → flat Parquet, engineer features, cache to S3
2. [GPU SLURM job] Load flat Parquet → XGBoost DMatrix → train → save model

TRAIN/TEST SPLIT
----------------
- Temporal split only (never random) — last 3 years (2022-2025) as test
- Spatial random split would leak future information into training

"""

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
CACHE_PARQUET = True                  # write flat df to S3 after extraction
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

# ─────────────────────────────────────────────
# STEP 1 — LOAD & FLATTEN
# ─────────────────────────────────────────────
def load_spatial_zarr() -> xr.Dataset:
    """
    Load spatial vars from Zarr — chunked along time so resample stays lazy.
    Chunk size ~200 timesteps × full spatial extent keeps each chunk ~500 MB.
    """
    paths = {
        "ice_velocity": "s3://gaia/cjense/data/velocity/basin_velocity_ZI.zarr",
        "meltwater":    "s3://gaia/cjensen/data/mar/meltwater_regridded_daily_ZI.zarr",
    }

    arrays = {}
    for var, path in paths.items():
        ds = xr.open_dataset(
            path,
            storage_options=storage_options,
            consolidated=False,
            engine="zarr",
            chunks={"time": 200},   # ← chunk at open time, not after
        )
        arrays[var] = ds[list(ds.data_vars)[0]]

    # Align chunks across variables before merging
    ds = xr.Dataset(arrays)
    return ds


def flatten_to_dataframe(ds: xr.Dataset, batch_size: int = 200) -> pd.DataFrame:
    """
    Materialize in time-batches to avoid loading the full array at once.
    Each batch is ~500 MB for 6D resolution.
    """
    times = ds.time.values
    batches = []

    for i in range(0, len(times), batch_size):
        batch_times = times[i : i + batch_size]
        chunk = ds.sel(time=batch_times).compute()          # pulls ~500 MB
        df_chunk = chunk.to_dataframe().reset_index()
        df_chunk = df_chunk.dropna(subset=["ice_velocity"])
        batches.append(df_chunk)
        del chunk
        print(f"  Flattened timesteps {i}–{i+len(batch_times)-1} / {len(times)}")

    df = pd.concat(batches, ignore_index=True)
    return df


def broadcast_non_spatial(df: pd.DataFrame, glacier_name: str, fs: s3fs.S3FileSystem, storage_options) -> pd.DataFrame:
    """
    Load non-spatial variables (melange, terminus_change, OTF) and broadcast to
    every pixel row by merging on 'time'. Values repeat per pixel per timestep
    but this is required by XGBoost and is cheap at this data size.
    """
    ns = pd.read_parquet(
        f"{S3_BUCKET}/cjense/data/testmodel/{glacier_name}_non_spatial.parquet",
        storage_options=storage_options
    )
    ns["time"] = pd.to_datetime(ns["time"])
    df["time"] = pd.to_datetime(df["time"])

    df = df.merge(ns, on="time", how="left")
    return df


# ─────────────────────────────────────────────
# STEP 2 — DTYPE OPTIMIZATION
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# STEP 4 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
def temporal_split(df: pd.DataFrame):
    """
    Split by time only — never randomly.
    Random splits leak future observations into training via lag features
    at nearby pixels and timesteps.
    """
    train = df[df["time"] <  TRAIN_CUTOFF].copy()
    test  = df[df["time"] >= TEST_START].copy()
    print(f"Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    return train, test


# ─────────────────────────────────────────────
# STEP 5 — XGBOOST TRAINING
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # Spatial vars
    # "ice_elevation",
    "meltwater",
    # Non-spatial vars (broadcast)
    "melange_area", "melange_velocity", "terminus_area_change", "OTF",
    # Lag features
    "vel_lag_1step", "vel_lag_30d", "vel_lag_60d", "vel_lag_90d",
    "vel_roll_30d_mean", "vel_roll_30d_std",
    # Time
    "season_sin", "season_cos", "year_norm", "time_days",
    # Space
    "x_norm", "y_norm",
]
TARGET_COL = "ice_velocity"


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> xgb.Booster:
    # Drop rows with NaN in any feature (from lag windows at start of timeseries)
    train_clean = train_df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    test_clean  = test_df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    dtrain = xgb.DMatrix(train_clean[FEATURE_COLS], label=train_clean[TARGET_COL])
    dtest  = xgb.DMatrix(test_clean[FEATURE_COLS],  label=test_clean[TARGET_COL])

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


# ─────────────────────────────────────────────
# STEP 6 — EVALUATE
# ─────────────────────────────────────────────
def evaluate(model: xgb.Booster, test_df: pd.DataFrame):
    test_clean = test_df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    dtest = xgb.DMatrix(test_clean[FEATURE_COLS])
    preds = model.predict(dtest)
    truth = test_clean[TARGET_COL].values

    rmse = np.sqrt(np.mean((preds - truth) ** 2))
    mae  = np.mean(np.abs(preds - truth))
    r2   = 1 - np.sum((truth - preds) ** 2) / np.sum((truth - truth.mean()) ** 2)

    print(f"\n── Test metrics ──────────────────")
    print(f"  RMSE : {rmse:.4f} m/day")
    print(f"  MAE  : {mae:.4f} m/day")
    print(f"  R²   : {r2:.4f}")

    # Feature importance
    importance = model.get_score(importance_type="gain")
    importance = pd.Series(importance).sort_values(ascending=False)
    print(f"\n── Top 10 features by gain ───────")
    print(importance.head(10).to_string())

    return {"rmse": rmse, "mae": mae, "r2": r2, "feature_importance": importance}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    fs = s3fs.S3FileSystem()

    # ── Check for cached flat parquet first ──
    cache_path = f"{S3_BUCKET}/cjense/data/testmodel/flat_2{RESOLUTION}.parquet"
    try:
        print(f"Looking for cached flat parquet at {cache_path} ...")
        # Use pyarrow chunked read to avoid one big allocation
        df = pd.read_parquet(
            cache_path,
            storage_options={"anon": False},
            engine="pyarrow",
        )
        # Or for very large files, read in chunks via pyarrow directly:
        # import pyarrow.parquet as pq
        # pq.read_table(cache_path, filesystem=fs).to_pandas()
        df["time"] = pd.to_datetime(df["time"])
        print(f"Loaded from cache: {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    except Exception:
        print("No cache found — building from Zarr sources ...")

        # 1. Load spatial vars from Zarr (lazy)
        ds = load_spatial_zarr()

        # 2. Resample while still lazy (dask-backed, no data pulled yet)
        # if RESOLUTION == "6D":
        #     ds = ds.resample(time="6D").mean()   # lazy — just builds the graph
        #     # Re-chunk after resample since it resets chunk boundaries
        #     ds = ds.chunk({"time": 200})

        # # 3. Flatten in batches (this is where data is actually downloaded)
        # print("Flattening xarray → dataframe in batches ...")
        # df = flatten_to_dataframe(ds, batch_size=50)
        
        df = pd.read_parquet('s3://gaia/cjense/data/testmodel/flat_6D.parquet', storage_options=storage_options)
        
        # 4. Broadcast non-spatial vars
        df = broadcast_non_spatial(df, GLACIER_NAME, fs, storage_options)

        # 5. Feature engineering
        resolution_days = int(RESOLUTION.replace("D", ""))
        df = engineer_features(df, resolution_days=resolution_days)
        df = optimize_dtypes(df)   # re-optimize after new float columns added

        # 6. Cache to S3
        if CACHE_PARQUET:
            print(f"Writing flat parquet to {cache_path} ...")
            df.to_parquet(cache_path, storage_options=storage_options, index=False)
            print("Cached.")

    # ── Train / test split ──
    train_df, test_df = temporal_split(df)

    # ── Free memory before training ──
    del df

    # ── Train ──
    print("\nTraining XGBoost ...")
    model = train_model(train_df, test_df)

    # ── Evaluate ──
    metrics = evaluate(model, test_df)

    # ── Save model ──
    model_path = f"/gpfs/scrubbed/jensencc/negis_seasonality/{GLACIER_NAME}_xgb_{RESOLUTION}.json"
    model.save_model(model_path)
    fs.put(model_path, f"{S3_BUCKET}/cjense/data/testmodel/{GLACIER_NAME}_xgb_{RESOLUTION}.json")
    print(f"\nModel saved to S3.")

    return model, metrics


if __name__ == "__main__":
    model, metrics = main()