import xarray as xr
import s3fs
import os
import pandas as pd
import dask

storage_options = {
    "client_kwargs": {"endpoint_url": "https://s3.kopah.uw.edu"},
    "config_kwargs": {
        "request_checksum_calculation": "when_required",
        "response_checksum_validation": "when_required",
    },
}

def make_tabular_meltwater_data(resolution, glacier):
    meltwater = xr.open_dataset(f"s3://gaia/cjense/data/mar/basin_meltwater_regridded_daily_{glacier}_2000_2025.zarr", consolidated=False, storage_options=storage_options)

    meltwaterds = meltwater.chunk(chunks={"TIME": 100, "x": -1, "y": -1})
    meltwaterds = meltwaterds[["RU", "in_basin"]].drop_vars(['SECTOR', 'spatial_ref', 'crs', 'name', 'time', 'time1', 'time2'])
    meltwaterds = meltwaterds.resample(TIME=resolution).mean().chunk({"TIME": 10})
    meltwater = meltwaterds.to_dask_dataframe()
    meltwater = meltwater.reset_index(drop=True)
    meltwater = meltwater.rename(columns={"TIME":'time'})

    # Save to parquet
    with dask.config.set(scheduler="threads", num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 1))):
        meltwater = meltwater[meltwater.in_basin].drop(columns="in_basin")
        meltwater.to_parquet(f"s3://gaia/cjense/data/mar/basin_meltwater_regridded_daily_{glacier}_2000_2025_{resolution}.parquet", storage_options=storage_options)
    print(f"Saved meltwater parquet for {glacier}")
    
    return meltwater

def main():
    melt_79N = make_tabular_meltwater_data('MS', '79N')
    melt_ZI = make_tabular_meltwater_data('MS', 'ZI')

if __name__ == "__main__":
    main()