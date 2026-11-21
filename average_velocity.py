import pandas as pd
import xarray as xr
import os
import s3fs

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

glaciers = ['ZI', '79N']

for glacier in glaciers:
    vel = xr.open_dataset(f's3://gaia/cjense/data/velocity/basin_velocity_{glacier}.zarr', storage_options=storage_options, consolidated=False)

    vel_mean = vel.VelocityMap.mean(dim=['x', 'y']).compute()

    # Save basin-mean velocity time series as zarr and parquet
    vel_mean_ds = vel_mean.drop_vars([c for c in vel_mean.coords if c not in ('time', 'band')]).to_dataset(name='VelocityMap')
    vel_mean_ds.to_zarr(f's3://gaia/cjense/data/velocity/basin_velocity_mean_{glacier}.zarr', consolidated=False, storage_options=storage_options, mode='w-')

    vel_mean_df = vel_mean_ds.VelocityMap.to_pandas()  # rows: time, columns: vx, vy, vv
    vel_mean_df.to_parquet(f's3://gaia/cjense/data/velocity/basin_velocity_mean_{glacier}.parquet', storage_options=storage_options)
    print(vel_mean_df.head())