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
    runoff = xr.open_dataset(f's3://gaia/cjense/data/mar/basin_meltwater_regridded_daily_{glacier}_2000_2025.zarr', storage_options=storage_options, consolidated=False)

    ru_mean = runoff.RU.mean(dim=['x', 'y']).compute()

    # Save basin-mean runoff time series as zarr and parquet
    ru_mean_ds = ru_mean.drop_vars([c for c in ru_mean.coords if c != 'TIME']).to_dataset(name='RU')
    ru_mean_ds.to_zarr(f's3://gaia/cjense/data/mar/basin_meltwater_mean_daily_{glacier}_2000_2025.zarr', consolidated=False, storage_options=storage_options, mode='w-')

    ru_mean_df = ru_mean_ds.to_dataframe()
    ru_mean_df.to_parquet(f's3://gaia/cjense/data/mar/basin_meltwater_mean_daily_{glacier}_2000_2025.parquet', storage_options=storage_options)
    ru_mean_df.head()