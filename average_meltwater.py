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

# Near-terminus regions: everything east (downstream) of the line p1 -> p2 and between p1/p2 in y.
# Same lines as the near-terminus velocity in velocity_inbasins.ipynb.
terminus_lines = {
    'ZI': ((450000, -1.08e6), (470000, -1.14e6)),
    '79N': ((404000, -1.05e6), (444000, -1.13e6)),
}

for glacier in glaciers:
    runoff = xr.open_dataset(f's3://gaia/cjense/data/mar/basin_meltwater_regridded_daily_{glacier}_2000_2025.zarr', storage_options=storage_options, consolidated=False)

    # Basin mean (skipped if it already exists, since mode='w-' won't overwrite)
    basin_zarr = f's3://gaia/cjense/data/mar/basin_meltwater_mean_daily_{glacier}_2000_2025.zarr'
    if not fs.exists(basin_zarr):
        ru_mean = runoff.RU.mean(dim=['x', 'y']).compute()

        # Save basin-mean runoff time series as zarr and parquet
        ru_mean_ds = ru_mean.drop_vars([c for c in ru_mean.coords if c != 'TIME']).to_dataset(name='RU')
        ru_mean_ds.to_zarr(basin_zarr, consolidated=False, storage_options=storage_options, mode='w-')

        ru_mean_df = ru_mean_ds.to_dataframe()
        ru_mean_df.to_parquet(f's3://gaia/cjense/data/mar/basin_meltwater_mean_daily_{glacier}_2000_2025.parquet', storage_options=storage_options)
        print(ru_mean_df.head())

    # Near-terminus mean
    (p1x, p1y), (p2x, p2y) = terminus_lines[glacier]
    east_of_line = ((p2x - p1x) * (runoff.y - p1y) - (p2y - p1y) * (runoff.x - p1x)) > 0
    in_region = east_of_line & (runoff.y <= p1y) & (runoff.y >= p2y) & runoff.in_basin
    ru_term = runoff.RU.where(in_region).mean(dim=['x', 'y']).compute()  # keep 0s: zero runoff is real

    ru_term_ds = ru_term.drop_vars([c for c in ru_term.coords if c != 'TIME']).to_dataset(name='RU')
    ru_term_ds.to_zarr(f's3://gaia/cjense/data/mar/terminus_meltwater_mean_daily_{glacier}_2000_2025.zarr', consolidated=False, storage_options=storage_options, mode='w')

    ru_term_df = ru_term_ds.to_dataframe()
    ru_term_df.to_parquet(f's3://gaia/cjense/data/mar/terminus_meltwater_mean_daily_{glacier}_2000_2025.parquet', storage_options=storage_options)
    print(glacier, 'near-terminus pixels:', int(in_region.sum()))
    print(ru_term_df.head())