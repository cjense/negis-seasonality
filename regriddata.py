import geopandas as gpd
import s3fs
import xarray as xr
import rioxarray
from rasterio.features import geometry_mask
import os
import dask

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

def clip_meltwater(dataset, basin, glacier):
    # Clip glaciers to basin
    geom = basin.to_crs(dataset.rio.crs).geometry.union_all()

    clipped = dataset.rio.clip(
        [geom],
        crs=dataset.rio.crs,
        drop=True,
    )

    # Drop netCDF encoding inherited from the source file
    clipped.encoding = {}

    # Mask values in basin for easier parquet conversion later
    # Zarr doesn't care about the extra NaN values (b/c it stores data in a rectangle)
    # but they will take up unnecessary space in the parquet.
    in_basin = geometry_mask(
        [geom],
        out_shape=(clipped.sizes["y"], clipped.sizes["x"]),
        transform=clipped.rio.transform(),
        invert=True,
    )
    clipped = clipped.to_dataset(name="RU")
    clipped["in_basin"] = (("y", "x"), in_basin)

    with dask.config.set(scheduler="synchronous"):
        clipped.to_zarr(
            f"s3://gaia/cjense/data/mar/basin_meltwater_regridded_daily_{glacier}_2000_2025.zarr",
            storage_options=storage_options,
            mode="w",
            consolidated=False,
        )
    
    return clipped

def main():
    meltwater = xr.open_dataset('/gpfs/scrubbed/jensencc/negis-seasonality/meltwater2000_2025_daily.nc')
    # TODO: save meltwater data as Zarr file in Kopah

    # Open velocity maps for regridding
    fs.get('s3://gaia/cjense/data/velocity/ZI_79N_velseries_612day.nc', '/tmp/ZI_79N_velseries_612day.nc')
    velocity = xr.open_dataset('/tmp/ZI_79N_velseries_612day.nc')
    v = velocity.VelocityMap.sel(time='2025-12-09T12:00:00.000000000').mean(dim='band')
    
    # Remove from memory since only one is needed
    del velocity

    meltwater = meltwater.RU.sel(SECTOR=1).assign_coords(
        x=meltwater.x * 1000,
        y=meltwater.y * 1000,
    )
    meltwater["x"].attrs["units"] = "m"
    meltwater["y"].attrs["units"] = "m"
    meltwater = meltwater.rio.write_crs("EPSG:3413")
    meltwater = meltwater.chunk({"TIME": 100, "x": -1, "y": -1})

    meltwater_linear = meltwater.regrid.linear(v, time_dim="TIME")
    meltwater_linear = meltwater_linear.rio.write_crs("EPSG:3413")  # regrid drops the CRS

    # Get basins for clipping data
    basins = gpd.read_file("/gpfs/scrubbed/jensencc/negis-seasonality/basins/Greenland_Basins_PS_v1.4.2.shp")
    # TODO save basin data as Zarr file in Kopah
    zibasin = basins.loc[basins['NAME']=='ZACHARIAE_ISSTROM']
    basin79n = basins.loc[basins['NAME']=='NIOGHALVFJERDSFJORDEN']
    
    zi_clipped = clip_meltwater(meltwater_linear, zibasin, 'ZI')
    clipped_79N = clip_meltwater(meltwater_linear, basin79n, '79N')
    
    return zi_clipped, clipped_79N

if __name__ == "__main__":
    zi_clipped, clipped_79N = main()