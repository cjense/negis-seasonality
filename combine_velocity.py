import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import os
import s3fs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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

# Download netCDFs locally
fs.get('gaia/cjense/data/velocity/ZI_79N_velseries_monthly.nc', '/tmp/ZI_79N_velseries_monthly.nc')
fs.get('gaia/cjense/data/velocity/ZI_79N_velseries_612day.nc', '/tmp/ZI_79N_velseries_612day.nc')
fs.get('gaia/cjense/data/velocity/ZI_79N_velseries_optical_individual_78.95N_ZI.nc', '/tmp/ZI_79N_velseries_optical_individual_78.95N_ZI.nc')
fs.get('gaia/cjense/data/velocity/ZI_79N_velseries_optical_individual_E79.40N_79N.nc', '/tmp/ZI_79N_velseries_optical_individual_E79.40N_79N.nc')

monthly = xr.open_dataset('/tmp/ZI_79N_velseries_monthly.nc')
print(monthly)
six12 = xr.open_dataset('/tmp/ZI_79N_velseries_612day.nc')
print(six12)
individual_ZI = xr.open_dataset('/tmp/ZI_79N_velseries_optical_individual_78.95N_ZI.nc')
print(individual_ZI)
individual_79N = xr.open_dataset('/tmp/ZI_79N_velseries_optical_individual_E79.40N_79N.nc')
print(individual_79N)


# Extract bounds for each dataset
bounds = {
    'monthly': {
        'x': (monthly.x.values.min(), monthly.x.values.max()),
        'y': (monthly.y.values.min(), monthly.y.values.max()),
        'time': (monthly.time.values.min(), monthly.time.values.max()),
    },
    'six12day': {
        'x': (six12.x.values.min(), six12.x.values.max()),
        'y': (six12.y.values.min(), six12.y.values.max()),
        'time': (six12.time.values.min(), six12.time.values.max()),
    },
    'individual_ZI': {
        'x': (individual_ZI.x.values.min(), individual_ZI.x.values.max()),
        'y': (individual_ZI.y.values.min(), individual_ZI.y.values.max()),
        'time': (individual_ZI.time.values.min(), individual_ZI.time.values.max()),
    },
    'individual_79N': {
        'x': (individual_79N.x.values.min(), individual_79N.x.values.max()),
        'y': (individual_79N.y.values.min(), individual_79N.y.values.max()),
        'time': (individual_79N.time.values.min(), individual_79N.time.values.max()),
    }
}

# Print bounds for reference
for name, b in bounds.items():
    print(f"{name}:")
    print(f"  x: {b['x'][0]:.0f} to {b['x'][1]:.0f} (width: {b['x'][1]-b['x'][0]:.0f})")
    print(f"  y: {b['y'][0]:.0f} to {b['y'][1]:.0f} (height: {b['y'][0]-b['y'][1]:.0f})")
    print(f"  time: {pd.Timestamp(b['time'][0])} to {pd.Timestamp(b['time'][1])}")
    print()
    
    
# Regrid the individual optical series onto the monthly/six12day spatial grid.
# Interpolate over x/y ONLY -- interp_like(monthly) also interpolates time onto
# monthly's 3 timesteps, which throws away the whole 2000-2014 record.
target_grid = dict(x=monthly.x, y=monthly.y)
print("Regridding individual_79N to match monthly/six12day grid...")
individual_79N_regridded = individual_79N.interp(**target_grid, method='nearest')
print(f"  Regridded individual_79N shape: {individual_79N_regridded.VelocityMap.shape}")

print("Regridding individual_ZI to match monthly/six12day grid...")
individual_ZI_regridded = individual_ZI.interp(**target_grid, method='nearest')
print(f"  Regridded individual_ZI shape: {individual_ZI_regridded.VelocityMap.shape}")

# The two individual series share timestamps but cover different (slightly overlapping)
# areas, so merge them spatially rather than stacking them in time (which would create
# duplicate timestamps). 79N takes priority where they overlap; ZI fills the rest.
individual = (
    individual_79N_regridded.drop_vars(['time1', 'time2'])
    .combine_first(individual_ZI_regridded.drop_vars(['time1', 'time2']))
)
individual['VelocityMap'] = individual.VelocityMap.astype('float32')
print(f"  Merged individual shape: {individual.VelocityMap.shape}")

# Concatenate all four datasets
print("\nConcatenating all 4 datasets in time order...")

# Remove overlap: keep only monthly data before six12day starts
monthly_filtered = monthly.where(monthly.time < six12.time.values[0], drop=True)

print(f"  Individual (79N + ZI, regridded): {len(individual.time)} timesteps")
print(f"  Monthly (after removing overlap): {len(monthly_filtered.time)} timesteps")
print(f"  Six12day: {len(six12.time)} timesteps")

# Concatenate in chronological order
# First sort by time to ensure proper ordering
datasets_to_concat = [
    ds.drop_vars(['time1', 'time2'], errors='ignore')
    for ds in [individual, monthly_filtered, six12]
]
combined = xr.concat(datasets_to_concat, dim='time', join='exact')
combined = combined.sortby('time')

print(f"\n✓ Combined dataset created")
print(f"  Shape: {combined.VelocityMap.shape}")
print(f"  Time range: {pd.Timestamp(combined.time.values.min())} to {pd.Timestamp(combined.time.values.max())}")
print(f"  Total timesteps: {len(combined.time)}")
print(f"  Grid: {combined.sizes['y']} × {combined.sizes['x']}")
nan_total = combined.VelocityMap.isnull().sum().values
total = combined.VelocityMap.size
print(f"  Valid data: {100 * (1 - nan_total / total):.1f}%")

# Drop extra time coordinates (time1, time2) if they exist
coords_to_drop = [c for c in combined.coords if c.startswith('time') and c != 'time']
if coords_to_drop:
    combined_clean = combined.drop_vars(coords_to_drop)
    print(f"  Dropped coordinates: {coords_to_drop}")
else:
    combined_clean = combined

print(combined_clean)

combined_clean.to_zarr('s3://gaia/cjense/data/velocity/ZI_79N_velseries_combined_2000_2025.zarr', consolidated=False, storage_options=storage_options, mode='w')