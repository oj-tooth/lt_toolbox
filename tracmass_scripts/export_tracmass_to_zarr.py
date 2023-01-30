################################################################
# export_tracmass_to_zarr.py
# --------------------------------------------------------------
# Description: Script to transform TRACMASS Lagrangian particle
# tracking output to standard NCEI_NetCDF_Trajectory format.
#
# User Input: Locations where user modification is required
# are indicated beneath NOTE statements in the script.
#
# --------------------------------------------------------------
# Date Created: 2023-01-02
#
# Author: Ollie Tooth
###############################################################

# Import packages.
import zarr
import numpy as np
import polars as pl
import pandas as pd
import xarray as xr
from tqdm import tqdm
from scipy import interpolate

# ---------------------------------------------------------------------
# Stage 0:
# User Input: Directories, Filenames and Offsets.

# -- Defining TRACMASS output data directories --
# ---------------------------------------------------------------------
# NOTE: Modify the following directory paths and filenames as required:
# Directory to TRACMASS _ini.csv (uncompressed) file.
iniDir = '/home/ocean_shared_data1/TRACMASS/ORCA025/Sim03/FinT/ini/'
# Name of TRACMASS _ini.csv (uncompressed) file.
iniFile = 'ORCA025_Sim03_East_1976_1981_ini.csv'

# Directory to TRACMASS _run.csv file (uncompressed).
runDir = '/home/ocean_shared_data1/TRACMASS/ORCA025/Sim03/FinT/raw/'
# Name of TRACMASS _run.csv (uncompressed) file.
runFile = 'ORCA025_Sim03_East_Testing_FinT_1976_1981_run.csv'
# ---------------------------------------------------------------------
# -- Defining .zarr data store output directory and filename --
# ---------------------------------------------------------------------
# NOTE: Modify directory path to where we would like output .zarr
# data store to be located:
outDir = '/home/ocean_shared_data1/TRACMASS/ORCA025/Sim03/FinT/zarr/'
# NOTE: Modify to store trajectories released at each seeding step in
# a single .zarr data store. Default is one .zarr store per step.
# A single .zarr store is only permitted when the maximum number of
# observations is consistent across all seeding steps.
multistore = True
# NOTE: Modify the output file name prefix as required for your
# simulation. The final output data store will take the form:
#
# {outFile_YYYY_MM_DD.zarr}
#
outFile = "ORCA025-GJM189_"
# ---------------------------------------------------------------------
# -- Defining column names and types --
# ---------------------------------------------------------------------
# NOTE: Modify the following list and dictionaries as required:
# List of variables (columns) in both _ini.csv and _run.csv files output
# from TRACMASS.
col_names = ['id', 'x', 'y', 'z', 'vol', 't', 'boxface', 'temp', 'sal', 'sigma0', 'mld']
# Dictionary of dtypes for each variable (column) in both the _ini.csv
# and _run.csv files output by TRACMASS.
col_dtypes = {'id': np.int64,
              'x': np.float64,
              'y': np.float64,
              'z': np.float64,
              'vol': np.float64,
              't': np.float64,
              'boxface': np.int64,
              'temp': np.float64,
              'sal': np.float64,
              'sigma0': np.float64,
              'mld': np.float64
              }
# ---------------------------------------------------------------------
# -- Define offsets for particle ID and Time (s) --
# ---------------------------------------------------------------------
# Use this to rebase all particle IDs and Times against a single date
# marking the start of an experiment composed of multiple simulations.
# For example, consider a 10-yr experiment starting in 1970-01-01
# consisting of two 5-yr TRACMASS simulations. To rebase the second
# experiment's particle IDs and times, we could do the following:
#
# date_ini_str = '1970-01-01'
# id_offset = id_max_1970_1974_experiment
# t_offset = (np.datetime64('1975-01-01') - np.datetime64('1970-01-01')) / np.timedelta64(1, 's')
#
# NOTE: Modify the following variables as required:
# Defining the initial date of this simulation (not the overall experiment).
# When using the multifile = False option, date_ini_str YYYY is used for
# the output file name following the user specified file prefix.
date_ini_str = '1976-01-01'
id_offset = 0
t_offset = 0
# ---------------------------------------------------------------------
# -- Define model grid data directories --
# ---------------------------------------------------------------------
# Defining path and file containing model grid lat/lon/depth arrays.
# NOTE: change directory path and field file name as required.
gridDir = '/home/ocean_shared_data1/DRAKKAR/ORCA025.L75-GJM189-S/GRID/'
gridFile = "ORCA025.L75-GJM189_mesh_mask_merged_dep3d.nc"

# Import depth (1D) variable of numerical model grid.
# NOTE: change the depth variable as required - e.g. deptht.
depth = xr.open_dataset(gridDir+gridFile).nav_lev

# Import longitude and latitude (2D) variables from input gridFile.
# NOTE: change the lon/lat variable as required.
lat_mdl = xr.open_dataset(gridDir+gridFile).nav_lat
lon_mdl = xr.open_dataset(gridDir+gridFile).nav_lon
# ---------------------------------------------------------------------
# -- Linear interpolation of particle depth from model coordinates --
# ---------------------------------------------------------------------
# Inserting a value for the sea surface (0 m) for use with
# interpolation indexes < 1.
depth = np.insert(depth, 0, 0)

# Storing the index values for depth.
index = np.arange(0, len(depth))

# Utilise Scipy interp1d for linear interpolation function of depth.
f_depth = interpolate.interp1d(index, depth)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Stage 1:
# Opening the raw ouput files from TRACMASS.

# Read Lagrangian particle inflow positions and properties
# files stored in _ini.csv file.
df_ini = pd.read_csv(iniDir+iniFile, names=col_names, dtype=col_dtypes)

# Defining time-levels associated with each particle seeding step.
tlevels = np.unique(df_ini.t)
# Defining total number of seeding steps in _ini/run.csv.
nsteps = len(tlevels)
# Defining minimum particle ID for seeding step 1.
id_min = 0

# ------------- Header to Script -------------
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('==== Exporting TRACMASS Output from csv to zarr ====')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-- Description --')
print('Script to transform TRACMASS Lagrangian particle')
print('tracking output to standard NCEI_NetCDF_Trajectory,')
print('format storing trajectories released at each seeding')
print('step in a zarr data store.')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-- Info --')
print('Simulation Start Date:', np.datetime64(date_ini_str))
print('Total No. Seeding Steps:', nsteps)
print('Total No. Trajectories:', len(np.unique(df_ini.id)))
print('Multistore Option:', multistore)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('In Progress: Extracting and exporting trajectories')
# ------------- Header to Script -------------

# Extract Lagrangian particle positions and properties stored in _run.csv file
# for each seeding step, post-process and store in .zarr data store.
for n in tqdm(range(nsteps)):
    # Defining maximum particle ID within this seeding step:
    id_max = np.max(df_ini[df_ini.t == tlevels[n]].id)
    # Use Polars to extract Lagrangian particle trajectories released in
    # current seeding step. Implement parallel query using scan_csv()
    # to extract particle trajectories by seeding step:
    ds_scan = (
               pl.scan_csv(runDir+runFile, has_header=False)
               .filter((pl.col("column_1") <= id_max) & (pl.col("column_1") > id_min))
               .collect(streaming=True)
              )

    # Redefining id_min to previous id_max for next seeding step:
    id_min = id_max

    # Transform polars lazy DataFrame to eager pandas DataFrame:
    df_run = ds_scan.to_pandas()
    # Reassign column header with col_names list:
    df_run.columns = col_names
    # Reassign column dtypes with col_dtypes dictionary:
    df_run.astype(col_dtypes)

    # Apply particle ID and time offsets:
    df_run.id = df_run.id + id_offset
    df_run.t = df_run.t + t_offset

    # Determine date of seeding step:
    date = np.datetime64(date_ini_str) + np.timedelta64(1, 's') * tlevels[n]
    # Export date of seeding step to string in YYYY_MM_DD format.
    date_str = pd.to_datetime(date).strftime('%Y_%m_%d')

    # ---------------------------------------------------------------------------
    # Stage 2:
    # Defining and allocating obs variables.

    # Create new column 'obs' to store the observation number associated with
    # each particle position/properties:
    df_run['obs'] = np.zeros(len(df_run.id))

    # Defining a utility function to allocate the observation numbers as an
    # array (0, N) where N is the total number of observations associated
    # with each particle:
    def allocate_obs(x):
        # For the DataFrame subset x, allocate observation numbers (0, N).
        x['obs'] = np.arange(len(x.id))
        return x

    # Apply allocate_obs() to each particle stored in DataFrame.
    df = df_run.groupby('id').apply(allocate_obs)

    # ---------------------------------------------------------------------------
    # Stage 3:
    # Interpolating particle depth from model vertical coordinate, z.

    # Reassign z as interpolated particle depth from model vertical
    # coordinate.
    df['z'] = f_depth(df['z'])

    # ---------------------------------------------------------------------------
    # Stage 4:
    # Interpolating particle latitudes and longtidues using position indexes.

    # Transform particle positions in model coordinates to latitudes
    # and longitudes, storing values in temporary DataArrays.
    temp_lat = lat_mdl.interp(x=xr.DataArray(df['x'].values, dims="z"), y=xr.DataArray(df['y'].values, dims="z"))
    temp_lon = lon_mdl.interp(x=xr.DataArray(df['x'].values, dims="z"), y=xr.DataArray(df['y'].values, dims="z"))

    # Reassign x, y in DataFrame to particle longitudes and latitudes.
    df['x'] = temp_lat.values
    df['y'] = temp_lon.values

    # ---------------------------------------------------------------------------
    # Stage 5:
    # Transform output variables to numpy arrays with dimensions (traj x obs).

    # Transform particle positions into (traj x obs) pandas DataFrames.
    X = df.pivot(index='id', columns='obs', values='x')
    Y = df.pivot(index='id', columns='obs', values='y')
    Z = df.pivot(index='id', columns='obs', values='z')

    # Transform tracers into (traj x obs) pandas DataFrames.
    # NOTE: modify the number of tracers as required.
    Temp = df.pivot(index='id', columns='obs', values='temp')
    Sal = df.pivot(index='id', columns='obs', values='sal')

    # Transform time into (traj x obs) pandas DataFrames.
    Time = df.pivot(index='id', columns='obs', values='t')

    # Transform IDs and volume transport into (traj) pandas DataSeries.
    Traj = df.pivot(index='id', columns='obs', values='id')
    Volume = df.pivot(index='id', columns='obs', values='vol')

    # ---------------------------------------------------------------------------
    # Stage 6:
    # Converting all of our pandas DataFrames to ndarrays.

    # ID & Transport arrays.
    trajectory = Traj.to_numpy()
    vol = Volume.to_numpy()

    # Time array.
    time = Time.to_numpy()

    # Position arrays.
    # The suffix _index is included to differentiate the position
    # arrays of grid indexes from those of latitude, longitude and depth.
    lon = X.to_numpy()
    lat = Y.to_numpy()
    z = Z.to_numpy()

    # Tracer arrays.
    # NOTE: modify the number of tracers as required.
    temp = Temp.to_numpy()
    sal = Sal.to_numpy()

    # ---------------------------------------------------------------------------
    # Stage 7:
    # Creating a NCEI_NetCDF_Trajectory file storing our transformed
    # TRACMASS output.

    # Using xarray to generate a DataSet with data variables and attributes.
    # For trajectory ID (trajectory) and volume transport (vol), data is
    # stored as 1-dimensional arrays. The obs dimension is not needed
    # since the trajectory ID and volume transport of each particle is
    # conserved throughout our simulations.
    dataset = xr.Dataset(
        data_vars={
            "trajectory": (["traj"], trajectory[:, 0]),
            "vol": (["traj"], vol[:, 0]),
            "time": (["traj", "obs"], time),
            "lat": (["traj", "obs"], lat),
            "lon": (["traj", "obs"], lon),
            "z": (["traj", "obs"], z),
            "temp": (["traj", "obs"], temp),
            "sal": (["traj", "obs"], sal),
            },

        # NOTE: modify dataset attributes below to include important features
        # of your simulation.
        attrs={
               "ncei_template_version": "NCEI_NetCDF_Trajectory_Template_v2.0",
               "featureType": "trajectory",
               "title": "ORCA025-GJM189 OSNAP East FinT Lagrangian Overturning " + date_str,
               "summary": "Simulation 1 ORCA025-GJM189 - seeding particles northwards across OSNAP East",
               "TRACMASS_version": "v7 (2020-10-28)",
               "Conventions": "CF-1.6/CF-1.7",
               "date_created": "2023-01-02",  # Use ISO 8601:2004 for date.
               "creator_name": "Ollie Tooth",
               "creator_email": "oliver.tooth@seh.ox.ac.uk",
               "project": "ORCA025_Sim03",
               "creator_type": "person",
               "creator_institution": "University of Oxford",
               "product_version": "1.0",
               "references": "TRACMASS - https://github.com/TRACMASS",
              }
    )

    # Specifying variable attributes according to the NCEI_NetCDF_Trajectory_Template_v2.0.
    # See: https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl

    # trajectory
    dataset.trajectory.attrs = {
                               'long_name': "water parcel ID",
                               'cf_role': "trajectory_id"
                               }
    # time
    dataset.time.attrs = {
                         'long_name': "time since begining of the simulation",
                         'standard_name': "time",
                         'unit': 'nanoseconds',
                         'calendar': "none"
                         }

    # lat
    dataset.lat.attrs = {
                        'long_name': "latitude",
                        'standard_name': "latitude",
                        'units': "degrees_north"
                        }
    # lon
    dataset.lon.attrs = {
                        'long_name': "longitude",
                        'standard_name': "longitude",
                        'units': "degrees_east"
                        }
    # z
    dataset.z.attrs = {
                      'long_name': "depth",
                      'standard_name': "depth",
                      'units': "meters",
                      }
    # vol
    dataset.vol.attrs = {
                        'long_name': "volume transport",
                        'standard_name': "volume",
                        'units': "meters^3"
                        }
    # NOTE: modify tracer attributes below as required.
    # temp
    dataset.temp.attrs = {
                         'long_name': "temperature",
                         'standard_name': "temperature",
                         'units': "C"
                         }
    # sal
    dataset.sal.attrs = {
                        'long_name': "salinity",
                        'standard_name': "salinity",
                        'units': "PSU"
                        }

    # ---------------------------------------------------------------------------
    # Stage 8:
    # Saving our NCEI_NetCDF_Trajectory file as a zarr data store.

    # Defining standard Blosc compressor with 3 levels.
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)

    # Defining an encoding dictionary to compress all variables in DataSet.
    enc = {x: {"compressor": compressor} for x in dataset}

    # -----------------------
    # (1) Singlestore Option: [A single .zarr data store for all seeding steps]
    # -----------------------
    # Uses YYYY from initial date string of simulation in output filename.
    if multistore is False:
        if n == 0:
            # Save initial seeding step DataSet to zarr data store format:
            dataset.to_zarr(outDir+outFile+ date_ini_str[:4] + ".zarr", encoding=enc)
        else:
            # Append subsequent DataSets to existing zarr data store:
            dataset.to_zarr(outDir+outFile+ date_ini_str[:4]  + ".zarr", append_dim="traj")

    # -----------------------
    # (2) Multistore Option: [A single .zarr data store per seeding step]
    # -----------------------
    elif multistore is True:
        # Save DataSet to zarr data store format:
        dataset.to_zarr(outDir+outFile+ date_str + ".zarr", encoding=enc)

# ------------- Footer to Script -------------
print('Completed: Exported TRACMASS .csv to .zarr')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# ------------- Footer to Script -------------
