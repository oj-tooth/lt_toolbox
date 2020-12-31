################################################################
# export_tracmass_to_nc.py
# --------------------------------------------------------------
# Description: Script to transform TRACMASS model
# output to standard NCEI_NetCDF_Trajectory format.
#
# User Input: Locations where user modification is required
# are indicated beneath NOTE statements in the script.
#
# --------------------------------------------------------------
# Date Created: 2020-11-27
#
# Author: Ollie Tooth
###############################################################

# Import packages.
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate
from tqdm import tqdm

# ---------------------------------------------------------------------------

# Stage 1:
# Opening the raw ouput files from TRACMASS.

# NOTE: change directory path to TRACMASS output data as required.
# os.chdir('OUTPUT_DIR_PATH')
os.chdir('/Users/ollietooth/Desktop/D.Phil./Tracmass/projects/NEMO/data/output/')

# Read Tracmass output_run.csv output file to pandas DataFrame
# with specified headers.

# NOTE: change the raw data file name and variable names to
# correspond with your run as required. 'FILENAME_run.csv'
df_run = pd.read_csv('ORCA1_output_run.csv',
                     names=[
                        'ntrac',        # Trajectory no.
                        'x',            # Position in zonal direction.
                        'y',            # Position in meridional direction.
                        'z',            # Position in the vertical direction.
                        'subvol',       # Transport of particle.
                        'time_s',       # Time since start of simulation (s).
                        'To_C',         # Temperature (read - degrees C) .
                        'S_psu',        # Salinity (read - g per kg).
                        'sigma0_kgm-3'  # Density (computed - kg per m3).
                        ])

# Read Tracmass output_out.csv output file to pandas DataFrame
# with specified headers.

# NOTE: change the raw data file name and variable names to
# correspond with your run as required. 'FILENAME_out.csv'
df_out = pd.read_csv('ORCA1_output_out.csv',
                     names=[
                        'ntrac',        # Trajectory no.
                        'x',            # Position in zonal direction.
                        'y',            # Position in meridional direction.
                        'z',            # Position in the vertical direction.
                        'subvol',       # Transport of particle.
                        'time_s',       # Time since start of simulation (s).
                        'To_C',         # Temperature (read - degrees C) .
                        'S_psu',        # Salinity (read - g per kg).
                        'sigma0_kgm-3'  # Density (computed - kg per m3).
                        ])

# Concantenate pandas DataFrames, df_run and df_out.
# Since indexes overlap, ignore index in concantenation.
df = pd.concat([df_run, df_out], ignore_index=True, sort=False)
df.drop_duplicates()

# ---------------------------------------------------------------------------

# Stage 2:
# Defining time and obs variables.

# Create new time column where time_s is stored in timedelta64 format.
# timedelta64 has units of nanoseconds.
df['time'] = pd.to_timedelta(df['time_s'], unit='s')

# NOTE: specify TRACMASS output time step for your simulation -
# modify unit to min/hours/days as required.
t_step = pd.Timedelta(30, unit='D').total_seconds()

# Create obs variable to store the observation no., equivalent
# to the time-level of output in the model.
df['obs'] = np.ceil((df['time_s']/t_step) + 1)
# Ensure obs variable is of in64 type.
df = df.astype({'obs': 'int64'})

# ---------------------------------------------------------------------------

# Stage 3:
# Transform output variables to numpy arrays with dimensions traj x obs.

# Transform particle positions into (traj x obs) pandas DataFrames.
X = df.pivot(index='ntrac', columns='obs', values='x')
Y = df.pivot(index='ntrac', columns='obs', values='y')
Z = df.pivot(index='ntrac', columns='obs', values='z')

# Transform subvol into (traj x obs) pandas DataFrames.
Volume = df.pivot(index='ntrac', columns='obs', values='subvol')

# Transform tracers into (traj x obs) pandas DataFrames.
# NOTE: modify the number of tracers as required.
Temp = df.pivot(index='ntrac', columns='obs', values='To_C')
Sal = df.pivot(index='ntrac', columns='obs', values='S_psu')
Sigma0 = df.pivot(index='ntrac', columns='obs', values='sigma0_kgm-3')

# Transform time and ntrac into (traj x obs) pandas DataFrames.
Time = df.pivot(index='ntrac', columns='obs', values='time')
Traj = df.pivot(index='ntrac', columns='obs', values='ntrac')

# ---------------------------------------------------------------------------

# Stage 4:
# Converting all of our pandas DataFrames to np arrays.

# Position arrays.
# The suffix _index is included to differentiate the position
# arrays of grid indexes from those of latitude, longitude and depth.
x_index = X.to_numpy()
y_index = Y.to_numpy()
z_index = Z.to_numpy()

# Transport arrays.
vol = Volume.to_numpy()

# Tracer arrays.
# NOTE: modify the number of tracers as required.
temp = Temp.to_numpy()
sal = Sal.to_numpy()
sigma0 = Sigma0.to_numpy()

# Time/ID arrays.
time = Time.to_numpy()
trajectory = Traj.to_numpy()

# ---------------------------------------------------------------------------

# Stage 5:
# Interpolating depth using deptht field from NEMO input data and z_index.

# Move to fields input data directory.
# NOTE: change directory path to lat/lon/depth files as required.
# os.chdir('FIELD_DIR_PATH')
os.chdir('/Users/ollietooth/Desktop/D.Phil./Tracmass/projects/NEMO/data/fields/')

# Set field file name containing nav_lat/nav_lon/depth data.
# NOTE: change field file name as required.
field_file = "ORCA1-N406_2000T.nc4"  # 'FIELD_FILE.nc'

# Import deptht/u/v variable from input fields to TRACMASS.
# NOTE: change the depth variable as required - deptht/u/v.
depth = xr.open_dataset(field_file).deptht

# Inserting a value for the sea surface (0 m) for use with
# interpolation indexes < 1.
depth = np.insert(depth, 0, 0)

# Storing the index values for depth.
index = np.arange(0, len(depth))

# Utilise Scipy interp1d for linear interpolation function of depth.
f_depth = interpolate.interp1d(index, depth)

# Store the dimension sizes of our output matrix, equal to z_index.
nrows, ncols = np.shape(z_index)

# Configuiring the size of our empty array for z, particle depths.
z = np.zeros([nrows, ncols])

# For loop to use interpolate the particle depth from f_depth using z_index.
for i in np.arange(0, nrows):
    # Defining z to be negative since the z-axis is traditionally
    # positive-upwards in physical oceanography.
    z[i, :] = - f_depth(z_index[i, :])

# ---------------------------------------------------------------------------

# Stage 6:
# Interpolating particle latitudes and longtidues using position indexes.

# Import nav_lat and nav_lon variables from input field_file.
lat_mdl = xr.open_dataset(field_file).nav_lat
lon_mdl = xr.open_dataset(field_file).nav_lon

# Configuiring the size of our empty array for lat and lon.

# Since all output data are stored in traj x obs,
# nrows and ncols are consistent across all arrays.
lat = np.zeros([nrows, ncols])
lon = np.zeros([nrows, ncols])

# For loop to interpolate particle poisitions in lat/lon space from
# (x_index, y_index) pairs.
# Uses tqdm package for progress bar of linear interpolation loop.
for i in tqdm(np.arange(0, nrows)):
    lat[i, :] = lat_mdl.interp(
                                x=xr.DataArray(x_index[i, :], dims="z"),
                                y=xr.DataArray(y_index[i, :], dims="z")
                                )

    lon[i, :] = lon_mdl.interp(
                                x=xr.DataArray(x_index[i, :], dims="z"),
                                y=xr.DataArray(y_index[i, :], dims="z")
                                )

# ---------------------------------------------------------------------------

# Stage 7:
# Creating a NCEI_NetCDF_Trajectory file storing our transformed
# TRACMASS output.

# Using xarray to generate a DataSet with data variables and attributes.
dataset = xr.Dataset(
    data_vars={
         "trajectory": (["traj", "obs"], trajectory),
         "time": (["traj", "obs"], time),
         "lat": (["traj", "obs"], lat),
         "lon": (["traj", "obs"], lon),
         "z": (["traj", "obs"], z),
         "temp": (["traj", "obs"], temp),
         "sal": (["traj", "obs"], sal),
         "sigma0": (["traj", "obs"], sigma0),
         },

    # NOTE: modify dataset attributes below to include important features
    # of your simulation.
    attrs={
        "ncei_template_version": "NCEI_NetCDF_Trajectory_Template_v2.0",
        "featureType": "trajectory",
        "title": "",
        "summary": "Output of TRACMASS",
        "TRACMASS_version": "",
        "Conventions": "CF-1.6/CF-1.7",
        "date_created": "YYYY-MM-DD",  # Use ISO 8601:2004 for date.
        "creator_name": "",
        "creator_email": "",
        "project": "",
        "creator_type": "person",
        "creator_institution": "",
        "product_version": "",
        "references": "TRACMASS - https://github.com/TRACMASS",
        }
)

# Specifying variable attributes according to the NCEI_NetCDF_Trajectory_Template_v2.0.
# See: https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl

# trajectory
dataset.trajectory.attrs = {
    'long_name': "Unique identifier for each particle",
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
    "positive": "upward"
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
# sigma0
dataset.sigma0.attrs = {
    'long_name': "sigma0",
    'standard_name': "sigma0",
    'units': "kg/m3"
}

# ---------------------------------------------------------------------------

# Stage 8:
# Saving our NCEI_NetCDF_Trajectory file as a netCDF file.

# Save dataset to netCDF format -
# NOTE: modify the output file path/name as required for your simulation.
# dataset.to_netcdf('OUTPUT_FILE_PATH', format="NETCDF4")
dataset.to_netcdf('/Users/ollietooth/Desktop/D.Phil./PrelimPhase/data/ORCA1-N406_TRACMASS_output_run.nc', format="NETCDF4")
