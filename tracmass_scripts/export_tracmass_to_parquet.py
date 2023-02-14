################################################################
# export_tracmass_to_parquet.py
# --------------------------------------------------------------
# Description: Script to transform TRACMASS Lagrangian particle
# tracking output run.csv files into individual parquet files.
#
# User Input: Locations where user modification is required
# are indicated beneath NOTE statements in the script.
#
# --------------------------------------------------------------
# Date Created: 2023-02-07
#
# Author: Ollie Tooth
###############################################################
# Import packages.
import numpy as np
import polars as pl
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------
# Stage 0:
# User Input: Directories, Filenames and Offsets.

# -- Defining TRACMASS output data directories --
# ---------------------------------------------------------------------
# NOTE: Modify the following directory paths and filenames as required:
# Directory to TRACMASS _ini.csv (uncompressed) file.
iniDir = '/gws/nopw/j04/oxford_es/otooth/Tracmass/ORCA1-N006/tracking/eSPNA/'
# Name of TRACMASS _ini.csv (uncompressed) file.
iniFile = 'ORCA1-N006_eSPNA_1990_1999_ini.csv'

# Directory to TRACMASS _run.csv file (uncompressed).
runDir = '/gws/nopw/j04/oxford_es/otooth/Tracmass/ORCA1-N006/tracking/eSPNA/'
# Name of TRACMASS _run.csv (uncompressed) file.
runFile = 'ORCA1-N006_eSPNA_1990_1999_run.csv'
# ---------------------------------------------------------------------
# -- Defining .nc file output directory and filename --
# ---------------------------------------------------------------------
# NOTE: Modify directory path to where we would like output .nc file
# to be located:
outDir = '/gws/nopw/j04/oxford_es/otooth/Tracmass/ORCA1-N006/tracking/eSPNA/run/'
# NOTE: Modify the output file name prefix as required for your
# simulation. The final output data store will take the form:
#
# {outFile_YYYY_MM_DD.nc}
#
outFile = "ORCA1-N006_eSPNA_"
# ---------------------------------------------------------------------
# -- Defining column names and types --
# ---------------------------------------------------------------------
# NOTE: Modify the following list and dictionaries as required:
# List of variables (columns) in both _ini.csv and _run.csv files output
# from TRACMASS.
col_names = ['id', 'x', 'y', 'z', 'vol', 't', 'boxface', 'temp', 'sal', 'mld', 'mask']
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
              'mld': np.float64,
              'mask': np.float64,
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
# date_start_str = '1970-01-01'
# date_ini_str = '1975-01-01'
# id_offset = id_max_1970_1974_experiment
# t_offset is computed using the date strings provided.
#
# NOTE: Modify the following variables as required:
# Defining the start date of the overall experiment.
date_start_str = '1975-01-05'
# Defining the initial date of this simulation (not the overall experiment).
date_ini_str = '1990-01-05'
# Defining offset for water parcel ID.
id_offset = 1316311

# Compute time offset in seconds between simulation start date and overall
# experiment start date:
t_offset = (np.datetime64(date_ini_str) - np.datetime64(date_start_str)) / np.timedelta64(1, 's')

# ---------------------------------------------------------------------
# Stage 1:
# Opening the raw ouput files from TRACMASS.

# Read Lagrangian particle inflow positions and properties
# files stored in _ini.csv file.
df_ini = pd.read_csv(iniDir+iniFile, names=col_names) #, dtype=col_dtypes)

# Defining time-levels associated with each particle seeding step.
tlevels = np.unique(df_ini.t)

# Determine dates of seeding step:
dates = np.datetime64(date_ini_str) + np.timedelta64(1, 's') * tlevels
# Export date of seeding step to string in YYYY_MM_DD format.
date_str = pd.to_datetime(dates).strftime('%Y_%m_%d')

# Defining total number of seeding steps in _ini/run.csv.
nsteps = len(tlevels)
# Defining minimum particle ID for seeding step 1.
id_min = 0

# ------------- Header to Script -------------
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('==== Exporting TRACMASS Output from csv to parquet files ====')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-- Description --')
print('Script to export TRACMASS Lagrangian particle')
print('tracking output to store trajectories released')
print('at each seeding step in individual parquet files.')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('-- Info --')
print('Simulation Start Date:', np.datetime64(date_ini_str))
print('Total No. Seeding Steps:', nsteps)
print('Total No. Trajectories:', len(np.unique(df_ini.id)))
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
    # Remove empty column assigned for domain mask status:
    df_run = df_run.drop(columns=['mask'])

    # Apply particle ID and time offsets:
    df_run.id = df_run.id + id_offset
    df_run.t = df_run.t + t_offset

    # ---------------------------------------------------------------------------
    # Stage 2:
    # Saving extracted particle trajectories to new _run.parquet file.

    # Defining filename:
    fname = outFile + date_str[n] + '_run.parquet'
    # Save DataFrame to parquet file without headers:
    df_run.to_parquet(path=outDir+fname, engine='pyarrow', index=False)
# ------------- Footer to Script -------------
print('Completed: Exported TRACMASS .csv to individual .parquet files')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# ------------- Footer to Script -------------
