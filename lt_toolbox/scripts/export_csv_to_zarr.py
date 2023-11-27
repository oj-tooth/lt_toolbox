################################################################
# export_csv_to_zarr.py
# --------------------------------------------------------------
# Description: Script to transform Lagrangian particle
# tracking output .csv files into .zarr files.
#
# User Input: Locations where user modification is required
# are indicated beneath NOTE statements in the script.
#
# --------------------------------------------------------------
# Date Created: 2023-11-10
#
# Author: Ollie Tooth
###############################################################

# Import packages.
import zarr
import xarray as xr
import numpy as np
import polars as pl
from tqdm import tqdm

# --- Define eval_obs_numbers() function ---
def eval_obs_numbers(time:pl.Series):
    """
    Create Series with observation number (int)
    for the corresponding positions and properties
    stored at regular or irregular time-intervals
    along a Lagrangian trajectory.

    Parameters
    ----------
    time : Series
        times corresponding to each observation along
        Lagrangian trajectory.

    Returns
    -------
    obs : Series
        observation number along Lagrangian trajectory.
    """
    # Define observation number along Lagrangian trajectory:
    obs = pl.Series(name='obs', values=np.arange(1, len(time)+1))

    return obs

# --- Define export_csv_to_zarr() function ---
def export_csv_to_zarr(csv_filename, zarr_filename, variables, attrs, read_options):
    """
    Export Lagrangian trajectory properties and positions
    stored in tabular format in .csv file to array format
    in xarray DataSet saved to .zarr store.

    Parameters
    ----------
    csv_filename : str
        Path to input .csv file storing Lagrangian trajectories.
    zarr_filename : str
        Path to output .zarr store containing Lagrangian trajectories.
    vars : list
        List of variable names to include in .zarr store.
    attrs : dict
        Attributes of resulting xarray DataSet.
    read_options : dict
        Dictionary of optional keyword arguments to be
        passed to polars read_csv().

    Returns
    -------
    zarr store
        DataSet containing Lagrangian trajectory properties
        and positions in array format.
    """
    # Import Lagrangian trajectories .csv file as DataFrame:
    df_run = pl.read_csv(csv_filename, **(read_options or {}))

    # Add observation numbers for each Lagrangian trajectory in DataFrame:
    df_obs = (df_run
        .sort(by='id')
        .group_by(by=pl.col('id'), maintain_order=True)
        .agg(
            pl.map_groups(exprs=['time'], function=lambda args : eval_obs_numbers(args[0])).alias('obs')
            )
            )

    # Add exploded observations column to DataFrame:
    df = (df_run
        .sort(by='id')
        .with_columns(
            df_obs.explode(columns='obs')['obs']
            ))

    # Initialise empty list to store ndarrays for column variables
    var_arrays = []

    # Iterate over column variables:
    for col in tqdm(variables):
        # For non-ID variables pivot to form table of values with
        # rows -> IDs and columns -> obs dimensions:
        var = df.pivot(values=col, index='id', columns='obs')
        # Drop first column storing unique ID and append as
        # ndarray to list:
        var_arrays.append(var.drop(['id']).to_numpy())

    # Define trajectory dimension 1-D array as unique IDs:
    traj_array = var['id'].to_numpy()
    # Define observation dimension 1-D as sequence to largest no. of observations:
    obs_array = np.array(var.drop('id').columns, dtype=np.int32)

    # Construct dictionary to define DataArrays from ndarrays:
    data_dict = {variables[n]: var_arrays[n] for n in range(len(variables))}

    # Define DataSet using (traj x obs) arrays for each variable:
    ds = xr.Dataset(
        data_vars=data_dict,
        coords=dict(
            traj=(["traj"], traj_array),
            obs=(["obs"], obs_array),
        ),
        attrs=attrs,
        )

    # Define compressor:
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    # Define encodings dictionary:
    enc = {var: {"compressor": compressor} for var in ds}

    # Write Lagrangian trajectories to compressed .zarr store:
    ds.to_zarr(zarr_filename, encoding=enc)

    return 
