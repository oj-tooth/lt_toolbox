##############################################################################
# export_utils.py
#
# Description:
# Defines export functions for conversions module of Lagrangian Trajectories.
# Toolbox
#
# Last Edited:
# 2023/12/24
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.

import zarr
import numpy as np
import xarray as xr
import polars as pl
from tqdm import tqdm

##############################################################################
# Define eval_obs_numbers() function.

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

##############################################################################
# Define export_csv_to_zarr() function.

def export_csv_to_zarr(csv_filename:str, zarr_filename:str, variables:list, attrs:dict, read_options:dict):
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

    # Define DataSet using (trajectory x obs) arrays for each variable:
    ds = xr.Dataset(
        data_vars=data_dict,
        coords=dict(
            trajectory=(["trajectory"], traj_array),
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

##############################################################################
# Define export_zarr_to_parquet() function.

def export_zarr_to_parquet(zarr_filename:str, parquet_filename:str, read_options:dict=None, write_options:dict=None):
    """
    Export Lagrangian trajectory properties and positions
    stored in tabular format in .csv file to array format
    in xarray DataSet saved to .zarr store.

    Parameters
    ----------
    zarr_filename : str
        Path to input .zarr store containing Lagrangian trajectories.
    parquet_filename : str
        Path to output .csv file storing Lagrangian trajectories.
    vars : list
        List of variable names to include in .parquet file.
    read_options : dict
        Dictionary of optional keyword arguments to be
        passed to xarray open_zarr().
    write_options : dict
        Dictionary of optional keyword arguments to be
        passed to polars write_parquet().

    Returns
    -------
    parquet file
        DataFrame containing Lagrangian trajectory properties
        and positions in tabular format.
    """
    # -------------
    # Raise errors:
    # -------------
    # Mandatory arguments:
    if isinstance(parquet_filename, str) is False:
        raise TypeError("invalid type: parquet_filename must be a string.")
    if isinstance(zarr_filename, str) is False:
        raise TypeError("invalid type: zarr_filename must be a string.")

    # Default value arguments:
    if read_options is not None:
        if isinstance(read_options, dict) is False:
            raise TypeError("invalid type: read_options must be a dictionary.")
    else:
        read_options = {}
    if write_options is not None:
        if isinstance(write_options, dict) is False:
            raise TypeError("invalid type: write_options must be a dictionary.")
    else:
        write_options = {}

    # -----------------------------------------------
    # Write Lagrangian trajectories to .parquet file:
    # -----------------------------------------------
    # Import Lagrangian trajectories .zarr file as DataSet:
    ds_run = xr.open_zarr(zarr_filename, **(read_options))

    # Transform DataSet to Dask DataFrame:
    df_dask = (ds_run
               .to_dask_dataframe(dim_order=['trajectory', 'obs'], set_index=False)
               .dropna()
               .drop(columns='obs')
               .rename(columns={'trajectory':'id'})
               )

    # Defining polars Dataframe from Dask DataFrame:
    # NOTE: This loads the entire Dask DataFrame into memory.
    df_polars = pl.from_pandas(df_dask.compute())

    # Write polars DataFrame to .parquet file:
    df_polars.write_parquet(parquet_filename, **(write_options))

    return f"Completed: Lagrangian trajectories written to single .parquet file: {parquet_filename}"

##############################################################################
# Define export_zarr_to_mfparquet() function.

def export_zarr_to_mfparquet(zarr_filename:str, parquet_filedir:str, parquet_filename:str, npartitions:int='auto', read_options:dict=None, write_options:dict=None):
    """
    Export Lagrangian trajectory properties and positions
    stored in array format in .zarr store to tabular format
    in multifile .parquet store.

    Parameters
    ----------
    zarr_filename : str
        Path to input .zarr store containing Lagrangian trajectories.
    parquet_filedir : str
        Path to output .parquet store containing Lagrangian trajectories.
    parquet_filename : str
        Prefix of .parquet files containing Lagrangian trajectories.
    npartitions : int, optional
        Number of partitions to split Dask DataFrame into.
        The default is 'auto' (i.e., automatically determined
        by the xarray to_dask_dataframe() function).
    read_options : dict
        Dictionary of optional keyword arguments to be
        passed to xarray open_zarr().
    write_options : dict
        Dictionary of optional keyword arguments to be
        passed to dask to_parquet().

    Returns
    -------
    parquet store
        DataFrame containing Lagrangian trajectory properties
        and positions in tabular format.
    """
    # -------------
    # Raise errors:
    # -------------
    # Mandatory arguments:
    if isinstance(parquet_filename, str) is False:
        raise TypeError("invalid type: parquet_filename must be a string.")
    if isinstance(parquet_filedir, str) is False:
        raise TypeError("invalid type: parquet_filedir must be a string.")
    if isinstance(zarr_filename, str) is False:
        raise TypeError("invalid type: zarr_filename must be a string.")

    # Default value arguments:
    if npartitions != 'auto':
        if isinstance(npartitions, int) is False:
            raise TypeError("invalid type: npartitions must be an integer.")
    if read_options is not None:
        if isinstance(read_options, dict) is False:
            raise TypeError("invalid type: read_options must be a dictionary.")
    else:
        read_options = {}
    if write_options is not None:
        if isinstance(write_options, dict) is False:
            raise TypeError("invalid type: write_options must be a dictionary.")
    else:
        write_options = {}

    # -----------------------------------------------
    # Write Lagrangian trajectories to .parquet file:
    # -----------------------------------------------
    # Import Lagrangian trajectories .zarr file as DataSet:
    ds_run = xr.open_zarr(zarr_filename, **(read_options))

    # Transform DataSet to Dask DataFrame:
    df_dask = (ds_run
               .to_dask_dataframe(dim_order=['trajectory', 'obs'], set_index=False)
               .dropna()
               .drop(columns='obs')
               .rename(columns={'trajectory':'id'})
               )

    # Repartition Dask DataFrame:
    if npartitions != 'auto':
        df_dask = df_dask.repartition(npartitions=npartitions)

    # Write Dask DataFrame to multiple .parquet files:
    df_dask.to_parquet(path=parquet_filedir,
                       write_index=False,
                       name_function=lambda n: f"{parquet_filename}_{n}.parquet",
                       **(write_options)
                       )

    return f"Completed: Lagrangian trajectories written to multi-file .parquet store: {parquet_filedir}"
