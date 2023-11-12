################################################################
# export_zarr_to_parquet.py
# --------------------------------------------------------------
# Description: Script to transform Lagrangian particle
# tracking output .zarr files into .parquet files.
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
import xarray as xr
import polars.selectors as cs
import polars as pl
from tqdm import tqdm

# --- Define export_zarr_to_parquet() function ---
def export_zarr_to_parquet(zarr_filename, parquet_filename, variables, read_options):
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

    Returns
    -------
    parquet file
        DataFrame containing Lagrangian trajectory properties
        and positions in tabular format.
    """
    # Import Lagrangian trajectories .zarr file as DataSet:
    ds_run = xr.open_zarr(zarr_filename, chunks=None, **(read_options or {}))

    # Initialise empty list to store Series of column variables:
    var_series = []

    # Iterate over specified column variables in DataSet:
    for var in tqdm(variables):
        # Construct DateFrame from column variable ndarray:
        df = pl.DataFrame(ds_run[var].values)

        # Assign the name of columns in the DataFrame to
        # be observation numbers:
        df.columns = ds_run.obs.to_numpy().astype('str')
        # Insert Lagrangian trajectory IDs as a new column
        # in DataFrame:
        df.insert_at_idx(index=0, series=pl.Series(name='id', values=ds_run.traj.to_numpy().astype('int')))
        # Melt DataFrame from (traj x obs) dimensions to tabular
        # DataFrame:
        df = df.melt(id_vars='id', value_vars=cs.numeric(), value_name=var, variable_name='obs')
        # Remove NaN values stored in original sparse array from
        # DataFrame and remove ID values before sorting:
        df = df.filter(pl.col(var).is_not_nan()).filter(pl.col('obs') != 'id').sort(by='id')
        # Append column variable Series to list:
        var_series.append(df.select([var]).to_series())

    # Add obs column variable with integer dtype:
    df = df.with_columns(pl.col('obs').cast(pl.Int32))
    # Add observation number and ID Series to list:
    var_series.insert(0, df.select(['obs']).to_series())
    var_series.insert(0, df.select(['id']).to_series())

    # Construct DataFrame from list of Series:
    df_out = pl.DataFrame(data=var_series)

    # Write polars DataFrame to .parquet file:
    df_out.write_parquet(parquet_filename, use_pyarrow=True)

    return
