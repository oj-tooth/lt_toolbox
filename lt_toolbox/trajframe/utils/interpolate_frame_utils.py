##############################################################################
"""
interpolate_frame_utils.py

Description:
Defines functions for interpolating values from trajectories stored in
TrajFrames.
"""
##############################################################################
# Importing relevant packages.
import polars as pl
import numpy as np

##############################################################################
# Define interpolation_1d() function.

def interpolation_1d(df:pl.DataFrame | pl.LazyFrame,
                     field:np.ndarray,
                     dim:str,
                     alias:str
                     ) -> pl.DataFrame | pl.LazyFrame:
    """
    Interpolate values from a 1D field along Lagrangian water
    parcel trajectories.

    Values of the specified field(s) are determined via linear
    interpolation.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing Lagrangian trajectories with positions {x, y, z}.
    fields : ndarray
        Array containing values of 1D field to interpolate.
    dim : str
        Name of column variable corresponding to 1D array dimension.
    alias : str
        Name of column variable to store interpolated values.

    Returns
    -------
    df : DataFrame | LazyFrame
        DataFrame | LazyFrame containing Lagrangian trajectories and interpolated
        values of specified column variable in a new column variable.
    """
    # -------------------
    # Raising exceptions.
    # -------------------
    if isinstance(df, pl.DataFrame) | isinstance(df, pl.LazyFrame) is False:
        raise TypeError("df must be specified as either a polars DataFrame or LazyFrame")
    if isinstance(field, np.ndarray) is False:
        raise TypeError("field must be specified as an ndarray")
    if isinstance(dim, str) is False:
        raise TypeError("dim must be specified as a string")
    if isinstance(alias, str) is False:
        raise TypeError("alias must be specified as a string")

    # ---------------------
    # Linear Interpolation.
    # ---------------------
    # Defining dimension sizes of 1D field:
    width = len(field)
    # Defining neighbouring indices either side of
    # specified (x) coords to interpolate:
    df_x = (df
          .select(
              id = pl.col('id'),
              x = pl.col(dim),
              x1 = pl.col(dim).cast(pl.Int64),
              x2=pl.when((pl.col(dim).cast(pl.Int64) + 1) >= width)
              .then(pl.col(dim).cast(pl.Int64))
              .otherwise(pl.col(dim).cast(pl.Int64) + 1),
              ))

    x = df_x['x'].to_numpy()
    x1 = df_x['x1'].to_numpy()
    x2 = df_x['x2'].to_numpy()

    # Determine values of 1D field at surrounding grid points:
    field_1 = field[x1]
    field_2 = field[x2]

    # Determine difference between given (x) coords and
    # bottom left (x1) grid cell coordinates:
    x_diff = x - x1

    # Linear interpolation by weighting field values at
    # surrounding grid points:
    interpolated = field_1 + x_diff * (field_2-field_1)

    return df.with_columns(pl.Series(name=alias, values=interpolated))

##############################################################################
# Define interpolation_2d() function.

def interpolation_2d(df:pl.DataFrame | pl.LazyFrame,
                     fields:np.ndarray,
                     dims:list,
                     aliases:list
                     ) -> pl.DataFrame | pl.LazyFrame:
    """
    Interpolate values from one or more 2D fields along Lagrangian water
    parcel trajectories.

    Values of the specified field(s) are determined via bi-linear
    interpolation.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing Lagrangian trajectories with positions {x, y, z}.
    fields: ndarray
        List of 2D field arrays containing values to interpolate.
    dims : list(str)
        List of column variable names corresponding to 2D array dimensions.
        All 2D fields must share the same dimensions if a list of arrays
        is specified.
    aliases : str
        Names of column variables to store interpolated values.

    Returns
    -------
    df : DataFrame | LazyFrame
        DataFrame | LazyFrame containing Lagrangian trajectories and interpolated
        values of specified column variable in a new column variable.
    """
    # -------------------
    # Raising exceptions.
    # -------------------
    if isinstance(df, pl.DataFrame) | isinstance(df, pl.LazyFrame) is False:
        raise TypeError("df must be specified as either a polars DataFrame or LazyFrame")
    if isinstance(fields, list) is False:
        raise TypeError("field must be specified as list of ndarrays")
    if isinstance(dims, list) is False:
        raise TypeError("dims must be specified as a list of strings")
    if isinstance(aliases, list) is False:
        raise TypeError("aliases must be specified as a list of strings")

    # ------------------------
    # Bi-Linear Interpolation.
    # ------------------------
    # Defining dimension sizes of 2D field:
    height, width = fields[0].shape
    # Defining neighbouring indices either side of
    # specified (x, y) coords to interpolate:
    df_xy = (df
          .select(
              id = pl.col('id'),
              x = pl.col('x'),
              y = pl.col('y'),
              x1 = pl.col('x').cast(pl.Int64),
              y1 = pl.col('y').cast(pl.Int64),
              x2=pl.when((pl.col('x').cast(pl.Int64) + 1) >= width)
              .then(pl.col('x').cast(pl.Int64))
              .otherwise(pl.col('x').cast(pl.Int64) + 1),
              y2=pl.when((pl.col('y').cast(pl.Int64) + 1) >= height)
              .then(pl.col('y').cast(pl.Int64))
              .otherwise(pl.col('y').cast(pl.Int64) + 1)
              ))

    x = df_xy['x'].to_numpy()
    y = df_xy['y'].to_numpy()
    x1 = df_xy['x1'].to_numpy()
    y1 = df_xy['y1'].to_numpy()
    x2 = df_xy['x2'].to_numpy()
    y2 = df_xy['y2'].to_numpy()

    # Determine difference between given (x, y) coords and
    # bottom left (x1, y1) grid cell coordinates:
    x_diff = x - x1
    y_diff = y - y1

    # Iterate over fields:
    for n, field in enumerate(fields):
        # Determine values of 2D field at surrounding grid points:
        field_11 = field[y1, x1]
        field_12 = field[y2, x1]
        field_21 = field[y1, x2]
        field_22 = field[y2, x2]

        # Bi-linear interpolation by weighting field values at
        # surrounding grid points:
        interpolated = (field_11 * (1 - x_diff) * (1 - y_diff) +
                        field_21 * x_diff * (1 - y_diff) +
                        field_12 * (1 - x_diff) * y_diff +
                        field_22 * x_diff * y_diff)

        df = df.with_columns(pl.Series(name=aliases[n], values=interpolated))

    return df
