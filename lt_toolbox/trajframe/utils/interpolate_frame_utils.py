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
from numba import njit

##############################################################################
# Define norm_diff_1D() function.

@njit(fastmath=True, parallel=True)
def norm_diff_1d(x, x1, x2):
    """
    Compute normalised difference between given 1-D vector and
    model grid cell coordinates.

    Parameters
    ----------
    x : ndarray
        Lagrangian trajectory coordinates.
    x1 : ndarray
        Left-hand model grid coordinates.
    x2 : ndarray
        Right-hand model grid coordinates.

    Returns
    -------
    x_d : ndarray
        Normalised difference between given Lagrangian trajectory
        coords and model grid cell coordinates.
    """
    # Define normalised difference array:
    x_d = np.zeros(x.size)

    # Compute normalised difference coords with exception of
    # final coordinate along the given axis.
    for i, _ in enumerate(x1):
        if x1[i] != x2[i]:
            x_d[i] = (x[i] - x1[i]) / (x2[i] - x1[i])
 
    return x_d

##############################################################################
# Define norm_diff_2D() function.

@njit(fastmath=True, parallel=True)
def norm_diff_2d(x, x1, x2, y, y1, y2):
    """
    Compute normalised difference between two given 1-D vectors and
    model grid cell coordinates.

    Parameters
    ----------
    x : ndarray
        Lagrangian trajectory coordinates.
    x1 : ndarray
        Left-hand model grid coordinates.
    x2 : ndarray
        Right-hand model grid coordinates.
    y : ndarray
        Lagrangian trajectory coordinates.
    y1 : ndarray
        Left-hand model grid coordinates.
    y2 : ndarray
        Right-hand model grid coordinates.

    Returns
    -------
    x_d : ndarray
        Normalised difference between given Lagrangian trajectory
        coords and model grid cell coordinates.
    y_d : ndarray
        Normalised difference between given Lagrangian trajectory
        coords and model grid cell coordinates.
    """
    # Define normalised difference arrays:
    x_d = np.zeros(x.size)
    y_d = np.zeros(y.size)

    # Compute normalised difference coords with exception of
    # final coordinate along the given axes.
    for i, _ in enumerate(x1):
        if x1[i] != x2[i]:
            x_d[i] = (x[i] - x1[i]) / (x2[i] - x1[i])
        if y1[i] != y2[i]:
            y_d[i] = (y[i] - y1[i]) / (y2[i] - y1[i])
    return x_d, y_d

#############################################################################
# Define norm_diff_3D() function.

@njit(fastmath=True, parallel=True)
def norm_diff_3d(x, x1, x2, y, y1, y2, z, z1, z2):
    """
    Compute normalised difference between three given 1-D vectors and
    model grid cell coordinates.

    Parameters
    ----------
    x : ndarray
        Lagrangian trajectory coordinates.
    x1 : ndarray
        Left-hand model grid coordinates.
    x2 : ndarray
        Right-hand model grid coordinates.
    y : ndarray
        Lagrangian trajectory coordinates.
    y1 : ndarray
        Left-hand model grid coordinates.
    y2 : ndarray
        Right-hand model grid coordinates.
    z : ndarray
        Lagrangian trajectory coordinates.
    z1 : ndarray
        Left-hand model grid coordinates.
    z2 : ndarray
        Right-hand model grid coordinates.

    Returns
    -------
    x_d : ndarray
        Normalised difference between given Lagrangian trajectory
        coords and model grid cell coordinates.
    y_d : ndarray
        Normalised difference between given Lagrangian trajectory
        coords and model grid cell coordinates.
    z_d : ndarray
        Normalised difference between given Lagrangian trajectory
        coords and model grid cell coordinates.
    """
    # Define normalised difference arrays:
    x_d = np.zeros(x.size)
    y_d = np.zeros(y.size)
    z_d = np.zeros(z.size)

    # Compute normalised difference coords with exception of
    # final coordinate along the given axes.
    for i, _ in enumerate(x1):
        if x1[i] != x2[i]:
            x_d[i] = (x[i] - x1[i]) / (x2[i] - x1[i])
        if y1[i] != y2[i]:
            y_d[i] = (y[i] - y1[i]) / (y2[i] - y1[i])
        if z1[i] != z2[i]:
            z_d[i] = (z[i] - z1[i]) / (z2[i] - z1[i])

    return x_d, y_d, z_d

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

    # Determine normalised difference between given (x) coords and
    # bottom left (x1) grid cell coordinates:
    x_d = norm_diff_1d(x, x1, x2)

    # Linear interpolation by weighting field values at
    # surrounding grid points:
    interpolated = field_1 + x_d * (field_2-field_1)

    return df.with_columns(pl.Series(name=alias, values=interpolated))

##############################################################################
# Define interpolation_2d() function.

def interpolation_2d(df:pl.DataFrame | pl.LazyFrame,
                     fields:list,
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
    fields: list(ndarray)
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

    # Determine normalised difference between given (x, y) coords and
    # bottom left (x1, y1) grid cell coordinates:
    x_d, y_d = norm_diff_2d(x, x1, x2, y, y1, y2)

    # Iterate over fields:
    for n, field in enumerate(fields):
        # Determine values of 2D field at surrounding grid points:
        field_11 = field[y1, x1]
        field_12 = field[y2, x1]
        field_21 = field[y1, x2]
        field_22 = field[y2, x2]

        # Bi-linear interpolation by weighting field values at
        # surrounding grid points:
        field_1 = field_11*(1 - x_d) + field_21*x_d
        field_2 = field_12*(1 - x_d) + field_22*x_d

        interpolated = field_1*(1 - y_d) + field_2*y_d

        df = df.with_columns(pl.Series(name=aliases[n], values=interpolated))

    return df

##############################################################################
# Define interpolation_3d() function.

def interpolation_3d(df:pl.DataFrame | pl.LazyFrame,
                     fields:list,
                     dims:list,
                     aliases:list
                     ) -> pl.DataFrame | pl.LazyFrame:
    """
    Interpolate values from one or more 3D fields along Lagrangian water
    parcel trajectories.

    Values of the specified field(s) are determined via tri-linear
    interpolation.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing Lagrangian trajectories with positions {x, y, z}.
    fields: list(ndarray)
        List of 3D field arrays containing values to interpolate.
    dims : list(str)
        List of column variable names corresponding to 3D array dimensions.
        All 3D fields must share the same dimensions if a list of arrays
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
    # Tri-Linear Interpolation.
    # ------------------------
    # Defining dimension sizes of 3D field:
    height, width, depth = fields[0].shape
    # Defining neighbouring indices either side of
    # specified (x, y, z) coords to interpolate:
    df_xyz = (df
          .select(
              id = pl.col('id'),
              x = pl.col('x').cast(pl.Float64),
              y = pl.col('y').cast(pl.Float64),
              z = pl.col('z').cast(pl.Float64),
              x1 = pl.col('x').cast(pl.Int64),
              y1 = pl.col('y').cast(pl.Int64),
              z1 = pl.col('z').cast(pl.Int64),
              x2=pl.when((pl.col('x').cast(pl.Int64) + 1) >= width)
              .then(pl.col('x').cast(pl.Int64))
              .otherwise(pl.col('x').cast(pl.Int64) + 1),
              y2=pl.when((pl.col('y').cast(pl.Int64) + 1) >= height)
              .then(pl.col('y').cast(pl.Int64))
              .otherwise(pl.col('y').cast(pl.Int64) + 1),
              z2=pl.when((pl.col('z').cast(pl.Int64) + 1) >= depth)
              .then(pl.col('z').cast(pl.Int64))
              .otherwise(pl.col('z').cast(pl.Int64) + 1),
              ))

    x = df_xyz['x'].to_numpy()
    y = df_xyz['y'].to_numpy()
    z = df_xyz['z'].to_numpy()
    x1 = df_xyz['x1'].to_numpy()
    y1 = df_xyz['y1'].to_numpy()
    z1 = df_xyz['z1'].to_numpy()
    x2 = df_xyz['x2'].to_numpy()
    y2 = df_xyz['y2'].to_numpy()
    z2 = df_xyz['z2'].to_numpy()

    # Determine normalised difference between given (x, y, z) coords and
    # bottom left (x1, y1, z1) grid cell coordinates:
    x_d, y_d, z_d = norm_diff_3d(x, x1, x2, y, y1, y2, z, z1, z2)

    # Iterate over 3-D fields:
    for n, field in enumerate(fields):
        # Determine values of 3D field at surrounding grid points:
        field_111 = field[z1, y1, x1]
        field_211 = field[z2, y1, x1]
        field_121 = field[z1, y2, x1]
        field_112 = field[z1, y1, x2]
        field_212 = field[z2, y1, x2]
        field_221 = field[z2, y2, x1]
        field_122 = field[z1, y2, x2]
        field_222 = field[z2, y2, x2]

        # Interpolate along first dimension, z:
        field_11 = field_111*(1 - z_d) + field_211*z_d
        field_12 = field_112*(1 - z_d) + field_212*z_d
        field_21 = field_121*(1 - z_d) + field_221*z_d
        field_22 = field_122*(1 - z_d) + field_222*z_d

        # Interpolate along second dimension, y:
        field_1 = field_11*(1 - y_d) + field_21*y_d
        field_2 = field_12*(1 - y_d) + field_22*y_d

        # Interpolate along third dimension, x:
        interpolated = field_1*(1 - x_d) + field_2*x_d

        df = df.with_columns(pl.Series(name=aliases[n], values=interpolated))

    return df
