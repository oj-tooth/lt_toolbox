##############################################################################
# transform_frame_utils.py
#
# Description:
# Defines functions for transforming TrajFrames within TrajStore objects.
#
# Date Created:
# 2023/11/09
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.
import polars as pl
import xarray as xr
import numpy as np
from scipy import interpolate

##############################################################################
# Define transform_coords() function.


def transform_coords(df: pl.DataFrame | pl.LazyFrame, lon:xr.DataArray, lat:xr.DataArray, depth:xr.DataArray) -> pl.DataFrame | pl.LazyFrame:
    """
    Transform Lagrangian trajectories from model grid coordinates {i,j,k}.
    to geographical coordinates {lon, lat, depth}.

    Water parcel locations are (bi-)linearly interpolated from ocean general
    circulation model grid.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing Lagrangian trajectories in model coords {i, j, k}.
    lon : DataArray
        Longitudes associated with the center of each model grid cell.
    lat : DataArray
        Latitudes associated with the center of each model grid cell.
    depth : DataArray
        Depths associated with model vertical grid levels.

    Returns
    -------
    df : DataFrame
        DataFrame containing Lagrangian trajectories in geographical
        coords {lon, lat, depth}.
    """
    # ----------------------------------------------
    # Interpolate trajectory {lon, lat} from {x, y}.
    # ----------------------------------------------
    # Bi-linear interpolation of trajectory geographical {lon, lat}
    # positions from model coordinates {i,j}.
    part_lat = lat.interp(x=xr.DataArray(df["x"].to_numpy(), dims="z"), y=xr.DataArray(df["y"].to_numpy(), dims="z"))
    part_lon = lon.interp(x=xr.DataArray(df["x"].to_numpy(), dims="z"), y=xr.DataArray(df["y"].to_numpy(), dims="z"))

    # Replace model coordinate positions with {lon, lat}:
    df = df.with_columns(pl.Series(name="x", values=part_lon.values))
    df = df.with_columns(pl.Series(name="y", values=part_lat.values))

    # ------------------------------------
    # Interpolate trajectory depth from z.
    # ------------------------------------
    # Inserting a value for the sea surface (0 m) for use with
    # interpolation indexes < 1.
    depth = np.insert(depth, 0, 0)
    # Storing the index values for depth.
    index = np.arange(0, len(depth))

    # Use scipy interp1d for linear interpolation function of depth.
    f_depth = interpolate.interp1d(index, depth)

    # Interpolate the particle depth from f_depth using part_z.
    df = df.with_columns(pl.Series(name="z", values=f_depth(df["z"].to_numpy())))

    # Return updated DataFrame with Lagrangian trajectories in geographic
    # coordinates {lat, lon, depth}.
    return df
