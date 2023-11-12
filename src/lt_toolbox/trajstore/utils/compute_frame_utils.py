##############################################################################
# compute_frame_utils.py
#
# Description:
# Defines functions for computing new properties for TrajStore objects.
#
# Date Created:
# 2023/11/09
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.
import numpy as np
import xarray as xr
import polars as pl

##############################################################################
# Define haversine_distance() function.


def haversine_dist(lon, lat, cum_dist):
    """
    Compute distance (m) between trajectory positions using Haversine formula.

    Latitude and Longitude vectors for the start (1) and end (2) positions
    are used to compute the distance (m) between them.

    Parameters
    ----------
    lon : Series
        Longitudes of trajectory positions.
    lat : Series
        Latitudes of trajectory positions.
    cum_dist : boolean
        Compute the cumulative distance along trajectory.

    Returns
    -------
    Series
        Haversine distance between neighbouring trajectory positions
        or cumulative distance along trajectory.
    """
    # -------------------
    # Raising Exceptions:
    # -------------------
    if isinstance(cum_dist, bool) is False:
        raise TypeError('invalid type - cumdist must be specified as a boolean')

    # -------------------------------------------
    # Defining Variables and Physical Parameters
    # -------------------------------------------
    # Defining difference between longitudes and latitudes, dLon and dLat.
    dlon = np.radians(lon).diff()
    dlat = np.radians(lat).diff()

    # Defining radius of the Earth, re (m), as
    # volumetric mean radius from NASA.
    # See: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    re = 6371000

    # ------------------------------------------------------------------
    # Computing the distance (km) between (Lat1, Lon1) and (Lat2, Lon2).
    # ------------------------------------------------------------------
    # Compute displacement angle in radians.
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat)) * np.cos(np.radians(lat)) * np.sin(dlon/2)**2
    b = 2 * np.arcsin(np.sqrt(a))

    # Distance (m) obtain by multiplying displacement angle by
    # radius of Earth.
    dist = b * re

    if cum_dist is True:
        # Calculate the accumulated distance travelled along trajectory:
        dist = dist.cumsum()

    # Returning distance array, d (m).
    return dist

##############################################################################
# Define binned_statistic_1d() function.


def binned_statistic_1d(var:pl.Series, values:pl.Series, statistic:str, bin_breaks:list) -> xr.DataArray:
    """
    Compute a 1-dimensional binned statistic using the Series stored in a
    DataFrame.

    This is a generalization of a histogram function. A histogram divides
    the chosen column varaible into bins, and returns the count of the number
    of points in each bin. This function allows the computation of the sum,
    mean, median, or other statistic of the values within each bin.

    Parameters
    var : Series
        A sequence of values to be binned.

    values : Series
        The values on which the statistic will be computed.
        This must be the same length as var

    statistic: str
        The statistic to compute.
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by null.
          * 'std' : compute the standard deviation within each bin.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by null.
          * 'count' : compute the count of points within each bin. This is
            identical to an unweighted histogram.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by null.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by null.

    bin_breaks: list
          List of bin edges used in the binning of var variable.

    Returns
    -------
    statistic : DataArray
        DataArray containing values of the selected statistic in each bin.
    """
    # --- Raise Errors ---
    available_stats = ['mean', 'median', 'count', 'sum', 'std', 'min', 'max']
    if not callable(statistic) and statistic not in available_stats:
        raise ValueError(f'invalid statistic {statistic!r}')

    # --- Definitions ---
    # Define DataFrame from var and values Series:
    df = pl.DataFrame(data=[var, values])
    # Define column names:
    var_name = var.name
    values_name = values.name

    # Define bin width:
    bin_width = bin_breaks[1] - bin_breaks[0]

    # Define extended bin breaks:
    bin_breaks_ext = bin_breaks.copy()
    # Extend bin breaks at start and end:
    bin_breaks_ext.insert(0, bin_breaks_ext[0] - bin_width)
    bin_breaks_ext.extend([bin_breaks_ext[-1] + bin_width])

    # Define bin labels as Polars Series:
    bin_labels = pl.Series(name='labels', values=bin_breaks_ext)
    # Determine bin labels from mid-points of extended bin breaks:
    bin_labels = (bin_labels[:-1] + bin_labels[1:]) / 2
    # Transform labels to string dtype:
    bin_labels = bin_labels.cast(pl.Utf8)

    # --- Label Variable with Discrete Bins ---
    # Calculate 1-dimensional binned statistics:
    df_binned = (df
                 .with_columns(
                     pl.col(var_name).cut(breaks=bin_breaks, labels=bin_labels).alias('values_binned')
                     )
                 )

    # --- Calculate Statistic in Discrete Bins ---
    # Evaluate statistic over values stored in each bin:
    if statistic == 'mean':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values_name).mean())
                        )
    elif statistic == 'median':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values_name).median())
                        )
    elif statistic == 'count':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values_name).count())
                        )
    elif statistic == 'sum':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values_name).sum())
                        )
    elif statistic == 'std':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values_name).std())
                        )
    elif statistic == 'min':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values_name).min())
                        )
    elif statistic == 'max':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values_name).max())
                        )

    # Cast value bins dtype from categorical to float:
    result = result.with_columns(pl.col('values_binned').cast(pl.Utf8).cast(pl.Float64))
    # Sort DataFrame by bin value in ascending order:
    result = result.sort(by='values_binned', descending=False)

    # --- Transform Binned Statistic to DataArray ---
    # Construct xarray DataArray from Polars Series:
    result_array = xr.DataArray(data=result[values_name].to_numpy(),
                                dims=[var_name],
                                coords={
                                    var_name:([var_name], result['values_binned'].to_numpy().astype(np.float64))
                                },
                                name=values_name
                                )

    # Return 1-dimensional binned statistics as DataArray:
    return result_array

##############################################################################
# Define binned_group_statistic_1d() function.


def binned_group_statistic_1d(var:pl.Series, values:pl.Series, groups:pl.Series, statistic:str, bin_breaks:list) -> xr.DataArray:
    """
    Compute a grouped  1-dimensional binned statistic using the Series stored
    in a DataFrame.

    This is a generalization of a histogram function. A histogram divides
    the chosen column varaible into bins, and returns the count of the number
    of points in each bin. This function allows the computation of the sum,
    mean, median, or other statistic of the values within each bin.

    Parameters
    var : Series
        A sequence of values to be binned.

    values : Series
        The values on which the statistic will be computed.
        This must be the same length as var

    groups : Series
        A sequence of values to grouped according to unique
        values using group_by() method.

    statistic: str
        The statistic to compute.
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by null.
          * 'std' : compute the standard deviation within each bin.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by null.
          * 'count' : compute the count of points within each bin. This is
            identical to an unweighted histogram.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by null.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by null.

    bin_breaks: list
          List of bin edges used in the binning of var variable.

    Returns
    -------
    statistic : DataArray
        DataArray containing values of the selected statistic in each bin.
    """
    # --- Raise Errors ---
    available_stats = ['mean', 'median', 'count', 'sum', 'std', 'min', 'max']
    if not callable(statistic) and statistic not in available_stats:
        raise ValueError(f'invalid statistic {statistic!r}')

    # --- Definitions ---
    # Define DataFrame from var and values Series:
    df = pl.DataFrame(data=[var, values, groups])
    # Define column names:
    var_name = var.name
    values_name = values.name
    groups_name = groups.name

    # Define bin width:
    bin_width = bin_breaks[1] - bin_breaks[0]

    # Define extended bin breaks:
    bin_breaks_ext = bin_breaks.copy()
    # Extend bin breaks at start and end:
    bin_breaks_ext.insert(0, bin_breaks_ext[0] - bin_width)
    bin_breaks_ext.extend([bin_breaks_ext[-1] + bin_width])

    # Define bin labels as Polars Series:
    bin_labels = pl.Series(name='labels', values=bin_breaks_ext)
    # Determine bin labels from mid-points of extended bin breaks:
    bin_labels = (bin_labels[:-1] + bin_labels[1:]) / 2
    # Transform labels to string dtype:
    bin_labels = bin_labels.cast(pl.Utf8)

    # --- Label Variable with Discrete Bins ---
    # Calculate 1-dimensional binned statistics:
    df_binned = (df
                 .with_columns(
                     pl.col(var_name).cut(breaks=bin_breaks, labels=bin_labels).alias('values_binned')
                     )
                 )

    # --- Calculate Statistic in Discrete Bins ---
    # Initialise empty list to store results:
    result_array_list = []
    # Iterate over DataFrame groups:
    for group, df_group in df_binned.group_by(by=groups_name, maintain_order=True):
        # Evaluate statistic over values stored in each bin:
        if statistic == 'mean':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values_name).mean())
                            )
        elif statistic == 'median':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values_name).median())
                            )
        elif statistic == 'count':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values_name).count())
                            )
        elif statistic == 'sum':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values_name).sum())
                            )
        elif statistic == 'std':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values_name).std())
                            )
        elif statistic == 'min':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values_name).min())
                            )
        elif statistic == 'max':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values_name).max())
                            )

        # Cast value bins dtype from categorical to float:
        result = result.with_columns(pl.col('values_binned').cast(pl.Utf8).cast(pl.Float64))
        # Sort DataFrame by bin value in ascending order:
        result = result.sort(by='values_binned', descending=False)

        # --- Transform Binned Statistic to DataArray ---
        # Construct xarray DataArray from Polars Series:
        group_array = xr.DataArray(data=result[values_name].to_numpy()[np.newaxis, :],
                                    dims=[groups_name, var_name],
                                    coords={
                                        groups_name:([groups_name], np.array([group])),
                                        var_name:([var_name], result['values_binned'].to_numpy().astype(np.float64))
                                    },
                                    name=values_name
                                    )
        # Append DataArray to list:
        result_array_list.append(group_array)

    # Concatenate DataArrays contained in list along group dimension:
    result_array = xr.concat(result_array_list, dim=groups_name)

    # Return 1-dimensional binned statistics as DataArray:
    return result_array

##############################################################################
# Define binned_statistic_2d() function.


def binned_statistic_2d(var_x:pl.Series, var_y:pl.Series, values:pl.Series, statistic:str, bin_breaks:list) -> xr.DataArray:
    """
    Compute a 2-dimensional binned statistic using the Series stored in a
    DataFrame.

    This is a generalization of a 2-D histogram function. A 2-D histogram divides
    the chosen column varaibles into bins, and returns the count of the number
    of points in each bin. This function allows the computation of the sum,
    mean, median, or other statistic of the values within each bin.

    Parameters
    var_x : Series
        A sequence of values to be binned along the first dimension.

    var_y : Series
        A sequence of values to be binned along the second dimension.

    values : Series
        The values on which the statistic will be computed.
        This must be the same length as var_x & var_y.

    statistic: str
        The statistic to compute.
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by null.
          * 'std' : compute the standard deviation within each bin.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by null.
          * 'count' : compute the count of points within each bin. This is
            identical to an unweighted histogram.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by null.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by null.

    bin_breaks: list
          List of lists including bin edges used in the binning of var_x
          and var_y variables.

    Returns
    -------
    statistic : DataArray
        DataArray containing values of the selected statistic in each bin.
    """
    # --- Raise Errors ---
    available_stats = ['mean', 'median', 'count', 'sum', 'std', 'min', 'max']
    if not callable(statistic) and statistic not in available_stats:
        raise ValueError(f'invalid statistic {statistic!r}')

    # --- Definitions ---
    # Define DataFrame from var_x, var_y and values Series:
    df = pl.DataFrame(data=[var_x, var_y, values])
    # Define column names:
    var_x_name = var_x.name
    var_y_name = var_y.name
    values_name = values.name

    # Separate specified bins edges:
    bin_x_breaks = bin_breaks[0]
    bin_y_breaks = bin_breaks[1]

    # Define bin widths:
    bin_x_width = bin_x_breaks[1] - bin_x_breaks[0]
    bin_y_width = bin_y_breaks[1] - bin_y_breaks[0]

    # Define extended bin breaks:
    bin_x_breaks_ext = bin_x_breaks.copy()
    # Extend bin breaks at start and end:
    bin_x_breaks_ext.insert(0, bin_x_breaks_ext[0] - bin_x_width)
    bin_x_breaks_ext.extend([bin_x_breaks_ext[-1] + bin_x_width])

    # Define bin labels as Polars Series:
    bin_x_labels = pl.Series(name='labels', values=bin_x_breaks_ext)
    # Determine bin labels from mid-points of extended bin breaks:
    bin_x_labels = (bin_x_labels[:-1] + bin_x_labels[1:]) / 2
    # Transform labels to string dtype:
    bin_x_labels = bin_x_labels.cast(pl.Utf8)

    # Define extended bin breaks:
    bin_y_breaks_ext = bin_y_breaks.copy()
    # Extend bin breaks at start and end:
    bin_y_breaks_ext.insert(0, bin_y_breaks_ext[0] - bin_y_width)
    bin_y_breaks_ext.extend([bin_y_breaks_ext[-1] + bin_y_width])

    # Define bin labels as Polars Series:
    bin_y_labels = pl.Series(name='labels', values=bin_y_breaks_ext)
    # Determine bin labels from mid-points of extended bin breaks:
    bin_y_labels = (bin_y_labels[:-1] + bin_y_labels[1:]) / 2
    # Transform labels to string dtype:
    bin_y_labels = bin_y_labels.cast(pl.Utf8)

    # --- Label Variable with Discrete Bins ---
    # Calculate 2-dimensional binned statistics:
    df_binned = (df
                 .with_columns(
                     pl.col(var_x_name).cut(breaks=bin_x_breaks, labels=bin_x_labels).alias('values_x_binned'),
                     pl.col(var_y_name).cut(breaks=bin_y_breaks, labels=bin_y_labels).alias('values_y_binned'),
                     )
                 )

    # --- Calculate Statistic in Discrete Bins ---
    # Evaluate statistic over values stored in each bin:
    if statistic == 'mean':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values_name).mean())
                        )
    elif statistic == 'median':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values_name).median())
                        )
    elif statistic == 'count':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values_name).count())
                        )
    elif statistic == 'sum':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values_name).sum())
                        )
    elif statistic == 'std':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values_name).std())
                        )
    elif statistic == 'min':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values_name).min())
                        )
    elif statistic == 'max':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values_name).max())
                        )

    # Cast value bins dtypes from categorical to float:
    result = (result
            .with_columns(
                pl.col('values_x_binned').cast(pl.Utf8).cast(pl.Float64),
                pl.col('values_y_binned').cast(pl.Utf8).cast(pl.Float64),
                )
            )
    # Sort DataFrame by bin value in ascending order:
    result = result.sort(by='values_x_binned', descending=False)

    # --- Pivot 2-D Statistic DataFrame ---
    # Pivot DataFrame to store statistic values in form:
    # [variable x-bins -> rows] x [variable y-bins -> columns].
    result_pivot = result.pivot(values=values_name, index='values_x_binned', columns='values_y_binned')

    # Store variable x bins as ndarray:
    var_x_array = np.sort(result_pivot['values_x_binned'].to_numpy())
    # Drop variable x bins from DataFrame:
    result_pivot = result_pivot.drop(columns='values_x_binned')

    # Store variable y bins as sorted ndarray:
    var_y_array = np.sort(np.array(result_pivot.columns, dtype=np.float64))
    # Define sorted column names in list:
    column_names = var_y_array.astype('str').tolist()
    # Re-order columns of pivoted DataFrame:
    result_pivot = result_pivot.select(column_names)

    # --- Transform Binned Statistic to DataArray ---
    # Construct xarray DataArray from Polars Series:
    result_array = xr.DataArray(data=result_pivot.to_numpy(),
                                dims=[var_x_name, var_y_name],
                                coords={
                                    var_x_name:([var_x_name], var_x_array),
                                    var_y_name:([var_y_name], var_y_array),
                                },
                                name=values_name
                                )

    # Return 2-dimensional binned statistics as DataArray:
    return result_array

##############################################################################
# Define binned_group_statistic_2d() function.


def binned_group_statistic_2d(var_x:pl.Series, var_y:pl.Series, values:pl.Series, groups:pl.Series, statistic:str, bin_breaks:list) -> xr.DataArray:
    """
    Compute a 2-dimensional grouped binned statistic using the Series stored in a
    DataFrame.

    This is a generalization of a 2-D histogram function. A 2-D histogram divides
    the chosen column varaibles into bins, and returns the count of the number
    of points in each bin. This function allows the computation of the sum,
    mean, median, or other statistic of the values within each bin.

    Parameters
    var_x : Series
        A sequence of values to be binned along the first dimension.

    var_y : Series
        A sequence of values to be binned along the second dimension.

    values : Series
        The values on which the statistic will be computed.
        This must be the same length as var_x & var_y.

    groups : Series
        A sequence of values to grouped according to unique
        values using group_by() method.

    statistic: str
        The statistic to compute.
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by null.
          * 'std' : compute the standard deviation within each bin.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by null.
          * 'count' : compute the count of points within each bin. This is
            identical to an unweighted histogram.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by null.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by null.

    bin_breaks: list
          List of lists including bin edges used in the binning of var_x
          and var_y variables.

    Returns
    -------
    statistic : DataArray
        DataArray containing values of the selected statistic in each bin.
    """
    # --- Raise Errors ---
    available_stats = ['mean', 'median', 'count', 'sum', 'std', 'min', 'max']
    if not callable(statistic) and statistic not in available_stats:
        raise ValueError(f'invalid statistic {statistic!r}')

    # --- Definitions ---
    # Define DataFrame from var_x, var_y and values Series:
    df = pl.DataFrame(data=[var_x, var_y, values, groups])
    # Define column names:
    var_x_name = var_x.name
    var_y_name = var_y.name
    values_name = values.name
    groups_name = groups.name

    # Separate specified bins edges:
    bin_x_breaks = bin_breaks[0]
    bin_y_breaks = bin_breaks[1]

    # Define bin widths:
    bin_x_width = bin_x_breaks[1] - bin_x_breaks[0]
    bin_y_width = bin_y_breaks[1] - bin_y_breaks[0]

    # Define extended bin breaks:
    bin_x_breaks_ext = bin_x_breaks.copy()
    # Extend bin breaks at start and end:
    bin_x_breaks_ext.insert(0, bin_x_breaks_ext[0] - bin_x_width)
    bin_x_breaks_ext.extend([bin_x_breaks_ext[-1] + bin_x_width])

    # Define bin labels as Polars Series:
    bin_x_labels = pl.Series(name='labels', values=bin_x_breaks_ext)
    # Determine bin labels from mid-points of extended bin breaks:
    bin_x_labels = (bin_x_labels[:-1] + bin_x_labels[1:]) / 2
    # Transform labels to string dtype:
    bin_x_labels = bin_x_labels.cast(pl.Utf8)

    # Define extended bin breaks:
    bin_y_breaks_ext = bin_y_breaks.copy()
    # Extend bin breaks at start and end:
    bin_y_breaks_ext.insert(0, bin_y_breaks_ext[0] - bin_y_width)
    bin_y_breaks_ext.extend([bin_y_breaks_ext[-1] + bin_y_width])

    # Define bin labels as Polars Series:
    bin_y_labels = pl.Series(name='labels', values=bin_y_breaks_ext)
    # Determine bin labels from mid-points of extended bin breaks:
    bin_y_labels = (bin_y_labels[:-1] + bin_y_labels[1:]) / 2
    # Transform labels to string dtype:
    bin_y_labels = bin_y_labels.cast(pl.Utf8)

    # --- Label Variable with Discrete Bins ---
    # Calculate 2-dimensional binned statistics:
    df_binned = (df
                 .with_columns(
                     pl.col(var_x_name).cut(breaks=bin_x_breaks, labels=bin_x_labels).alias('values_x_binned'),
                     pl.col(var_y_name).cut(breaks=bin_y_breaks, labels=bin_y_labels).alias('values_y_binned'),
                     )
                 )

    # --- Calculate Statistic in Discrete Bins ---
    # Initialise empty list to store results:
    result_array_list = []
    # Iterate over DataFrame groups:
    for group, df_group in df_binned.group_by(by=groups_name, maintain_order=True):
        # Evaluate statistic over values stored in each bin:
        if statistic == 'mean':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values_name).mean())
                            )
        elif statistic == 'median':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values_name).median())
                            )
        elif statistic == 'count':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values_name).count())
                            )
        elif statistic == 'sum':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values_name).sum())
                            )
        elif statistic == 'std':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values_name).std())
                            )
        elif statistic == 'min':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values_name).min())
                            )
        elif statistic == 'max':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values_name).max())
                            )

        # Cast value bins dtypes from categorical to float:
        result = (result
                .with_columns(
                    pl.col('values_x_binned').cast(pl.Utf8).cast(pl.Float64),
                    pl.col('values_y_binned').cast(pl.Utf8).cast(pl.Float64),
                    )
                )
        # Sort DataFrame by bin value in ascending order:
        result = result.sort(by='values_x_binned', descending=False)

        # --- Pivot 2-D Statistic DataFrame ---
        # Pivot DataFrame to store statistic values in form:
        # [variable x-bins -> rows] x [variable y-bins -> columns].
        result_pivot = result.pivot(values=values_name, index='values_x_binned', columns='values_y_binned')

        # Store variable x bins as ndarray:
        var_x_array = np.sort(result_pivot['values_x_binned'].to_numpy())
        # Drop variable x bins from DataFrame:
        result_pivot = result_pivot.drop(columns='values_x_binned')

        # Store variable y bins as sorted ndarray:
        var_y_array = np.sort(np.array(result_pivot.columns, dtype=np.float64))
        # Define sorted column names in list:
        column_names = var_y_array.astype('str').tolist()
        # Re-order columns of pivoted DataFrame:
        result_pivot = result_pivot.select(column_names)

        # --- Transform Binned Statistic to DataArray ---
        # Construct xarray DataArray from Polars Series:
        group_array = xr.DataArray(data=result_pivot.to_numpy()[np.newaxis, :, :],
                                    dims=[groups_name, var_x_name, var_y_name],
                                    coords={
                                        groups_name:([groups_name], np.array([group])),
                                        var_x_name:([var_x_name], var_x_array),
                                        var_y_name:([var_y_name], var_y_array),
                                    },
                                    name=values_name
                                    )
        
        # Append DataArray to list:
        result_array_list.append(group_array)

    # Concatenate DataArrays contained in list along group dimension:
    result_array = xr.concat(result_array_list, dim=groups_name)

    # Return 2-dimensional binned statistics as DataArray:
    return result_array