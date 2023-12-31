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
# Define binned_statistic_1d() function.


def binned_statistic_1d(df:pl.DataFrame | pl.LazyFrame, var:str, values:str, statistic:str, bin_breaks:list) -> xr.DataArray:
    """
    Compute a 1-dimensional binned statistic using the Series stored in a
    DataFrame.

    This is a generalization of a histogram function. A histogram divides
    the chosen column varaible into bins, and returns the count of the number
    of points in each bin. This function allows the computation of the sum,
    mean, median, or other statistic of the values within each bin.

    Parameters
    ----------
    df : DataFrame | LazyFrame
        DataFrame or LazyFrame containing variable and values columns.

    var : str
        Name of variable whose values will be binned.

    values : str
        Name of values over which the statistic will be computed.

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
                 .select(
                     pl.col(values),
                     pl.col(var).cut(breaks=bin_breaks, labels=bin_labels).alias('values_binned')
                     )
                 )

    # --- Calculate Statistic in Discrete Bins ---
    # Evaluate statistic over values stored in each bin:
    if statistic == 'mean':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values).mean())
                        )
    elif statistic == 'median':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values).median())
                        )
    elif statistic == 'count':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values).count())
                        )
    elif statistic == 'sum':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values).sum())
                        )
    elif statistic == 'std':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values).std())
                        )
    elif statistic == 'min':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values).min())
                        )
    elif statistic == 'max':
        result = (df_binned
                        .group_by(pl.col('values_binned'), maintain_order=True)
                        .agg(pl.col(values).max())
                        )

    # Cast value bins dtype from categorical to float:
    result = result.with_columns(pl.col('values_binned').cast(pl.Utf8).cast(pl.Float64))
    # Sort DataFrame by bin value in ascending order:
    result = result.sort(by='values_binned', descending=False)

    # Collect the result if input df is a LazyFrame:
    if isinstance(result, pl.LazyFrame):
        result = result.collect(streaming=True)

    # --- Transform Binned Statistic to DataArray ---
    # Construct xarray DataArray from Polars Series:
    result_array = xr.DataArray(data=result[values].to_numpy(),
                                dims=[var],
                                coords={
                                    var:([var], result['values_binned'].to_numpy().astype(np.float64))
                                },
                                name=values
                                )

    # Return 1-dimensional binned statistics as DataArray:
    return result_array

##############################################################################
# Define binned_group_statistic_1d() function.


def binned_group_statistic_1d(df:pl.DataFrame, var:str, values:str, groups:str, statistic:str, bin_breaks:list) -> xr.DataArray:
    """
    Compute a grouped  1-dimensional binned statistic using the Series stored
    in a DataFrame.

    This is a generalization of a histogram function. A histogram divides
    the chosen column varaible into bins, and returns the count of the number
    of points in each bin. This function allows the computation of the sum,
    mean, median, or other statistic of the values within each bin.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing variable and values columns.

    var : str
        Name of variable whose values will be binned.

    values : str
        Name of values over which the statistic will be computed.

    groups : str
        Name of values to grouped according to unique
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
                 .select(
                     pl.col(groups),
                     pl.col(values),
                     pl.col(var).cut(breaks=bin_breaks, labels=bin_labels).alias('values_binned'),
                     )
                 )

    # --- Calculate Statistic in Discrete Bins ---
    # Initialise empty list to store results:
    result_array_list = []
    # Iterate over DataFrame groups:
    for group_n, df_group in df_binned.group_by(by=groups, maintain_order=True):
        # Evaluate statistic over values stored in each bin:
        if statistic == 'mean':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).mean())
                            )
        elif statistic == 'median':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).median())
                            )
        elif statistic == 'count':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).count())
                            )
        elif statistic == 'sum':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).sum())
                            )
        elif statistic == 'std':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).std())
                            )
        elif statistic == 'min':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).min())
                            )
        elif statistic == 'max':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).max())
                            )

        # Cast value bins dtype from categorical to float:
        result = result.with_columns(pl.col('values_binned').cast(pl.Utf8).cast(pl.Float64))
        # Sort DataFrame by bin value in ascending order:
        result = result.sort(by='values_binned', descending=False)

        # Collect the result if input df is a LazyFrame:
        if isinstance(result, pl.LazyFrame):
            result = result.collect(streaming=True)

        # --- Transform Binned Statistic to DataArray ---
        # Construct xarray DataArray from Polars Series:
        group_array = xr.DataArray(data=result[values].to_numpy()[np.newaxis, :],
                                    dims=[groups, var],
                                    coords={
                                        groups:([groups], np.array([group_n])),
                                        var:([var], result['values_binned'].to_numpy().astype(np.float64))
                                    },
                                    name=values
                                    )
        # Append DataArray to list:
        result_array_list.append(group_array)

    # Concatenate DataArrays contained in list along group dimension:
    result_array = xr.concat(result_array_list, dim=groups)

    # Return 1-dimensional binned statistics as DataArray:
    return result_array

##############################################################################
# Define binned_lazy_group_statistic_1d() function.


def binned_lazy_group_statistic_1d(ldf:pl.LazyFrame, var:str, values:str, groups:str, statistic:str, bin_breaks:list) -> xr.DataArray:
    """
    Compute a grouped 1-dimensional binned statistic using the Series stored
    in a LazyFrame.

    This is a generalization of a histogram function. A histogram divides
    the chosen column varaible into bins, and returns the count of the number
    of points in each bin. This function allows the computation of the sum,
    mean, median, or other statistic of the values within each bin.

    Parameters
    ----------
    ldf : LazyFrame
        LazyFrame containing variable and values columns.

    var : str
        Name of variable whose values will be binned.

    values : str
        Name of values over which the statistic will be computed.

    groups : str
        Name of values to grouped according to unique
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
    df_binned = (ldf
                 .select(
                     pl.col(groups),
                     pl.col(values),
                     pl.col(var).cut(breaks=bin_breaks, labels=bin_labels).alias('values_binned'),
                     )
                 )
    
    # --- Group Members ---
    # Determine unique members of group column:
    grp_vals = (ldf
                .select(pl.col(groups).unique())
                .collect(streaming=True)[groups]
                .to_list()
                )

    # --- Calculate Statistic in Discrete Bins ---
    # Initialise empty list to store results:
    result_array_list = []
    # Iterate over LazyFrame groups:
    for val in grp_vals:
        df_group = (df_binned
                    .filter(pl.col(groups) == val)
                    )
        # Evaluate statistic over values stored in each bin:
        if statistic == 'mean':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).mean())
                            )
        elif statistic == 'median':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).median())
                            )
        elif statistic == 'count':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).count())
                            )
        elif statistic == 'sum':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).sum())
                            )
        elif statistic == 'std':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).std())
                            )
        elif statistic == 'min':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).min())
                            )
        elif statistic == 'max':
            result = (df_group
                            .group_by(pl.col('values_binned'), maintain_order=True)
                            .agg(pl.col(values).max())
                            )

        # Cast value bins dtype from categorical to float:
        result = result.with_columns(pl.col('values_binned').cast(pl.Utf8).cast(pl.Float64))
        # Sort DataFrame by bin value in ascending order:
        result = result.sort(by='values_binned', descending=False)

        # Collect the result with streaming:
        result = result.collect(streaming=True)

        # --- Transform Binned Statistic to DataArray ---
        # Construct xarray DataArray from Polars Series:
        group_array = xr.DataArray(data=result[values].to_numpy()[np.newaxis, :],
                                    dims=[groups, var],
                                    coords={
                                        groups:([groups], np.array([val])),
                                        var:([var], result['values_binned'].to_numpy().astype(np.float64))
                                    },
                                    name=values
                                    )
        # Append DataArray to list:
        result_array_list.append(group_array)

    # Concatenate DataArrays contained in list along group dimension:
    result_array = xr.concat(result_array_list, dim=groups)

    # Return 1-dimensional binned statistics as DataArray:
    return result_array

##############################################################################
# Define binned_statistic_2d() function.


def binned_statistic_2d(df:pl.DataFrame | pl.LazyFrame, var_x:str, var_y:str, values:str, statistic:str, bin_breaks:list, drop_duplicates:bool=False) -> xr.DataArray:
    """
    Compute a 2-dimensional binned statistic using the Series stored in a
    DataFrame.

    This is a generalization of a 2-D histogram function. A 2-D histogram divides
    the chosen column varaibles into bins, and returns the count of the number
    of points in each bin. This function allows the computation of the sum,
    mean, median, or other statistic of the values within each bin.

    Parameters
    ----------
    df : DataFrame | LazyFrame
        DataFrame or LazyFrame containing variable and values columns.

    var_x : string
        Name of values to be binned along the first dimension.

    var_y : Series
        Name of values to be binned along the second dimension.

    values : Series
        Name of values over which the statistic will be computed.
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

    drop_duplicates: bool
        Remove duplicate Lagrangian trajectory occurences in each
        bin before computing specified statistic. The default is False.

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
                 .select(
                     pl.col('id'),
                     pl.col(values),
                     pl.col(var_x).cut(breaks=bin_x_breaks, labels=bin_x_labels).alias('values_x_binned'),
                     pl.col(var_y).cut(breaks=bin_y_breaks, labels=bin_y_labels).alias('values_y_binned'),
                     )
                 )

    # Drop duplicate Lagrangian trajectory occurences in each bin:
    if drop_duplicates:
        df_binned = (df_binned
                     .unique(subset=['id', 'values_x_binned', 'values_y_binned'],
                             maintain_order=True
                             )
                    )

    # --- Calculate Statistic in Discrete Bins ---
    # Evaluate statistic over values stored in each bin:
    if statistic == 'mean':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values).mean())
                        )
    elif statistic == 'median':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values).median())
                        )
    elif statistic == 'count':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values).count())
                        )
    elif statistic == 'sum':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values).sum())
                        )
    elif statistic == 'std':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values).std())
                        )
    elif statistic == 'min':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values).min())
                        )
    elif statistic == 'max':
        result = (df_binned
                        .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                        .agg(pl.col(values).max())
                        )

    # Cast value bins dtypes from categorical to float:
    result = (result
            .with_columns(
                pl.col('values_x_binned').cast(pl.Utf8).cast(pl.Float64),
                pl.col('values_y_binned').cast(pl.Utf8).cast(pl.Float64),
                )
            )
    # Sort Data/LazyFrame by bin value in ascending order:
    result = result.sort(by='values_x_binned', descending=False)

    # If LazyFrame collect result with streaming:
    if isinstance(result, pl.LazyFrame):
        result = result.collect(streaming=True)

    # --- Pivot 2-D Statistic DataFrame ---
    # Pivot DataFrame to store statistic values in form:
    # [variable x-bins -> rows] x [variable y-bins -> columns].
    result_pivot = result.pivot(values=values, index='values_x_binned', columns='values_y_binned')

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
                                dims=[var_x, var_y],
                                coords={
                                    var_x:([var_x], var_x_array),
                                    var_y:([var_y], var_y_array),
                                },
                                name=values
                                )

    # Return 2-dimensional binned statistics as DataArray:
    return result_array

##############################################################################
# Define binned_group_statistic_2d() function.


def binned_group_statistic_2d(df:pl.DataFrame, var_x:str, var_y:str, values:str, groups:str, statistic:str, bin_breaks:list, drop_duplicates:bool=False) -> xr.DataArray:
    """
    Compute a 2-dimensional grouped binned statistic using the Series stored in a
    DataFrame.

    This is a generalization of a 2-D histogram function. A 2-D histogram divides
    the chosen column varaibles into bins, and returns the count of the number
    of points in each bin. This function allows the computation of the sum,
    mean, median, or other statistic of the values within each bin.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing variable and values columns.

    var_x : str
        Name of values to be binned along the first dimension.

    var_y : str
        Name of values to be binned along the second dimension.

    values : str
        Name of values over which the statistic will be computed.
        This must be the same length as var_x & var_y.

    groups : str
        Name of values to grouped according to unique
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

    drop_duplicates: bool
        Remove duplicate Lagrangian trajectory occurences in each
        bin before computing specified statistic. The default is False.

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
                 .select(
                     pl.col('id'),
                     pl.col(groups),
                     pl.col(values),
                     pl.col(var_x).cut(breaks=bin_x_breaks, labels=bin_x_labels).alias('values_x_binned'),
                     pl.col(var_y).cut(breaks=bin_y_breaks, labels=bin_y_labels).alias('values_y_binned'),
                     )
                 )

    # Drop duplicate Lagrangian trajectory occurences in each bin:
    if drop_duplicates:
        df_binned = (df_binned
                     .unique(subset=['id', 'values_x_binned', 'values_y_binned'],
                             maintain_order=True
                             )
                    )

    # --- Calculate Statistic in Discrete Bins ---
    # Initialise empty list to store results:
    result_array_list = []
    # Iterate over DataFrame groups:
    for group, df_group in df_binned.group_by(by=groups, maintain_order=True):
        # Evaluate statistic over values stored in each bin:
        if statistic == 'mean':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).mean())
                            )
        elif statistic == 'median':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).median())
                            )
        elif statistic == 'count':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).count())
                            )
        elif statistic == 'sum':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).sum())
                            )
        elif statistic == 'std':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).std())
                            )
        elif statistic == 'min':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).min())
                            )
        elif statistic == 'max':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).max())
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
        result_pivot = result.pivot(values=values, index='values_x_binned', columns='values_y_binned')

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
                                    dims=[groups, var_x, var_y],
                                    coords={
                                        groups:([groups], np.array([group])),
                                        var_x:([var_x], var_x_array),
                                        var_y:([var_y], var_y_array),
                                    },
                                    name=values
                                    )

        # Append DataArray to list:
        result_array_list.append(group_array)

    # Concatenate DataArrays contained in list along group dimension:
    result_array = xr.concat(result_array_list, dim=groups)

    # Return 2-dimensional binned statistics as DataArray:
    return result_array

##############################################################################
# Define binned_lazy_group_statistic_2d() function.


def binned_lazy_group_statistic_2d(ldf:pl.LazyFrame, var_x:str, var_y:str, values:str, groups:str, statistic:str, bin_breaks:list, drop_duplicates:bool=False) -> xr.DataArray:
    """
    Compute a 2-dimensional grouped binned statistic using the Series stored in a
    LazyFrame.

    This is a generalization of a 2-D histogram function. A 2-D histogram divides
    the chosen column varaibles into bins, and returns the count of the number
    of points in each bin. This function allows the computation of the sum,
    mean, median, or other statistic of the values within each bin.

    Parameters
    ----------
    ldf : LazyFrame
        LazyFrame containing variable and values columns.

    var_x : str
        Name of values to be binned along the first dimension.

    var_y : str
        Name of values to be binned along the second dimension.

    values : str
        Name of values over which the statistic will be computed.
        This must be the same length as var_x & var_y.

    groups : str
        Name of values to grouped according to unique
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

    drop_duplicates: bool
        Remove duplicate Lagrangian trajectory occurences in each bin before
        computing specified statistic. The default is False.

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
    df_binned = (ldf
                 .select(
                     pl.col('id'),
                     pl.col(groups),
                     pl.col(values),
                     pl.col(var_x).cut(breaks=bin_x_breaks, labels=bin_x_labels).alias('values_x_binned'),
                     pl.col(var_y).cut(breaks=bin_y_breaks, labels=bin_y_labels).alias('values_y_binned'),
                     )
                 )

    # Drop duplicate Lagrangian trajectory occurences in each bin:
    if drop_duplicates:
        df_binned = (df_binned
                     .unique(subset=['id', 'values_x_binned', 'values_y_binned'],
                             maintain_order=True
                             )
                    )

    # --- Group Members ---
    # Determine unique members of group column:
    grp_vals = (ldf
                .select(pl.col(groups).unique())
                .collect(streaming=True)[groups]
                .to_list()
                )

    # --- Calculate Statistic in Discrete Bins ---
    # Initialise empty list to store results:
    result_array_list = []
    # Iterate over LazyFrame groups:
    for val in grp_vals:
        df_group = (df_binned
                    .filter(pl.col(groups) == val)
                    )
        # Evaluate statistic over values stored in each bin:
        if statistic == 'mean':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).mean())
                            )
        elif statistic == 'median':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).median())
                            )
        elif statistic == 'count':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).count())
                            )
        elif statistic == 'sum':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).sum())
                            )
        elif statistic == 'std':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).std())
                            )
        elif statistic == 'min':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).min())
                            )
        elif statistic == 'max':
            result = (df_group
                            .group_by(pl.col('values_x_binned', 'values_y_binned'), maintain_order=True)
                            .agg(pl.col(values).max())
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
        result_pivot = result.pivot(values=values, index='values_x_binned', columns='values_y_binned')

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
                                    dims=[groups, var_x, var_y],
                                    coords={
                                        groups:([groups], np.array([val])),
                                        var_x:([var_x], var_x_array),
                                        var_y:([var_y], var_y_array),
                                    },
                                    name=values
                                    )

        # Append DataArray to list:
        result_array_list.append(group_array)

    # Concatenate DataArrays contained in list along group dimension:
    result_array = xr.concat(result_array_list, dim=groups)

    # Return 2-dimensional binned statistics as DataArray:
    return result_array
