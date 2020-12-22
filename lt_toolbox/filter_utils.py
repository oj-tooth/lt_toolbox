##############################################################################
# filter_utils.py
#
# Description:
# Defines function for filtering trajectories objects.
#
# Last Edited:
# 2020/12/22
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.

import numpy as np
import xarray as xr
import pandas as pd

##############################################################################
# Define filter_traj() function.


def filter_traj(self, filt_type, variable, val='NaN', min_val='NaN', max_val='NaN'):
    """
    Filter trajectories using attribute variable.

    Filtering returns the complete trajectories where the specified
    attribute variable takes a value between a specified min and max
    (including these values) or value equal to val.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    filt_type : string
        Name of filter method to be used - 'between' or 'equal'.
    variable : string
        Name of the variable in the trajectories object.
    val : numeric
        Value variable should equal (using 'equal' method).
    min_val : numeric
        Minimum value variable should equal or be greater than
        (using 'between' method).
    max_val : numeric
        Maximum value variable should equal or be less than
        (using 'between' method).

    Returns
    -------
    DataSet.
        Complete trajectories, including all attribute variables,
        which meet the filter specification.
    """

    # -----------------------------
    # Routine for filter_between().
    # -----------------------------
    if filt_type == 'between':

        # ------------------------------------
        # Sub-Routine for filtering with time.
        # ------------------------------------
        if variable == 'time':
            # Finding the minimum and maximum observations from specified min_val and max_val.
            obs_min = np.where(self.data[variable].values[0, :] == pd.to_timedelta(min_val, unit='s'))[0]
            obs_max = np.where(self.data[variable].values[0, :] == pd.to_timedelta(max_val, unit='s'))[0]

            # Returning the subseteed xarray DataSet.
            return self.data.isel(obs=xr.DataArray(np.arange(obs_min, obs_max + 1), dims=["obs"]))

        else:
            # -----------------------------------------------------------
            # Sub-Routine for filtering with attribute variables != time.
            # -----------------------------------------------------------

            # Defining rows as logical vector storing rows where trajectories
            # meeting conditions will be stored
            # Use numpy vectorisation, np.any().
            rows = np.any((self.data[variable].values <= max_val) & (self.data[variable].values >= min_val), axis=1)

            # Returning the subsetted xarray DataSet.
            return self.data.isel(traj=xr.DataArray(rows, dims=["traj"]))

    # ----------------------------
    # Routine for filter_equal().
    # ----------------------------
    else:

        # --------------------------------
        # Routine for filtering with time.
        # --------------------------------
        if variable == 'time':
            # Finding the observations for a specified time.
            obs_equal = np.where(self.data[variable].values[0, :] == pd.to_timedelta(val, unit='s'))[0]

            # Returning the sub-setted xarray DataSet.
            return self.data.isel(obs=xr.DataArray(obs_equal + 1, dims=["obs"]))

        else:
            # -------------------------------------------------------
            # Routine for filtering with attribute variables != time.
            # -------------------------------------------------------

            # Defining rows as logical vector storing rows where trajectories
            # meeting conditions will be stored
            # Uses numpy vectorisation, np.any().
            rows = np.any(self.data[variable].values == val, axis=1)

            # Returning the subsetted xarray DataSet.
            return self.data.isel(traj=xr.DataArray(rows, dims=["traj"]))
