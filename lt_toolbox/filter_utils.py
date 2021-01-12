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

##############################################################################
# Define filter_traj() function.


def filter_traj(self, filt_type, variable, val, min_val, max_val, drop):
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
    drop : boolean
        Determines if fitered trajectories should be returned as a
        new Dataset (False) or instead dropped from the existing
        Dataset (True).

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
            # Defining number of observations, obs.
            obs = np.shape(self.data[variable].values)[1]

            # Finding the minimum and maximum observations from specified
            # min_val and max_val.
            obs_min = np.where(self.data[variable].values[0, :] == min_val)[0]
            obs_max = np.where(self.data[variable].values[0, :] == max_val)[0]
            # Defining cols to contain indexes obs_min : obs_max.
            cols = np.arange(obs_min, obs_max + 1)

            # Returning the filtered trajectories as a subset of the original
            # DataSet.
            if drop is False:
                return self.data.isel(obs=xr.DataArray(cols, dims=["obs"]))

            # Where drop is True, remove filtered trajectories from original
            # Datset.
            else:
                # Defining cols to contain indexes not including obs_min :
                # obs_max.
                cols = np.concatenate([np.arange(obs_min), np.arange(obs_max + 1, obs)])

                return self.data.isel(obs=xr.DataArray(cols, dims=["obs"]))

        else:
            # -----------------------------------------------------------
            # Sub-Routine for filtering with attribute variables != time.
            # -----------------------------------------------------------

            if np.ndim(self.data[variable].values) > 1:
                # Defining rows as logical vector storing rows where
                # trajectories meeting conditions will be stored
                # Use numpy vectorisation, np.any().
                rows = np.any((self.data[variable].values <= max_val) & (self.data[variable].values >= min_val), axis=1)

            else:
                # For 1-dimensional array, use logical condition.
                rows = (self.data[variable].values <= max_val) & (self.data[variable].values >= min_val)

            # Returning the filtered trajectories as a subset of the original
            # DataSet.
            if drop is False:
                return self.data.isel(traj=xr.DataArray(rows, dims=["traj"]))

            # Where drop is True, remove filtered trajectories from original
            # Datset.
            else:
                return self.data.isel(traj=xr.DataArray(~rows, dims=["traj"]))

    # ----------------------------
    # Routine for filter_equal().
    # ----------------------------
    else:

        # --------------------------------
        # Routine for filtering with time.
        # --------------------------------
        if variable == 'time':
            # Defining number of observations, obs.
            obs = np.shape(self.data[variable].values)[1]

            # Finding the observations for a specified time.
            obs_equal = np.where(self.data[variable].values[0, :] == val)[0]

            # Defining cols to contain indexes obs_min : obs_max.
            cols = obs_equal

            # Returning the filtered trajectories as a subset of the original
            # DataSet.
            if drop is False:
                return self.data.isel(obs=xr.DataArray(cols, dims=["obs"]))

            # Where drop is True, remove filtered trajectories from original
            # Datset.
            else:
                # Defining cols to contain indexes not including obs_min :
                # obs_max.
                cols = np.concatenate([np.arange(obs_equal), np.arange(obs_equal + 1, obs)])

                return self.data.isel(obs=xr.DataArray(cols, dims=["obs"]))

        else:
            # -------------------------------------------------------
            # Routine for filtering with attribute variables != time.
            # -------------------------------------------------------

            if np.ndim(self.data[variable].values) > 1:
                # Defining rows as logical vector storing rows where
                # trajectories meeting conditions will be stored.
                # Uses numpy vectorisation, np.any().
                rows = np.any(self.data[variable].values == val, axis=1)

            else:
                # For 1-dimensional array, use logical condition.
                rows = self.data[variable].values == val

            # Returning the filtered trajectories as a subset of the original
            # DataSet.
            if drop is False:
                return self.data.isel(traj=xr.DataArray(rows, dims=["traj"]))

            # Where drop is True, remove filtered trajectories from original
            # Datset.
            else:
                return self.data.isel(traj=xr.DataArray(~rows, dims=["traj"]))
