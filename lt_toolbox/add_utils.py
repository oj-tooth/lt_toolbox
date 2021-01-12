##############################################################################
# add_utils.py
#
# Description:
# Defines functions to add new attribute variables
# to trajectories in a trajectories object.
#
# Last Edited:
# 2020/12/31
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.

import numpy as np
import xarray as xr
from .get_utils import get_start_time

##############################################################################
# Define add_seed() function.


def add_seed(self):
    """
    Returns seeding level when particles are released (start of trajectory).

    The seeding level, an integer between 1 and the total no. of seeding
    levels, marks when a particle is released into the system and is
    returned for all trajectories as a new ndarray.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.

    Returns
    -------
    seed_level : ndarray
        seeding level (integer) when each particle is released, with
        dimension (traj).
    """
    # ----------------------------------------------------------
    # Return start times for trajectories with get_start_time().
    # ----------------------------------------------------------
    start_time = get_start_time(self)

    # -------------------------------------------------
    # Determining particle seed_level from start_time.
    # -------------------------------------------------
    # Find the unique elements in start_time.
    uniq = np.unique(start_time)

    # Define seed_level as array of zeros with dimension (traj).
    seed_level = np.zeros(len(start_time))

    # Allocating common seed level for particles with equal
    # start_time values.
    # Initialising seed equal to 1.
    seed = 1
    for i in uniq:
        seed_level[start_time == i] = seed
        # Update seed.
        seed += 1

    # -------------------------------
    # Setting ouput array data type.
    # -------------------------------
    # Setting seed_level type to int.
    seed_level = seed_level.astype(np.int64)

    # Returning seed levels as ndarray, seed_level.
    return seed_level

##############################################################################
# Define add_id() function.


def add_id(self):
    """
    Returns unique identifier (integer) for each trajectory.

    The trajectory id, an integer between 1 and the total no. of
    trajectories, identifies every particle released into the system
    and is returned for all trajectories as a new ndarray.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.

    Returns
    -------
    traj_id : ndarray
        unique identifier (integer) for each particle released, with
        dimension (traj).
    """
    # -------------------------------------------------
    # Determining unique trajectory id from trajectory.
    # -------------------------------------------------
    # Defining trajectory variable from trajectories obj.
    trajectory = self.data.trajectory.values

    # Find the non-NaN unique elements in trajectory.
    traj_id = np.unique(trajectory[~np.isnan(trajectory)])

    # -------------------------------
    # Setting ouput array data type.
    # -------------------------------
    # Setting traj_id type to int.
    traj_id = traj_id.astype(np.int64)

    # Returning trajectory id as ndarray, traj_id.
    return traj_id

##############################################################################
# Define add_var() function.


def add_var(self, data, attrs):
    """
    Adds an new variable to the trajectory DataSet.

    The variable data must be provided as an ndarray with dimensions
    (traj) / (obs) / (traj x obs) and the attributes provided as a
    dictionary with keys - 'long_name', 'standard_name' and 'units' -
    as a minimum.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.

    Returns
    -------
    DataSet.
        Complete trajectories, including new attribute variable to
        be added.
    """
    # -------------------------------------------------
    # Determining dimensions of new attribute variable.
    # -------------------------------------------------
    # Determine no. dimenions of data.
    dims = np.ndim(data)

    # Determine size of traj and obs dims.
    traj = len(self.data.traj)
    obs = len(self.data.obs)

    # Defining variable name, var_name.
    var_name = attrs['standard_name']

    # -----------------------------------------
    # Adding 1-dimensional variable to DataSet.
    # -----------------------------------------
    if dims == 1:
        # Determine no. rows in data.
        rows = np.shape(data)[0]

        if rows == traj:
            # Append variable DataArray to original DataSet.
            self.data[var_name] = xr.DataArray(data, dims=["traj"])
            # Adding attributes to new variable in DataArray.
            self.data[var_name].attrs = attrs

        elif rows == obs:
            # Append variable DataArray to original DataSet.
            self.data[var_name] = xr.DataArray(data, dims=["obs"])
            # Adding attributes to new variable in DataArray.
            self.data[var_name].attrs = attrs

    # -----------------------------------------
    # Adding 2-dimensional variable to DataSet.
    # -----------------------------------------
    elif dims == 2:
        # Determine no. rows and cols in data.
        rows = np.shape(data)[0]
        cols = np.shape(data)[1]

        # Raising ValueError if array shape differs from (traj x obs).
        if rows != traj or cols != obs:
            raise ValueError("data should have dimensions traj x obs")

        # Append variable DataArray to original DataSet.
        self.data[var_name] = xr.DataArray(data, dims=["traj", "obs"])
        # Adding attributes to new variable in DataArray.
        self.data[var_name].attrs = attrs

    else:
        # -------------------------------------------------
        # Raise Error for incorrectly specified data array.
        # -------------------------------------------------
        # Raise ValueError if dimensions > 2.
        raise ValueError("data should be provided as a 1- or 2-dimensional ndarray")

    # ----------------------------------------------------
    # Returning DataSet with new attribute variable added.
    # ----------------------------------------------------
    return self.data
