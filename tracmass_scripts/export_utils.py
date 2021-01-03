##############################################################################
# export_utils.py
#
# Description:
# Defines functions to export TRACMASS output from .csv files to .nc.
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

##############################################################################
# Define get_start_time() function.


def get_start_time(self):
    """
    Returns times when particles are released (start of trajectory).

    The start time (ns) is the time elapsed since the begining of the
    simulation and is returned for all trajectories as a new ndarray.

    Parameters
    ----------
    self : DataSet
        DataSet composed of trajectory attribute variables stored as
        DataArrays.

    Returns
    -------
    start_time : ndarray
        release time (ns) of each particle, with dimension (traj).
    """
    # --------------------------------------------
    # Determining indices of starting time values.
    # --------------------------------------------
    # Defining Time variable with NaT values replaced with NaNs.
    Time = self.time.values
    Time = np.where(np.isnat(Time), np.nan, Time.astype(int))

    # Find indices of minimum values in time variable, start_ind.
    start_ind = np.nanargmin(Time, axis=1)

    # ----------------------------------------------------------
    # Using start_ind to determine start times for trajectories.
    # ----------------------------------------------------------
    # Using start_ind as the indices to determine start_time.
    # start_time is in timedelta64 format, unlike Time.
    start_time = np.take_along_axis(self.time.values, np.expand_dims(start_ind, axis=-1), axis=-1).squeeze(axis=-1)

    # Returning release times as ndarray, start_time.
    return start_time

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
    self : DataSet
        DataSet composed of trajectory attribute variables stored as
        DataArrays.

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
    seed_level = seed_level.astype('int')

    # Returning seed levels as ndarray, seed_level.
    return seed_level
