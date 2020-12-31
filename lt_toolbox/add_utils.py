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
from get_utils import get_start_time

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
    seed_level = seed_level.astype('int')

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
    traj_id = traj_id.astype('int')

    # Returning trajectory id as ndarray, traj_id.
    return traj_id
