##############################################################################
# get_utils.py
#
# Description:
# Defines functions to get exiting properties from
# trajectories in a trajectories object.
#
# Last Edited:
# 2020/12/24
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
    self : trajectories object
        Trajectories object passed from trajectories class method.

    Returns
    -------
    start_time : ndarray
        release time (ns) of each particle, with dimension (traj).
    """
    # --------------------------------------------
    # Determining indices of starting time values.
    # --------------------------------------------
    # Defining Time variable with NaT values replaced with NaNs.
    Time = self.data.time.values
    Time = np.where(np.isnat(Time), np.nan, Time.astype(int))

    # Find indices of minimum values in time variable, start_ind.
    start_ind = np.nanargmin(Time, axis=1)

    # ----------------------------------------------------------
    # Using start_ind to determine start times for trajectories.
    # ----------------------------------------------------------
    # Using start_ind as the indices to determine start_time.
    # start_time is in timedelta64 format, unlike Time.
    start_time = np.take_along_axis(self.data.time.values, np.expand_dims(start_ind, axis=-1), axis=-1).squeeze(axis=-1)

    # Returning release times as ndarray, start_time.
    return start_time

##############################################################################
# Define get_start_loc() function.


def get_start_loc(self):
    """
    Returns locations where particles are released (start of trajectory).

    The start location is divided into start_lon, start_lat and start_z
    which are returned for all trajectories as new ndarrays.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.

    Returns
    -------
    start_lat: ndarray
        latitude of each particle at the time of release, with
        dimension (traj).

    start_lon: ndarray
        longitude of each particle at the time of release, with
        dimension (traj).

    start_z: ndarray
        depth of each particle at the time of release, with
        dimension (traj).
    """
    # --------------------------------------------
    # Determining indices of starting time values.
    # --------------------------------------------
    # Defining Time variable with NaT values replaced with NaNs.
    Time = self.data.time.values
    Time = np.where(np.isnat(Time), np.nan, Time.astype(int))

    # Find indices of minimum values in time variable, start_ind.
    start_ind = np.nanargmin(Time, axis=1)

    # -----------------------------------------------------------------
    # Using start_ind to determine starting locations for trajectories.
    # -----------------------------------------------------------------
    # Using start_ind as the indices to determine start_lat.
    start_lat = np.take_along_axis(self.data.lat.values, np.expand_dims(start_ind, axis=-1), axis=-1).squeeze(axis=-1)
    # Using start_ind as the indices to determine start_lon.
    start_lon = np.take_along_axis(self.data.lon.values, np.expand_dims(start_ind, axis=-1), axis=-1).squeeze(axis=-1)
    # Using start_ind as the indices to determine start_z.
    start_z = np.take_along_axis(self.data.z.values, np.expand_dims(start_ind, axis=-1), axis=-1).squeeze(axis=-1)

    # Returning starting latitudes, longutudes and depths as ndarrays,
    # start_lat, start_lon and start_z.
    return start_lat, start_lon, start_z

##############################################################################
# Define get_end_time() function.


def get_end_time(self):
    """
    Returns times when particles exit the system (end of trajectory).

    The end time (ns) is given as the time elapsed since the begining
    of the simulation and is returned for all trajectories as an ndarray.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.

    Returns
    -------
    end_time : ndarray
        exit time (ns) of each particle, with dimension (traj).
    """
    # ---------------------------------------
    # Determining indices of end time values.
    # ---------------------------------------
    # Defining Time variable with NaT values replaced with NaNs.
    Time = self.data.time.values
    Time = np.where(np.isnat(Time), np.nan, Time.astype(int))

    # Find indices of maximum values in time variable, end_ind.
    end_ind = np.nanargmax(Time, axis=1)

    # ----------------------------------------------------------
    # Using end_ind to determine end times for trajectories.
    # ----------------------------------------------------------
    # Using end_ind as the indices to determine end_time.
    # end_time is in timedelta64 format, unlike Time.
    end_time = np.take_along_axis(self.data.time.values, np.expand_dims(end_ind, axis=-1), axis=-1).squeeze(axis=-1)

    # Returning exit times as np.array, end_time.
    return end_time

##############################################################################
# Define get_end_loc() function.


def get_end_loc(self):
    """
    Returns location when particles exit system (end of trajectory).

    The end location is divided into end_lon and end_lat which are
    returned for all trajectories as new ndarrays.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.

    Returns
    -------
    end_lat: ndarray
        latitude of each particle at the time of exit, with
        dimension (traj).

    end_lon: ndarray
        longitude of each particle at the time of exit, with
        dimension (traj).

    end_z: ndarray
        depth of each particle at the time of release, with
        dimension (traj).
    """
    # -------------------------------------------
    # Determining indices of ending time values.
    # -------------------------------------------
    # Defining Time variable with NaT values replaced with NaNs.
    Time = self.data.time.values
    Time = np.where(np.isnat(Time), np.nan, Time.astype(int))

    # Find indices of maximum values in time variable, end_ind.
    end_ind = np.nanargmax(Time, axis=1)

    # ------------------------------------------------------------
    # Using end_ind to determine final locations for trajectories.
    # ------------------------------------------------------------
    # Using end_ind as the indices to determine end_lat.
    end_lat = np.take_along_axis(self.data.lat.values, np.expand_dims(end_ind, axis=-1), axis=-1).squeeze(axis=-1)
    # Using end_ind as the indices to determine end_lon.
    end_lon = np.take_along_axis(self.data.lon.values, np.expand_dims(end_ind, axis=-1), axis=-1).squeeze(axis=-1)
    # Using end_ind as the indices to determine end_z.
    end_z = np.take_along_axis(self.data.z.values, np.expand_dims(end_ind, axis=-1), axis=-1).squeeze(axis=-1)

    # Returning final latitudes, longitudes and depths as ndarrays,
    # end_lat, end_lon and end_z.
    return end_lat, end_lon, end_z

##############################################################################
# Define get_duration() function.


def get_duration(self):
    """
    Returns time taken for particles to exit the system following release
    (duration of trajectory).

    The duration (ns) is given as the time elapsed from the release of a
    particle to it's exiting from the sytem and is returned for all
    trajectories as an ndarray.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.

    Returns
    -------
    duration : ndarray
        duration (ns) each particle remains in the sytem, with
        dimension (traj).
    """
    # ------------------------------
    # Returning start and end times.
    # ------------------------------
    start_time = get_start_time(self)
    end_time = get_end_time(self)

    # ------------------------------------
    # Computing duration for trajectories.
    # ------------------------------------
    # Computing the time duration a particle remains in the system for,
    # as the difference between end (exit) and start (release) times.
    duration = end_time - start_time

    # Returning duration as np.array, duration.
    return duration

##############################################################################
# Define get_minmax() function.


def get_minmax(self, variable, get_max=True):
    """
    Returns maximum (default) or minimum value of a specified variable
    for each trajectory.

    The maximum or minimum value of the variable is returned for all
    trajectories as an ndarray.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    variable : string
        Name of the variable in the trajectories object.
    get_max : logical
        Get the maximum values of a specified variable, default is
        True.

    Returns
    -------
    values : ndarray
        maximum (default) or minimum values of specified variable for
        each trajectory, with dimension (traj).
    """
    # ---------------------------
    # Determining maximum values.
    # ---------------------------
    if get_max is True:
        # Determining max value of variable for each traj (row).
        values = np.nanmax(self.data[variable].values, axis=1)

    # ---------------------------
    # Determining minimum values.
    # ---------------------------
    else:
        # Determining min values of variable for each traj (row).
        values = np.nanmin(self.data[variable].values, axis=1)

    # Returning max or min values as np.array, values.
    return values

##############################################################################
# Define get_val() function.


def get_val(self, variable, time_level):
    """
    Returns the value of a specified variable at a specified
    time level for each trajectory.

    The values of the specified variable are returned for all
    trajectories for a time level specified in the form
    'YYYY-MM-DD' as an ndarray.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    variable : string
        Name of the variable in the trajectories object.
    time_level : string
        Time level from which to get the values of variable.

    Returns
    -------
    values : ndarray
        values of specified variable at the specified time level
        for each trajectory, with dimension (traj).
    """
    # ------------------------------------------------------
    # Determining indices of observations at specified time.
    # ------------------------------------------------------
    obs_equal = (self.data.time.values[0, :] == np.datetime64(time_level))

    # -----------------------------------------------------
    # Determining the values of variable stored at obs-ind.
    # -----------------------------------------------------
    values = self.data[variable].values[:, obs_equal]

    # Removes single-dimensional entries from values array.
    values = np.squeeze(values)

    # Returning max or min values as np.array, values.
    return values
