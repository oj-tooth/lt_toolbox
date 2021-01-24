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
import pygeos

##############################################################################
# Define filter_traj_between() function.

def filter_traj_between(self, variable, min_val, max_val, drop):
    """
    Filter trajectories between two values of an attribute variable.

    Filtering returns the complete trajectories where the specified
    attribute variable takes a value between a specified min
    and max (including these values).

    When variable is specified as 'time' only the observations (obs)
    equal to and between the specified time-levels are returned for all
    trajectories.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    variable : string
        Name of the variable in the trajectories object.
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

    # -----------------------------------------------------------
    # Sub-Routine for filtering with attribute variables != time.
    # -----------------------------------------------------------
    else:
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


##############################################################################
# Define filter_traj_equal() function.


def filter_traj_equal(self, variable, val, drop):
    """
    Filter trajectories with attribute variable equal to value.

    Filtering returns the complete trajectories where the specified
    attribute variable takes the value specified by val.

    When variable is specified as 'time' only the observations (obs)
    equal to and between the specified time-levels are returned for
    all trajectories.

    Parameters
    ----------
    variable : string
        Name of the variable in the trajectories object.
    val : numeric
        Value variable should equal.
    drop : boolean
        Determines if fitered trajectories should be returned as a
        new trajectories object (False) or instead dropped from the
        existing trajectories object (True).

    Returns
    -------
    DataSet.
        Complete trajectories, including all attribute variables,
        which meet the filter specification.
    """
    # --------------------------------
    # Routine for filtering with time.
    # --------------------------------
    if variable == 'time':

        # Finding the observations for a specified time.
        # obs_equal = np.where(self.data[variable].values[0, :] == val)[0]
        obs_equal = np.isin(self.data[variable].values[0, :], val)

        # Defining cols to contain indexes obs_min : obs_max.
        # cols = obs_equal

        # Returning the filtered trajectories as a subset of the original
        # DataSet.
        if drop is False:
            return self.data.isel(obs=xr.DataArray(obs_equal, dims=["obs"]))

        # Where drop is True, remove filtered trajectories from original
        # Datset.
        else:
            return self.data.isel(obs=xr.DataArray(~obs_equal, dims=["obs"]))

    else:
        # -------------------------------------------------------
        # Routine for filtering with attribute variables != time.
        # -------------------------------------------------------

        if np.ndim(self.data[variable].values) > 1:
            # Defining rows as logical vector storing rows where
            # trajectories meeting conditions will be stored.
            # Uses numpy vectorisation, np.any().
            # self.data[variable].values == val
            rows = np.any(np.isin(self.data[variable].values, val), axis=1)

        else:
            # For 1-dimensional array, use logical condition.
            rows = np.isin(self.data[variable].values, val)

        # Returning the filtered trajectories as a subset of the original
        # DataSet.
        if drop is False:
            return self.data.isel(traj=xr.DataArray(rows, dims=["traj"]))

        # Where drop is True, remove filtered trajectories from original
        # Datset.
        else:
            return self.data.isel(traj=xr.DataArray(~rows, dims=["traj"]))


##############################################################################
# Define filter_traj_polygon() function.

def filter_traj_polygon(self, polygon, method, drop):
    """
    Filter trajectories which intersect a specified polygon.

    Filtering returns the complete trajectories of particles
    which have been inside the boundary of a given polygon at
    any point in their lifetime.

    Parameters
    ----------
    polygon : list
        List of coordinates, specified as an ordered sequence of tuples
        (Lon, Lat), representing the boundary of the polygon.
    method : string
        Method to filter trajectories using polygon - 'pos' considers
        trajectories as a collection of points, 'traj' considers
        trajectories as connected lines.
    drop : boolean
        Determines if fitered trajectories should be returned as a
        new trajectories object (False) or instead dropped from the
        existing trajectories object (True).

    Returns
    -------
    DataSet.
        Complete trajectories, including all attribute variables,
        which meet the filter specification.
    """
    # ------------------------------------------
    # Defining Latitude and Longitude variables.
    # ------------------------------------------
    Lon = np.copy(self.data['lon'].values)
    Lat = np.copy(self.data['lat'].values)

    # Defining number of trajectories.
    ntraj = np.shape(Lon)[0]

    # -------------------------------------------
    # Defining shapes from specified coordinates.
    # -------------------------------------------
    # Storing pygeos polygon, poly.
    poly = pygeos.creation.polygons(polygon)

    # ------------------------------------------------------------
    # Using pygeos to filter trajectories intersecting a polygon.
    # ------------------------------------------------------------
    # Configuiring rows as an array of boolean type (dimensions = traj).
    rows = np.zeros(ntraj, dtype='bool')

    # Setting NaNs in Lat and Lon to a missing data value (must not be
    # within range of coordinates -180E to 180E or -90N to 90N).
    Lon[np.isnan(Lon)] = -99999
    Lat[np.isnan(Lat)] = -99999

    # ------------------------------------------------
    # Routine for filtering with collection of points.
    # ------------------------------------------------
    if method == 'pos':

        # Iterating over all trajectories and defining a Point Collection
        # with trajectory points.
        for i in np.arange(ntraj):
            # Storing Lats and Lons for trajectory i.
            lon = Lon[i, :]
            lat = Lat[i, :]

            # Defining coordinates as transposed tuple of Lats and Lons.
            coords = np.array((lon, lat)).T

            # Defining points as a pygeos point collection.
            trajectory_points = pygeos.creation.points(coords)

            # Defining mask evaluating if every trajectory point is within
            # the polygon.
            mask = pygeos.predicates.contains(poly, trajectory_points)

            # Determining (boolean) if any point in trajectory is within a
            # given polygon.
            rows[i] = np.any(mask)

    elif method == 'traj':

        # Store lat and lon coordinates as list of tuples.
        lon = list(map(tuple, Lon))
        lat = list(map(tuple, Lat))

        # Defining lines as a pygeos linestrings collection.
        trajectory_lines = pygeos.creation.linestrings(lon, lat)

        # Determining (boolean) evaluating if every trajectory line
        # intersects the polygon.
        rows = pygeos.predicates.intersects(trajectory_lines, poly)

    # Returning the filtered trajectories as a subset of the original
    # DataSet.
    if drop is False:
        return self.data.isel(traj=xr.DataArray(rows, dims=["traj"]))

    # Where drop is True, remove filtered trajectories from original
    # Datset.
    else:
        return self.data.isel(traj=xr.DataArray(~rows, dims=["traj"]))
