##############################################################################
# find_utils.py
#
# Description:
# Defines functions for finding indices of attribute variables in trajectories
# objects.
#
# Last Edited:
# 2020/12/29
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.

import numpy as np
import pygeos

##############################################################################
# Define find_between() method.


def find_between(self, variable, min_val, max_val):
    """
    Finding indices of points within trajectories where a specified
    attribute variable takes a value between a specified min and
    max.

    Find returns the indices of trajectory points as a tuple of arrays
    where the specified attribute variable takes a value between a
    specified min and max (including these values).

    When variable is specified as 'time' only the indices of observations
    (obs) equal to and between the specified time-levels are returned for
    all trajectories.

    Parameters
    ----------
    variable : string
        Name of the variable in the trajectories object.
    min_val : numeric
        Minimum value variable should equal or be greater than.
    max_val : numeric
        Maximum value variable should equal or be less than.

    Returns
    -------
    tuple
        Indices of trajectory points where condition is satisfied.

    Examples
    --------
    Find indices of trajectory points where Latitude is between
    0 N - 20 N.

    >>> trajectories.find_between('lat', 0, 20)

    Finding indices of trajectory observations between two dates using
    numpy datetime64.

    >>> tmin = np.datetime64('2000-01-01')
    >>> tmax = np.datetime64('2000-03-01')
    >>> trajectories.find_between('time', tmin, tmax)

    Finding indices of trajectory observations with time using numpy
    timedelta64.

    >>> tmin = np.timedelta64(0, 'D')
    >>> tmax = np.timedelta64(90, 'D')
    >>> trajectories.find_between('time', tmin, tmax)
    """
    # -------------------
    # Raising exceptions.
    # -------------------
    # Defining list of variables contained in data.
    variables = list(self.data.variables)

    if isinstance(variable, str) is False:
        raise TypeError("variable must be specified as a string")

    if variable not in variables:
        raise ValueError("variable: \'" + variable + "\' not found in Dataset")

    # For non-time variables integers or floats only.
    if variable != 'time':
        if isinstance(min_val, (int, float)) is False:
            raise TypeError("min must be specified as integer or float")

        if isinstance(max_val, (int, float)) is False:
            raise TypeError("max must be specified as integer or float")
    # For time variable numpy datetime64 or timedelta64 format only.
    else:
        if isinstance(min_val, (np.datetime64, np.timedelta64)) is False:
            raise TypeError("min must be specified as datetime64 or timedelta64")

        if isinstance(max_val, (np.datetime64, np.timedelta64)) is False:
            raise TypeError("max must be specified as datetime64 or timedelta64")

    # ------------------------------------------------------------
    # Finding indices for attribute variable between min and max.
    # ------------------------------------------------------------

    # Defining indices where trajectory points meeting conditions
    # will be stored, use numpy vectorisation, np.where().
    indices = np.where((self.data[variable].values <= max_val) & (self.data[variable].values >= min_val))

    # Returning the tuple of arrays containing indices
    # for satisfactory trajectory points.
    return indices


##############################################################################
# Define find_equal() method.

def find_equal(self, variable, val):
    """
    Finding indices of points within trajectories where a specified
    attribute variable takes a value equal to a specified val.

    Find returns the indices of trajectory points as a tuple of arrays
    where the specified attribute variable takes a value equal to
    specified val.

    Parameters
    ----------
    variable : string
        Name of the variable in the trajectories object.
    val : numeric
        Value variable should equal.

    Returns
    -------
    tuple
        Indices of trajectory points where condition is satisfied.

    Examples
    --------
    Find indices of trajectory points where Latitude is equalt to
    0 N.

    >>> trajectories.find_equal('lat', 0)

    Finding indices of trajectory observations for one date using
    numpy datetime64.

    >>> tval = np.datetime64('2000-03-01')
    >>> trajectories.find_equal('time', tval)

    Finding indices of trajectory observations for one time using
    numpy timedelta64.

    >>> tval = np.timedelta64(90, 'D')
    >>> trajectories.find_equal('time', tval)
    """
    # -------------------
    # Raising exceptions.
    # -------------------
    # Defining list of variables contained in data.
    variables = list(self.data.variables)

    if isinstance(variable, str) is False:
        raise TypeError("variable must be specified as a string")

    if variable not in variables:
        raise ValueError("variable: \'" + variable + "\' not found in Dataset")

    # For non-time variables integers or floats only.
    if variable != 'time':
        if isinstance(val, (int, float)) is False:
            raise TypeError("val must be specified as integer or float")
    # For time variable numpy datetime64 or timedelta64 format only.
    else:
        if isinstance(val, (np.datetime64, np.timedelta64)) is False:
            raise TypeError("val must be specified as datetime64 or timedelta64")

    # -----------------------------------------------------
    # Finding indices for attribute variable equal to val.
    # -----------------------------------------------------

    # Defining indices where trajectory points meeting conditions
    # will be stored, use numpy vectorisation, np.where().
    indices = np.where(self.data[variable].values == val)

    # Returning the tuple of arrays containing indices
    # for satisfactory trajectory points.
    return indices


##############################################################################
# Define find_polygon() method.

def find_polygon(self, polygon):
    """
    Finding indices of trajectory points where a particle
    is contained within a specified polygon.

    Find returns the indices of trajectory points as a tuple of arrays.

    Parameters
    ----------
    variable : string
        Name of the variable in the trajectories object.
    polygon : list
    List of coordinates, specified as an ordered sequence of tuples
    (Lon, Lat), representing the boundary of the polygon.

    Returns
    -------
    tuple
        Indices of trajectory points where condition is satisfied.

    Examples
    --------
    Finding indices of trajectory points contained within a simple
    polygon, square.

    >>> square = [(-40, 30), (-40, 35), (-30, 35), (-30, 30), (-40, 30)]
    >>> trajectories.find_polygon(square)
    """
    # -------------------
    # Raising exceptions.
    # -------------------
    if isinstance(polygon, list) is False:
        raise TypeError("polygon must be specified as a list of lists")

    if len(polygon) > 1:
        if len(polygon) != np.shape(self.data['time'].values)[1]:
            raise ValueError("one polygon must be specified per observation (obs) when find is used with multiple polygons")

    # ------------------------------------------
    # Defining Latitude and Longitude variables.
    # ------------------------------------------
    Lon = np.copy(self.data['lon'].values)
    Lat = np.copy(self.data['lat'].values)

    # Defining number of trajectories.
    ntraj = np.shape(Lon)[0]
    # Defining number of observations.
    obs = np.shape(Lon)[1]

    # -------------------------------------------
    # Defining shapes from specified coordinates.
    # -------------------------------------------
    # Where multiple polygons are specified:
    if len(polygon) > 1:
        polygons = []
        # Iterate over polygons to create linearrings.
        for i in range(len(polygon)):
            shapes = list(shape for shape in polygon[i])
            polygons.append(pygeos.creation.linearrings(shapes))

        # Storing pygeos polygons, poly.
        poly = pygeos.creation.polygons(polygon)

    else:
        # Storing pygeos polygon, poly.
        poly = pygeos.creation.polygons(polygon)

    # ---------------------------------------------------------------
    # Using pygeos to find trajectory points intersecting a polygon.
    # ---------------------------------------------------------------
    # Setting NaNs in Lat and Lon to a missing data value (must not be
    # within range of coordinates -180E to 180E or -90N to 90N).
    Lon[np.isnan(Lon)] = -99999
    Lat[np.isnan(Lat)] = -99999

    # Configuiring masks as an empty array (dims - traj x obs) of
    # boolean type.
    mask = np.zeros([ntraj, obs], dtype='bool')

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

        # Defining mask_traj evaluating if every trajectory point intersects
        # the polygon.
        mask_traj = pygeos.predicates.contains(poly, trajectory_points)
        # Adding trajectory mask, mask_traj, to mask ndarray.
        mask[i, :] = mask_traj

    # --------------------------------------------------
    # Finding indices for points intersecting a polygon.
    # --------------------------------------------------
    # Defining indices where trajectory points meeting conditions
    # will be stored, use numpy vectorisation, np.where().
    indices = np.where(mask == True)

    # Returning the tuple of arrays containing indices
    # for satisfactory trajectory points.
    return indices
