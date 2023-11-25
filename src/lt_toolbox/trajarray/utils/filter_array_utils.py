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
# Define filter_traj() function.

def filter_traj(self, variable, operator, value, drop):
    """
    Filter trajectories using conditional on an attribute variable
    specified with a string expression.

    Filtering returns a reduced Dataset where only the
    trajectories meeting the specified condition are retained.
    The exception is when users specify drop=True, in which case
    trajectories meeting the specified condition are dropped from the
    Dataset.

    Parameters
    ----------
    self : TrajArray
        TrajArray containing Dataset with attribute variable to filter.
    variable : string
        Name of the attribute variable to filter.
    operator : string
        Logical operator used to filter attribute variable.
    value : string
        Value used to filter attribute variable.
    drop : boolean
        Indcates if fitered trajectories should be retained in the
        new Dataset (False) or dropped from the Dataset (True).

    Returns
    -------
    ds_reduced Dataset
        Reduced Dataset, including the Lagrangian trajectories
        which meet (do not meet) the specified filter condition.

    """
    # -------------------------------------------
    # Applying specified filter to time variable.
    # -------------------------------------------
    if variable == 'time':
        # Apply filter according to specfied comparison operator:
        # Case 1. Equal
        if operator == '==':
            # Filter Dataset and drop where False:
            ds_reduced = self.data.where(self.data[variable] == value, drop=True)

        # Case 2. Not Equal
        elif operator == '!=':
            # Filter Dataset and drop where False:
            ds_reduced = self.data.where(self.data[variable] != value, drop=True)

        # Case 3. Less Than
        elif operator == '<':
            # Filter Dataset and drop where False:
            ds_reduced = self.data.where(self.data[variable] < value, drop=True)

        # Case 4. Greater Than
        elif operator == '>':
            # Filter Dataset and drop where False:
            ds_reduced = self.data.where(self.data[variable] > value, drop=True)

        # Case 5. Less Than or Equal
        elif operator == '<=':
            # Filter Dataset and drop where False:
            ds_reduced = self.data.where(self.data[variable] <= value, drop=True)

        # Case 6. Greater Than or Equal
        elif operator == '>=':
            # Filter Dataset and drop where False:
            ds_reduced = self.data.where(self.data[variable] >= value, drop=True)

    # ------------------------------------------
    # Applying specified filter to 1-D variable.
    # ------------------------------------------
    elif len(self.data[variable].shape) == 1:
        # Apply filter according to specfied comparison operator:
        # Case 1. Equal
        if operator == '==':
            # Filter Dataset according to drop argument:
            if drop is True:
                ds_reduced = self.data.isel(traj=~(self.data[variable] == value))
            else:
                ds_reduced = self.data.isel(traj=(self.data[variable] == value))
        # Case 2. Not Equal
        elif operator == '!=':
            # Filter Dataset according to drop argument:
            if drop is True:
                ds_reduced = self.data.isel(traj=~(self.data[variable] != value))
            else:
                ds_reduced = self.data.isel(traj=(self.data[variable] != value))

        # Case 3. Less Than
        elif operator == '<':
            # Filter Dataset according to drop argument:
            if drop is True:
                ds_reduced = self.data.isel(traj=~(self.data[variable] < value))
            else:
                ds_reduced = self.data.isel(traj=(self.data[variable] < value))

        # Case 4. Greater Than
        elif operator == '>':
            # Filter Dataset according to drop argument:
            if drop is True:
                ds_reduced = self.data.isel(traj=~(self.data[variable] > value))
            else:
                ds_reduced = self.data.isel(traj=(self.data[variable] > value))

        # Case 5. Less Than or Equal
        elif operator == '<=':
            # Filter Dataset according to drop argument:
            if drop is True:
                ds_reduced = self.data.isel(traj=~(self.data[variable] <= value))
            else:
                ds_reduced = self.data.isel(traj=(self.data[variable] <= value))

        # Case 6. Greater Than or Equal
        elif operator == '>=':
            # Filter Dataset according to drop argument:
            if drop is True:
                ds_reduced = self.data.isel(traj=~(self.data[variable] >= value))
            else:
                ds_reduced = self.data.isel(traj=(self.data[variable] >= value))

    # ------------------------------------------
    # Applying specified filter to 2-D variable.
    # ------------------------------------------
    elif len(self.data[variable].shape) == 2:
        # Apply filter according to specfied comparison operator:
        # Case 1. Equal
        if operator == '==':
            # Filter Dataset according to drop argument:
            if drop is True:
                ds_reduced = self.data.isel(traj=~(self.data[variable] == value).any(dim='obs'))
            else:
                ds_reduced = self.data.isel(traj=(self.data[variable] == value).any(dim='obs'))
        # Case 2. Not Equal
        elif operator == '!=':
            # Filter Dataset according to drop argument:
            if drop is True:
                ds_reduced = self.data.isel(traj=~(self.data[variable] != value).any(dim='obs'))
            else:
                ds_reduced = self.data.isel(traj=(self.data[variable] != value).any(dim='obs'))

        # Case 3. Less Than
        elif operator == '<':
            # Filter Dataset according to drop argument:
            if drop is True:
                ds_reduced = self.data.isel(traj=~(self.data[variable] < value).any(dim='obs'))
            else:
                ds_reduced = self.data.isel(traj=(self.data[variable] < value).any(dim='obs'))

        # Case 4. Greater Than
        elif operator == '>':
            # Filter Dataset according to drop argument:
            if drop is True:
                ds_reduced = self.data.isel(traj=~(self.data[variable] > value).any(dim='obs'))
            else:
                ds_reduced = self.data.isel(traj=(self.data[variable] > value).any(dim='obs'))

        # Case 5. Less Than or Equal
        elif operator == '<=':
            # Filter Dataset according to drop argument:
            if drop is True:
                ds_reduced = self.data.isel(traj=~(self.data[variable] <= value).any(dim='obs'))
            else:
                ds_reduced = self.data.isel(traj=(self.data[variable] <= value).any(dim='obs'))

        # Case 6. Greater Than or Equal
        elif operator == '>=':
            # Filter Dataset according to drop argument:
            if drop is True:
                ds_reduced = self.data.isel(traj=~(self.data[variable] >= value).any(dim='obs'))
            else:
                ds_reduced = self.data.isel(traj=(self.data[variable] >= value).any(dim='obs'))

    # Return filtered Dataset:
    return ds_reduced

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
        # Filter Dataset and drop where False:
        ds_reduced = self.data.where((self.data[variable] <= max_val) & (self.data[variable] >= min_val), drop=True)

    # -------------------------------------------------------
    # Sub-Routine for filtering with 1-D attribute variables.
    # -------------------------------------------------------
    elif len(self.data[variable].shape) == 1:
        # Filter Dataset according to drop argument:
        if drop is True:
            ds_reduced = self.data.isel(traj=~((self.data[variable] <= max_val) & (self.data[variable] >= min_val)))
        else:
            ds_reduced = self.data.isel(traj=((self.data[variable] <= max_val) & (self.data[variable] >= min_val)))

    # -------------------------------------------------------
    # Sub-Routine for filtering with 2-D attribute variables.
    # -------------------------------------------------------
    elif len(self.data[variable].shape) == 2:
        # Filter Dataset according to drop argument:
        if drop is True:
            ds_reduced = self.data.isel(traj=~((self.data[variable] <= max_val) & (self.data[variable] >= min_val)).any(dim='obs'))
        else:
            ds_reduced = self.data.isel(traj=((self.data[variable] <= max_val) & (self.data[variable] >= min_val)).any(dim='obs'))

    # Return filtered Dataset:
    return ds_reduced

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
    self : trajectories object
        Trajectories object passed from trajectories class method.
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
    # Where multiple polygons are specified:
    if len(polygon) > 1:
        polygons = []
        # Iterate over polygons to create linearrings.
        for i in range(len(polygon)):
            shapes = list(shape for shape in polygon[i])
            polygons.append(pygeos.creation.linearrings(shapes))

        # Storing pygeos polygons, poly.
        poly = pygeos.creation.polygons(polygons)

    else:
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
