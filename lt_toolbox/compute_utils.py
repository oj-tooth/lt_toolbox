##############################################################################
# compute_utils.py
#
# Description:
# Defines functions for computing new properties for
# trajectories objects.
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

##############################################################################
# Define haversine_distance() function.


def haversine_dist(Lat1, Lon1, Lat2, Lon2):
    """
    Compute distance (km) between particle positions using Haversine formula.

    Latitude and Longitude vectors for the start (1) and end (2) positions
    are used to compute the distance (km) between them.

    Parameters
    ----------
    Lat1 : ndarray
        Latitudes of start points with dimensions (traj x obs).
    Lat2 : ndarray
        Latitudes of end points with dimensions (traj x obs).
    Lon1: ndarray
        Longitudes of start points with dimenions (traj x obs).
    Lon2: ndarray
        Longitudes of end points with dimensions (traj x obs).

    Returns
    -------
    DataSet.
        Original DataSet is returned with appended attribute
        variable dx/dy/dz DataArray containing particle
        displacements with dimensions (traj x obs).
    """
    # -------------------------------------------
    # Defining Variables and Physical Parameters
    # -------------------------------------------

    # Defining difference between Lon and Lat arrays, dLon and dLat.
    dLon = Lon2 - Lon1
    dLat = Lat2 - Lat1

    # Defining radius of the Earth, re (km), as
    # volumetric mean radius from NASA.
    # See: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    re = 6371

    # ------------------------------------------------------------------
    # Computing the distance (km) between (Lat1, Lon1) and (Lat2, Lon2).
    # ------------------------------------------------------------------

    # Compute displacement angle in radians.
    a = np.sin(dLat/2)**2 + np.cos(Lat1) * np.cos(Lat2) * np.sin(dLon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Distance (km) obtain by multiplying displacement angle by
    # radius of Earth.
    d = c * re

    # Returning distance array, d (km).
    return d

##############################################################################
# Define compute_displacement() function.


def compute_displacement(self, axis):
    """
    Compute particle displacements from trajectories.

    Zonal (x), meridional (y) or vertical (z) displacements
    between particle positions for all trajectories are
    returned as a new DataArray.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    axis : string
        Axis to compute particle displacements -
        e.g. 'x'/'y'/'z'.

    Returns
    -------
    DataSet.
        Original DataSet is returned with appended attribute
        variable DataArray containing particle displacements
        with dimensions (traj x obs).
        The first observation (obs) for all trajectories
        (traj) is NaN since the distance from the origin
        of a particle at the origin is not defined.
    """
    # -----------------------------------------------
    # Transforming Latitude and Longitude variables.
    # -----------------------------------------------

    # Transforming lat and lon variables into radians, Lat and Lon.
    Lat = np.radians(self.data['lat'].values)
    Lon = np.radians(self.data['lon'].values)

    # Store number of trajectories as rows.
    rows = len(Lat)

    if axis == 'x':

        # -------------------------------------------
        # Routine to compute zonal (x) displacements.
        # -------------------------------------------

        # Defining latitudes for haversine_dist(), Lat1 and Lat2.
        # Lat1 and Lat2 contain all latitudes except for final element in Lat.
        Lat1 = Lat[:, :-1]
        Lat2 = Lat[:, :-1]

        # Defining longitudes for haversine_dist(), Lon1 and Lon2.
        # Lon1 contains all longitudes except for final element in Lon.
        Lon1 = Lon[:, :-1]
        # Lon2 contains all longitudes except for the first element in Lon.
        Lon2 = Lon[:, 1:]
        # By definition, Lon1 and Lon2 are hence staggered by a
        # single time-level.

        # Defining direction array, direction, to use with distances array,
        # d, to generate displacements.
        # For x (zonal) displacements, if Lon2 > Lon1 displacement must be
        # eastwards (+).
        less = np.less(Lon2, Lon1)
        direction = (-1)**less

        # Computing zonal distances between trajectory points with
        # haversine_dist() function.
        d = haversine_dist(Lat1=Lat1, Lon1=Lon1, Lat2=Lat2, Lon2=Lon2)

        # Use direction, direction, to transform distance, d, to compute
        # displacement, dx.
        dx = direction * d

        # Concantenate array of NaNs to start of dx, ensuring
        # dx has dimensions (traj x obs).
        empty = np.empty([rows, 1])
        empty[:, 0] = np.NaN
        dx = np.append(empty, dx, axis=1)

        # Return zonal displacements np.array, dx.
        return dx

    elif axis == 'y':

        # ------------------------------------------------
        # Routine to compute meridional (y) displacements.
        # ------------------------------------------------

        # Defining longitudes for haversine_dist(), Lon1 and Lon2.
        # Lon1 and Lon2 contain all latitudes except for final element in Lon.
        Lon1 = Lon[:, :-1]
        Lon2 = Lon[:, :-1]

        # Defining latitudes for haversine_dist(), Lat1 and Lat2.
        # Lat1 contains all latitudes except for final element in Lat.
        Lat1 = Lat[:, :-1]
        # Lat2 contains all longitudes except for the first element in Lat.
        Lat2 = Lat[:, 1:]
        # By definition, Lon1 and Lon2 are hence staggered by a
        # single time-level.

        # Defining direction array, direction, to use with distances array,
        # d, to generate displacements.
        # For y (meridional) displacements, if Lat2 > Lat1 displacement must be
        # northwards (+).
        less = np.less(Lat2, Lat1)
        direction = (-1)**less

        # Computing meridional distances between trajectory points with
        # haversine_dist() function.
        d = haversine_dist(Lat1=Lat1, Lon1=Lon1, Lat2=Lat2, Lon2=Lon2)

        # Use direction, direction, to transform distance, d, to compute
        # displacement, dy.
        dy = direction * d

        # Concantenate array of NaNs to start of dy, ensuring
        # dy has dimensions (traj x obs).
        empty = np.empty([rows, 1])
        empty[:, 0] = np.NaN
        dy = np.append(empty, dy, axis=1)

        # Return meridional displacements np.array, dy.
        return dy

    else:

        # ------------------------------------------------
        # Routine to compute veritcal (z) displacements.
        # ------------------------------------------------

        # Computing vertical distances between trajectory points with
        # np.diff().
        Z = self.data['z'].values
        dz = np.diff(Z, axis=1)

        # Concantenate array of NaNs to start of dz, ensuring
        # dz has dimensions (traj x obs).
        empty = np.empty([rows, 1])
        empty[:, 0] = np.NaN
        dz = np.append(empty, dz, axis=1)

        # Return vertical displacements np.array, dz.
        return dz
