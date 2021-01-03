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
    Compute distance (m) between particle positions using Haversine formula.

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
        variable dx/dy/dz (m) DataArray containing particle
        displacements with dimensions (traj x obs).
    """
    # -------------------------------------------
    # Defining Variables and Physical Parameters
    # -------------------------------------------

    # Defining difference between Lon and Lat arrays, dLon and dLat.
    dLon = Lon2 - Lon1
    dLat = Lat2 - Lat1

    # Defining radius of the Earth, re (m), as
    # volumetric mean radius from NASA.
    # See: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    re = 6371000

    # ------------------------------------------------------------------
    # Computing the distance (km) between (Lat1, Lon1) and (Lat2, Lon2).
    # ------------------------------------------------------------------

    # Compute displacement angle in radians.
    a = np.sin(dLat/2)**2 + np.cos(Lat1) * np.cos(Lat2) * np.sin(dLon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Distance (km) obtain by multiplying displacement angle by
    # radius of Earth.
    d = c * re

    # Returning distance array, d (m).
    return d

##############################################################################
# Define compute_displacement() function.


def compute_displacement(self, axis, unit):
    """
    Compute particle displacements (m) from trajectories.

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
    unit : string
            Unit for particle displacement output -
            e.g. 'm'/'km'.

    Returns
    -------
    DataSet.
        Original DataSet is returned with appended attribute
        variable DataArray containing particle displacements
        (m) with dimensions (traj x obs).

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
        # Lat1 contains all latitudes except for final element in Lat.
        Lat1 = Lat[:, :-1]
        # Lat2 contains all latitudes except for first element in Lat.
        Lat2 = Lat[:, 1:]

        # Defining Lat_mean as the mean latitude between
        # each pair of points stored in Lat1 and Lat2.
        Lat_mean = (Lat1 + Lat2) / 2

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

        # NOTE: Using Lat_mean assumes zonal displacements occur
        # simultaneously with meridional displacements and hence
        # yields the zonal distance halfway between that computed
        # with Lat1 and Lat2.
        d = haversine_dist(Lat1=Lat_mean, Lon1=Lon1, Lat2=Lat_mean, Lon2=Lon2)

        # Use direction, direction, to transform distance, d, to compute
        # displacement, dx.
        dx = direction * d

        # Concantenate array of NaNs to start of dx, ensuring
        # dx has dimensions (traj x obs).
        empty = np.empty([rows, 1])
        empty[:, 0] = np.NaN
        dx = np.append(empty, dx, axis=1)

        # -----------------------------------
        # Conversions for zonal displacement.
        # -----------------------------------
        if unit == 'km':
            # No. meters in a km, kmeter.
            kmeter = 1000

            # Convert dx from m to km.
            dx = dx / kmeter

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

        # ----------------------------------------
        # Conversions for meridional displacement.
        # ----------------------------------------
        if unit == 'km':
            # No. meters in a km, kmeter.
            kmeter = 1000

            # Convert dy from m to km.
            dy = dy / kmeter

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

        # --------------------------------------
        # Conversions for vertical displacement.
        # --------------------------------------
        if unit == 'km':
            # No. meters in a km, kmeter.
            kmeter = 1000

            # Convert dz from m to km.
            dz = dz / kmeter

        else:
            pass

        # Return vertical displacements np.array, dz.
        return dz

##############################################################################
# Define compute_velocity() function.


def compute_velocity(self, axis, unit):
    """
    Compute particle Lagrangian velocity from displacements
    of trajectories.

    Zonal (x), meridional (y) or vertical (z) velocities are
    computed with displacements between particle positions
    and returned for all trajectories as a new DataArray.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    axis : string
        Axis to compute particle Lagrangian velocities -
        e.g. 'x'/'y'/'z'.
    unit : string
        Unit for Lagrangian velocity output -
        e.g 'm/s' / 'm/day' / 'km/day'.

    Returns
    -------
    DataSet.
        Original DataSet is returned with appended attribute
        variable DataArray containing particle displacements
        with dimensions (traj x obs).

        The first observation (obs) for all trajectories
        (traj) is NaN since the velocity from the origin
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
        # Lat1 contains all latitudes except for final element in Lat.
        Lat1 = Lat[:, :-1]
        # Lat2 contains all latitudes except for first element in Lat.
        Lat2 = Lat[:, 1:]

        # Defining Lat_mean as the mean latitude between
        # each pair of points stored in Lat1 and Lat2.
        Lat_mean = (Lat1 + Lat2) / 2

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

        # NOTE: Using Lat_mean assumes zonal displacements occur
        # simultaneously with meridional displacements and hence
        # yields the zonal distance halfway between that computed
        # with Lat1 and Lat2.
        d = haversine_dist(Lat1=Lat_mean, Lon1=Lon1, Lat2=Lat_mean, Lon2=Lon2)

        # Use direction, direction, to transform distance, d, to compute
        # displacement, dx.
        dx = direction * d

        # Concantenate array of NaNs to start of dx, ensuring
        # dx has dimensions (traj x obs).
        empty = np.empty([rows, 1])
        empty[:, 0] = np.NaN
        dx = np.append(empty, dx, axis=1)

        # -----------------------------------
        # Routine to compute zonal component
        # of Lagrangian velocity.
        # -----------------------------------

        # Defining time-step, dt.
        dt = np.diff(self.data.time.values, axis=1)
        dt = dt.astype(int)

        # Concantenate array of NaNs to start of dt, ensuring
        # dt has dimensions (traj x obs).
        empty = np.empty([rows, 1])
        empty[:, 0] = np.NaN
        dt = np.append(empty, dt, axis=1)

        # dt = int(self.data.time.values[0, 1] - self.data.time.values[0, 0])
        # Converting dt from nanoseconds to seconds.
        dt = dt * 1E-9

        # Computing zonal component of Lagrangian velocity
        # as u = dx/dt.
        u = dx / dt

        # --------------------------------------------
        # Conversions for zonal component of velocity.
        # --------------------------------------------
        if unit == 'm/day':
            # No. seconds in a day, day.
            day = 3600 * 24

            # Convert u from m/s to m/day.
            u = u * day

        elif unit == 'km/day':
            # No. seconds in a day, day.
            day = 3600 * 24
            # No. meters in a km, kmeter.
            kmeter = 1000

            # Convert u from m/s to km/day.
            u = u * (day / kmeter)

        else:
            pass

        # Return zonal velocity np.array, u.
        return u

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

        # ---------------------------------------
        # Routine to compute meridional component
        # of Lagrangian velocity.
        # ---------------------------------------

        # Defining time-step, dt.
        dt = np.diff(self.data.time.values, axis=1)
        dt = dt.astype(int)

        # Concantenate array of NaNs to start of dt, ensuring
        # dt has dimensions (traj x obs).
        empty = np.empty([rows, 1])
        empty[:, 0] = np.NaN
        dt = np.append(empty, dt, axis=1)

        # dt = int(self.data.time.values[0, 1] - self.data.time.values[0, 0])
        # Converting dt from nanoseconds to seconds.
        dt = dt * 1E-9

        # Computing meridional component of Lagrangian velocity
        # as v = dy/dt.
        v = dy / dt

        # -------------------------------------------------
        # Conversions for meridional component of velocity.
        # -------------------------------------------------
        if unit == 'm/day':
            # No. seconds in a day, day.
            day = 3600 * 24

            # Convert v from m/s to m/day.
            v = v * day

        elif unit == 'km/day':
            # No. seconds in a day, day.
            day = 3600 * 24
            # No. meters in a km, kmeter.
            kmeter = 1000

            # Convert v from m/s to km/day.
            v = v * (day / kmeter)

        else:
            pass

        # Return meridional velocity np.array, u.
        return v

    else:

        # ------------------------------------------------
        # Routine to compute vertical (z) displacements.
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

        # --------------------------------------
        # Routine to compute vertical component
        # of Lagrangian velocity.
        # --------------------------------------

        # Defining time-step, dt.
        dt = np.diff(self.data.time.values, axis=1)
        dt = dt.astype(int)

        # Concantenate array of NaNs to start of dt, ensuring
        # dt has dimensions (traj x obs).
        empty = np.empty([rows, 1])
        empty[:, 0] = np.NaN
        dt = np.append(empty, dt, axis=1)

        # dt = int(self.data.time.values[0, 1] - self.data.time.values[0, 0])
        # Converting dt from nanoseconds to seconds.
        dt = dt * 1E-9

        # Computing vertical component of Lagrangian velocity
        # as w = dz/dt.
        w = dz / dt

        # -----------------------------------------------
        # Conversions for vertical component of velocity.
        # -----------------------------------------------
        if unit == 'm/day':
            # No. seconds in a day, day.
            day = 3600 * 24

            # Convert w from m/s to m/day.
            w = w * day

        elif unit == 'km/day':
            # No. seconds in a day, day.
            day = 3600 * 24
            # No. meters in a km, kmeter.
            kmeter = 1000

            # Convert w from m/s to km/day.
            w = w * (day / kmeter)

        else:
            pass

        # Return vertical velocity np.array, w.
        return w

##############################################################################
# Define compute_distace() function.


def compute_distance(self, cumsum_dist, unit):
    """
    Compute distance travelled by particles along their
    of trajectories.

    Either the distance travelled between particle positions
    or the cumulative distance travelled is computed
    and returned for all trajectories as a new DataArray.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    cumsum_dist : logical
        Compute the cumulative distance travelled by each particle -
        default is False.
    unit : string
        Unit for distance travelled output -
        e.g 'm' / 'km'.

    Returns
    -------
    DataSet.
        Original DataSet is returned with appended attribute
        variable DataArray containing the (cumulative) distance
        travelled by each particle along its trajectory with
        dimensions (traj x obs).

        The first observation (obs) for all trajectories
        (traj) is NaN since the (cumulative) distance
        from the origin of a particle at the origin is
        not defined.
    """
    # -----------------------------------------------
    # Transforming Latitude and Longitude variables.
    # -----------------------------------------------

    # Transforming lat and lon variables into radians, Lat and Lon.
    Lat = np.radians(self.data['lat'].values)
    Lon = np.radians(self.data['lon'].values)

    # Store number of trajectories as rows.
    rows = len(Lat)

    # -----------------------------------------------------
    # Routine to compute haversine distance between points.
    # -----------------------------------------------------

    # Defining latitudes for haversine_dist(), Lat1 and Lat2.
    # Lat1 contains all latitudes except for final element in Lat.
    Lat1 = Lat[:, :-1]
    # Lat2 contains all latitudes except for first element in Lat.
    Lat2 = Lat[:, 1:]
    # By definition, Lat1 and Lat2 are hence staggered by a
    # single time-level.

    # Defining longitudes for haversine_dist(), Lon1 and Lon2.
    # Lon1 contains all longitudes except for final element in Lon.
    Lon1 = Lon[:, :-1]
    # Lon2 contains all longitudes except for the first element in Lon.
    Lon2 = Lon[:, 1:]
    # By definition, Lon1 and Lon2 are hence staggered by a
    # single time-level.

    # Computing distances between trajectory points with
    # haversine_dist() function.
    ds = haversine_dist(Lat1=Lat1, Lon1=Lon1, Lat2=Lat2, Lon2=Lon2)

    # Concantenate array of NaNs to start of ds, ensuring
    # dx has dimensions (traj x obs).
    empty = np.empty([rows, 1])
    empty[:, 0] = np.NaN
    ds = np.append(empty, ds, axis=1)

    # -------------------------------------------
    # Routine to compute vertical (z) distances.
    # -------------------------------------------

    # Computing vertical distances between trajectory points with
    # np.diff().
    Z = self.data['z'].values
    dz = abs(np.diff(Z, axis=1))

    # Concantenate array of NaNs to start of dz, ensuring
    # dz has dimensions (traj x obs).
    empty = np.empty([rows, 1])
    empty[:, 0] = np.NaN
    dz = np.append(empty, dz, axis=1)

    if cumsum_dist is False:
        # --------------------------------------------
        # Routine to compute distances between points.
        # --------------------------------------------
        # Computing distance between points on trajectory.
        dist = np.sqrt(dz**2 + ds**2)

    else:
        # --------------------------------------------------------
        # Routine to compute cumulative distance for trajectories.
        # --------------------------------------------------------
        # Computing distance between points on trajectory.
        dist = np.sqrt(dz**2 + ds**2)
        # Computing cumaltive sum of distance for each trajectory.
        # np.nancumsum treats NaN values as zeros.
        dist = np.nancumsum(dist, axis=1)

    # --------------------------
    # Conversions for distances.
    # --------------------------
    if unit == 'km':
        # No. meters in a km, kmeter.
        kmeter = 1000

        # Convert dist from m to km.
        dist = dist / kmeter

    else:
        pass

    # Return cumulative distance for each trajectory np.array, dist.
    return dist
