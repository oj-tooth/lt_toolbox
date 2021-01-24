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

import random
import numpy as np
import xarray as xr
import scipy.stats as stats
from scipy.ndimage import gaussian_filter

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
    ndarray
        Haversine distance between points (Lat1, Lon1) and
        (Lat2, Lon2) with dimensions (traj x obs-1).
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
# Define lagrangian_probability() function.


def lagrangian_probability(self, lat_lims, lon_lims, bin_res, method, gf_sigma=None):
    """
    Compute 2-dimensional binned Lagrangian probability
    distributions using particle positions or particle
    pathways.

    Particle positions are binned into a 2-dimensional
    (x-y) histogram and normalised by the total number
    of particle positions ('pos') or the total number
    of particles ('traj').

    A Gaussian filter with a specified standard deviation
    may alsobe included to smooth the distribution.

    Parameters
    ----------
    self : trajectories object
    lat_lims : list
        List containing [min, max] latitudes defining the grid
        domain.
    lon_lims : list
        List containing [min, max] longitudes defining the grid
        domain.
        Trajectories object passed from trajectories class method.
    bin_res : numeric
        The resolution (degrees) of the grid on to which particle
        positions will be binned.
    method : string
        The type of probability to be computed. 'pos' - particle
        positions are binned and then normalised by the total number
        of particle positions. 'traj' - for each particle positions
        are counted once per bin and then normalised by the total
        number of particles. To include a Gaussian filter modify the
        methods above to 'pos-gauss' or 'traj-gauss'.
    gf_sigma : numeric
        The standard deviation of the Gaussian filter (degrees) with
        which to smooth the Lagrangian probability distribution.

    Returns
    -------
    lon_centre : ndarray
        Longitude of the centre points of the grid with dimensions (x - y).
    lat_centre : ndarray
        Longitude of the centre points of the grid with dimensions (x - y).
    p : ndarray
        Binned 2-dimensional Lagrangian probability distribution with
        dimensions (x - y).
    """
    # ---------------------------------------------------------
    # Defining grid on which particle positions will be binned.
    # ---------------------------------------------------------
    # Defining lat and lon for trajectories.
    lat = np.copy(self.data.lat.values)
    lon = np.copy(self.data.lon.values)

    # Defining bin size with specified bin_res.
    dx = dy = bin_res

    # Defining bin edges in 1-dimensional arrays.
    bin_x = np.arange(lon_lims[0], lon_lims[1] + dx, dx)
    bin_y = np.arange(lat_lims[0], lat_lims[1] + dy, dy)

    # Defining 2-d grids from 1-d bin edge arrays.
    lat_grid, lon_grid = np.meshgrid(bin_y, bin_x)

    # Defining the coordinates of centre of each grid cell
    # with lat_centre, lon_centre.
    lon_centre = lon_grid[:-1, :] + np.diff(lon_grid, axis=0)/2
    lat_centre = lat_grid[:, :-1] + np.diff(lat_grid, axis=1)/2

    # Resizing to equal shape of probability array (x-y).
    lon_centre = lon_centre[:, :-1]
    lat_centre = lat_centre[:-1, :]

    # ---------------------------------------------------
    # Subroutine for probability with particle positions.
    # ---------------------------------------------------
    if method == 'pos':

        # -----------------------------------
        # Computing particle density on grid.
        # -----------------------------------
        # Using scipy to count the number of particle positions per bin
        stat = stats.binned_statistic_2d(x=lon.flatten(), y=lat.flatten(), values=None, statistic='count', bins=[bin_x, bin_y])

        # ------------------------------
        # Computing probability on grid.
        #  ------------------------------
        # Defining number of particle positions, npos.
        npos = len(lon.flatten())  # lon/lat could be used here.
        # Compute probability as a percentage, prob.
        probability = stat.statistic / npos

    # ---------------------------------------------------------------
    # Subroutine for probability with particle positions and gfilter.
    # ---------------------------------------------------------------
    elif method == 'pos-gauss':

        # -----------------------------------
        # Computing particle density on grid.
        # -----------------------------------
        # Using scipy to count the number of particle positions per bin
        stat = stats.binned_statistic_2d(x=lon.flatten(), y=lat.flatten(), values=None, statistic='count', bins=[bin_x, bin_y])

        # ------------------------------
        # Computing probability on grid.
        #  ------------------------------
        # Defining number of particle positions, npos.
        npos = len(lon.flatten())  # lon/lat could be used here.
        # Compute probability as a percentage, prob.
        probability = stat.statistic / npos

        # ----------------------
        # Apply Gaussian filter.
        # ----------------------
        # Spatially filter probability distribution with isotropic
        # Gaussian filter.
        probability = gaussian_filter(probability, sigma=gf_sigma, mode="constant")

    # -------------------------------------------------
    # Subroutine for probability with all trajectories.
    # -------------------------------------------------
    elif method == 'traj':

        # Defining array to store particle density.
        density = np.zeros([len(bin_x) - 1, len(bin_y) - 1])

        # Defining no. trajectories, ntraj.
        ntraj = np.shape(lat)[0]  # lat/lon could be used here.

        # Iterate over all trajectories.
        for i in np.arange(0, ntraj):
            # Using scipy to count the number of particle per bin.
            stat = stats.binned_statistic_2d(lon[i, :], lat[i, :], None, 'count', bins=[bin_x, bin_y])
            # Where a particle is counted more than once in bin set = 1.
            stat.statistic[stat.statistic > 1] = 1
            # Update density with counts from particle.
            density = density + stat.statistic

        # -----------------------------
        # Computing probability on grid.
        #  ------------------------------
        # Compute probability as a percentage, prob.
        probability = density / ntraj

    # ------------------------------------------------------------
    # Subroutine for probability with all trajectories and filter.
    # ------------------------------------------------------------
    elif method == 'traj-gauss':

        # Defining array to store particle density.
        density = np.zeros([len(bin_x) - 1, len(bin_y) - 1])

        # Defining no. trajectories, ntraj.
        ntraj = np.shape(lat)[0]  # lat/lon could be used here.

        # Iterate over all trajectories.
        for i in np.arange(0, ntraj):
            # Using scipy to count the number of particle per bin.
            stat = stats.binned_statistic_2d(lon[i, :], lat[i, :], None, 'count', bins=[bin_x, bin_y])
            # Where a particle is counted more than once in bin set = 1.
            stat.statistic[stat.statistic > 1] = 1
            # Update density with counts from particle.
            density = density + stat.statistic

        # -----------------------------
        # Computing probability on grid.
        #  ------------------------------
        # Compute probability as a percentage, prob.
        probability = density / ntraj

        # ----------------------
        # Apply Gaussian filter.
        # ----------------------
        # Spatially filter probability distribution with isotropic
        # Gaussian filter.
        probability = gaussian_filter(probability, sigma=gf_sigma, mode="constant")

    # ----------------------------------------------------------------
    # Returning computed variables to be added to trajectories object.
    # ----------------------------------------------------------------
    # Returning the latitude, longitude and probability gridded data.
    return lon_centre, lat_centre, probability

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
    ndarray
        Zonal, meridional or vertical displacement from particle
        trajectories with dimensions (traj x obs).

        The first observation (obs) for all trajectories
        (traj) is NaN since the meridional distance from the origin
        of a particle at the origin is not defined.
    """
    # -----------------------------------------------
    # Transforming Latitude and Longitude variables.
    # -----------------------------------------------

    # Transforming lat and lon variables into radians, Lat and Lon.
    Lat = np.radians(np.copy(self.data['lat'].values))
    Lon = np.radians(np.copy(self.data['lon'].values))

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
    ndarray
        Zonal, meridional or vertical velocity from particle
        trajectories with dimensions (traj x obs).

        The first observation (obs) for all trajectories (traj)
        is NaN since the meridional distance from the origin
        of a particle at the origin is not defined.
    """
    # -----------------------------------------------
    # Transforming Latitude and Longitude variables.
    # -----------------------------------------------

    # Transforming lat and lon variables into radians, Lat and Lon.
    Lat = np.radians(np.copy(self.data['lat'].values))
    Lon = np.radians(np.copy(self.data['lon'].values))

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
        dt = np.diff(np.copy(self.data.time.values), axis=1)
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
        dt = np.diff(np.copy(self.data.time.values), axis=1)
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
        Z = np.copy(self.data['z'].values)
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
        dt = np.diff(np.copy(self.data.time.values), axis=1)
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
    ndarray
        Distance or cumulative distance from particle
        trajectories with dimensions (traj x obs).

        The first observation (obs) for all trajectories
        (traj) is NaN since the meridional distance from
        the origin of a particle at the origin is not defined.
    """
    # -----------------------------------------------
    # Transforming Latitude and Longitude variables.
    # -----------------------------------------------

    # Transforming lat and lon variables into radians, Lat and Lon.
    Lat = np.radians(np.copy(self.data['lat'].values))
    Lon = np.radians(np.copy(self.data['lon'].values))

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
    Z = np.copy(self.data['z'].values)
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

    # Return cumulative distance for each trajectory ndarray, dist.
    return dist


##############################################################################
# Define compute_probability_distribution() function.

def compute_probability_distribution(self, bin_res, method, gf_sigma=None, group_by=None):
    """
    Compute 2-dimensional binned Lagrangian probability
    distributions using particle positions or particle
    pathways.

    Particle positions are binned into a 2-dimensional
    (x-y) histogram and normalised by the total number
    of particle positions ('pos') or the total number
    of particles ('traj').

    A Gaussian filter with a specified radius may also
    be included to smooth the distribution.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    bin_res : numeric
        The resolution (degrees) of the grid on to which particle
        positions will be binned.
    method : string
        The type of probability to be computed. 'pos' - particle
        positions are binned and then normalised by the total number
        of particle positions. 'traj' - for each particle positions
        are counted once per bin and then normalised by the total
        number of particles. To include a Gaussian filter modify the
        methods above to 'pos-gauss' or 'traj-gauss'.
    gf_sigma : numeric
        The standard deviation of the Gaussian filter (degrees) with
        which to smooth the Lagrangian probability distribution.
    group_by : string
        Grouping variable to compute Lagrangian probability
        distributions - one distribution is computed for every
        unique member of variable. See example below.

    Returns
    -------
    DataSet.
        Original DataSet is returned with appended attribute
        variable DataArray containing the binned 2-dimensional
        Lagrangian probability distribution with
        dimensions (x - y).
    """
    # ------------------------------------------------
    # Defining grid domain to determine probabilities.
    # ------------------------------------------------
    # Defining lat and lon for trajectories.
    lat = np.copy(self.data.lat.values)
    lon = np.copy(self.data.lon.values)

    # Finding the maximum and minimum values of lat and lon
    # to the nearest degree E/N for all trajectories
    lat_max = np.ceil(np.nanmax(lat))
    lat_min = np.floor(np.nanmin(lat))
    lon_max = np.ceil(np.nanmax(lon))
    lon_min = np.floor(np.nanmin(lon))

    # Storing min and max latitudes in lat_lims.
    lat_lims = [lat_min, lat_max]
    # Storing min and max longitudes in lon_lims.
    lon_lims = [lon_min, lon_max]

    # -------------------------------------
    # Subroutine without group_by variable.
    # -------------------------------------
    if group_by is None:

        # -----------------------------------------------
        # Computing Lagrangian Probability Distributions.
        # -----------------------------------------------

        # Compute probability distribution to return gridded
        # longitudes, latitudes and probabilities.
        nav_lon, nav_lat, probability = lagrangian_probability(self, lat_lims=lat_lims, lon_lims=lon_lims, bin_res=bin_res, method=method, gf_sigma=gf_sigma)

        # Append DataArrays to original DataSet.
        self.data['nav_lat'] = xr.DataArray(nav_lat, dims=["y", "x"])
        self.data['nav_lon'] = xr.DataArray(nav_lon, dims=["y", "x"])
        self.data['probability'] = xr.DataArray(probability, dims=["y", "x"])

    # ----------------------------------
    # Subroutine with group_by variable.
    # ----------------------------------
    else:
        # Determining the number of unique elements in group_by.
        vals = np.unique(self.data[group_by].values)
        # Defining empty list to store probs.
        probs = []

        # -----------------------------------------------
        # Computing Lagrangian Probability Distributions.
        # -----------------------------------------------

        # Iterate over all unique elements in group_by and
        # compute a Lagrangian probability distribution.
        for n in vals:
            # Filter trajectories where group_by equals n.
            traj = self.filter_equal(group_by, n)

            # Compute probability distribution to return gridded
            # longitudes, latitudes and probabilities.
            nav_lon, nav_lat, p = lagrangian_probability(traj, lat_lims=lat_lims, lon_lims=lon_lims, bin_res=bin_res, method=method, gf_sigma=gf_sigma)

            # Append probability distribution, p, in probability.
            probs.append(p)

        # ------------------------------------------
        # Appending data to our trajectories object.
        # ------------------------------------------
        # Convert probs to a 3-dimensional ndarray.
        probability = np.stack(probs)

        # Append DataArrays to original DataSet.
        self.data['nav_lat'] = xr.DataArray(nav_lat, dims=["y", "x"])
        self.data['nav_lon'] = xr.DataArray(nav_lon, dims=["y", "x"])
        self.data['probability'] = xr.DataArray(probability, dims=["sample", "y", "x"])

    # -----------------------------------
    # Adding variable attributes DataSet.
    # -----------------------------------
    self.data.nav_lat.attrs = {
                        'long_name': "Latitude",
                        'standard_name': "latitude",
                        'units': "degrees_north",
                        }
    self.data.nav_lon.attrs = {
                        'long_name': "Longitude",
                        'standard_name': "longitude",
                        'units': "degrees_east",
                        }
    self.data.probability.attrs = {
                        'long_name': "Lagrangian probability",
                        'standard_name': "probability",
                        }

    # Return updated DataSet.
    return self.data


##############################################################################
# Define compute_fuv() method.

def compute_fuv(self, bin_res, method, resample, repeats, gf_sigma=None, group_by=None):
    """
    Compute fraction of unexplained variance (FUV)
    between a series of 2-dimensional binned Lagrangian
    probability distributions and a reference distribution.

    Particle positions are binned into a 2-dimensional
    (x-y) histogram and normalised by the total number
    of particle positions ('pos') or the total number
    of particles ('traj').

    A Gaussian filter with a specified radius may also
    be included to smooth the distribution.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    bin_res : numeric
        The resolution (degrees) of the grid on to which particle
        positions will be binned.
    method : string
        The type of probability to be computed. 'pos' - particle
        positions are binned and then normalised by the total number
        of particle positions. 'traj' - for each particle positions
        are counted once per bin and then normalised by the total
        number of particles. To include a Gaussian filter modify the
        methods above to 'pos-gauss' or 'traj-gauss'.
    gf_sigma : numeric
        The standard deviation of the Gaussian filter (degrees) with
        which to smooth the Lagrangian probability distribution.
    resample : list
        A list of integers containing the number of trajectories to
        randomly resample from the reference simulation.
    repeats : numeric
        Number of times to repeat each random resampling of the
        reference simulation.
    group_by : string
        Grouping variable to compute reference Lagrangian probability
        distributions - one distribution is computed for every
        unique member of variable. See example below.

    Returns
    -------
    ndarray.
        The 95% upper bound FUV values for each resample (N) of the
        reference simulation are returned in a (1 x N) array. Where
        group_by to identify multiple reference simulations a (r x N)
        array will be returned, where r is equal to the number of
        reference simulations.
    """
    # ------------------------------------------------
    # Defining grid domain to determine probabilities.
    # ------------------------------------------------
    # Defining lat and lon for trajectories.
    lat = np.copy(self.data.lat.values)
    lon = np.copy(self.data.lon.values)

    # Finding the maximum and minimum values of lat and lon
    # to the nearest degree E/N for all trajectories
    lat_max = np.ceil(np.nanmax(lat))
    lat_min = np.floor(np.nanmin(lat))
    lon_max = np.ceil(np.nanmax(lon))
    lon_min = np.floor(np.nanmin(lon))

    # Storing min and max latitudes in lat_lims.
    lat_lims = [lat_min, lat_max]
    # Storing min and max longitudes in lon_lims.
    lon_lims = [lon_min, lon_max]

    # -------------------------------------
    # Subroutine without group_by variable.
    # -------------------------------------
    if group_by is None:

        # --------------------------------
        # Defining number of trajectories.
        # --------------------------------
        # Any (traj x obs) variable could be used here.
        ntraj = np.shape(self.data.lat)[0]

        # -----------------------------------------------
        # Computing Lagrangian probability distribution.
        # -----------------------------------------------
        # Compute reference probability distribution.
        _, _, reference_p = lagrangian_probability(self, lat_lims=lat_lims, lon_lims=lon_lims, bin_res=bin_res, method=method, gf_sigma=gf_sigma)

        # ----------------------------------------------------
        # Resample trajectories from the reference simulation.
        # ----------------------------------------------------
        # Defining fuv to store FUV values from repeats.
        fuv = np.zeros(repeats)

        # Defining fuv_upper_bound to store 95% upper bound of
        # FUV values for each resample.
        nsample = len(resample)
        fuv_upper_bound = np.zeros(nsample)

        for N in np.arange(nsample):
            for rep in np.arange(repeats):
                # Randomly sample indices of N trajectories.
                ind = random.sample(range(0, ntraj), resample[N])
                # Use indices to create list of ids to resample.
                sample_id = list(self.id.values[ind])
                # Store resample N trajectories by id.
                traj = self.filter_equal('id', sample_id)

                # ----------------------------------------------
                # Computing Lagrangian probability distribution.
                # ----------------------------------------------
                # Compute resampled Lagrangian probability distribution.
                _, _, resample_p = lagrangian_probability(traj, lat_lims=lat_lims, lon_lims=lon_lims, bin_res=bin_res, method=method, gf_sigma=gf_sigma)

                # -------------------------------------------
                # Compute FUV from reference and sample LPDs.
                # -------------------------------------------
                # Flatten probability 2-d arrays for comparison.
                reference_p = reference_p.flatten()
                resample_p = resample_p.flatten()

                # Find indices of NaN values in probability arrays.
                nan_vals = ~np.logical_or(np.isnan(reference_p), np.isnan(resample_p))

                # Extract only elements where reference/resample_p
                # have values to compare, x and y.
                x = np.compress(nan_vals, reference_p)
                y = np.compress(nan_vals, resample_p)

                # Compute the pearson correlation coefficient (r).
                r, _ = stats.pearsonr(x, y)

                # Compute Fraction of Unexplained Variance, fuv.
                fuv[rep] = (1 - r**2)

            # -------------------------------
            # Compute 95% upper bound of FUV.
            # -------------------------------
            # Finding the 95% percentile of the FUV distribution
            # stored in fuv.
            fuv_upper_bound[N] = np.percentile(fuv, 95)

    # ----------------------------------
    # Subroutine with group_by variable.
    # ----------------------------------
    else:
        # Determining the number of unique elements in group_by.
        vals = np.unique(self.data[group_by].values)

        # Defining number of reference simulations.
        nval = len(vals)
        # Defining number of resamples.
        nsample = len(resample)

        # -----------------------------------------------
        # Computing Lagrangian Probability Distributions.
        # -----------------------------------------------
        # Defining fuv_upper_bound to store 95% upper bound of
        # FUV values for each resample for each reference sim.
        fuv_upper_bound = np.zeros([nval, nsample])

        # Iterate over all unique elements in group_by and
        # compute a Lagrangian probability distribution.
        for n in np.arange(nval):
            # Filter trajectories where group_by equals n.
            traj = self.filter_equal(group_by, vals[n])

            # ----------------------------------------
            # Defining number of trajectories in traj.
            # ----------------------------------------
            # Any (traj x obs) variable could be used here.
            ntraj = np.shape(traj.data.lat)[0]

            # -----------------------------------------------
            # Computing Lagrangian probability distribution.
            # -----------------------------------------------
            # Compute reference probability distribution.
            _, _, reference_p = lagrangian_probability(traj, lat_lims=lat_lims, lon_lims=lon_lims, bin_res=bin_res, method=method, gf_sigma=gf_sigma)

            # ----------------------------------------------------
            # Resample trajectories from the reference simulation.
            # ----------------------------------------------------
            # Defining fuv to store FUV values from repeats.
            fuv = np.zeros(repeats)

            for N in np.arange(nsample):
                for rep in np.arange(repeats):
                    # Randomly generate indices of N trajectories.
                    ind = random.sample(range(0, ntraj), resample[N])
                    # Use indices to create list of ids to resample.
                    sample_id = list(traj.id.values[ind])
                    # Store resample N trajectories by id.
                    traj_resample = traj.filter_equal('id', sample_id)

                    # ----------------------------------------------
                    # Computing Lagrangian probability distribution.
                    # ----------------------------------------------
                    # Compute resampled Lagrangian probability distribution.
                    _, _, resample_p = lagrangian_probability(traj_resample, lat_lims=lat_lims, lon_lims=lon_lims, bin_res=bin_res, method=method, gf_sigma=gf_sigma)

                    # -------------------------------------------
                    # Compute FUV from reference and sample LPDs.
                    # -------------------------------------------
                    # Flatten probability 2-d arrays for comparison.
                    reference_p = reference_p.flatten()
                    resample_p = resample_p.flatten()

                    # Find indices of NaN values in probability arrays.
                    nan_vals = ~np.logical_or(np.isnan(reference_p), np.isnan(resample_p))

                    # Extract only elements where reference/resample_p
                    # have values to compare, x and y.
                    x = np.compress(nan_vals, reference_p)
                    y = np.compress(nan_vals, resample_p)

                    # Compute the pearson correlation coefficient (r).
                    r, _ = stats.pearsonr(x, y)

                    # Compute Fraction of Unexplained Variance, fuv.
                    fuv[rep] = (1 - r**2)

                # -------------------------------
                # Compute 95% upper bound of FUV.
                # -------------------------------
                # Finding the 95% percentile of the FUV distribution
                # stored in fuv.
                fuv_upper_bound[n, N] = np.percentile(fuv, 95)

    # Return updated DataSet.
    return fuv_upper_bound
