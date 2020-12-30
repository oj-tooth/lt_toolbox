##############################################################################
# trajectories.py
#
# Description:
# Defines the trajectories Class from a .nc file containing atmosphere
# ocean parcel trajectories (and accompanying tracers), stored following
# CF-conventions implemented with the NCEI trajectory template.
#
# Last Edited:
# 2020/12/22
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.

import xarray as xr
import numpy as np
from get_utils import get_start_time, get_start_loc, get_end_time, get_end_loc, get_duration, get_seed, get_val
from filter_utils import filter_traj
from find_utils import find_traj
from compute_utils import compute_displacement, compute_velocity, compute_distance
from plot_utils import plot_trajectories

##############################################################################
# Define trajectories Class.


class trajectories:
    def __init__(self, ds):
        """
        Create a trajectories object from an xarray DataSet.

        Parameters
        ----------
        ds: DataSet
            Trajectories stored in xarray DataSet following
            CF-conventions implemented with the NCEI trajectory
            template.

        For NCEI trajectory template see
        https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl

        Returns
        --------
        trajectories object
            Complete trajectories, including all attribute variables
            contained in DataSet, ds.

        Examples
        --------
        Creating trajectories object, traj, with output_file.nc file.
        >>> trajectories = trajectories(xr.open_dataset('output_file.nc'))
        """
        # ----------------------------------------
        # Storing input Dataset as data attribute.
        # ----------------------------------------
        # Defining data as input xarray DataSet.
        self.data = ds

        # --------------------------------------------
        # Define trajectories obj attribute variables.
        # --------------------------------------------
        # For improved useability, extract variables from data,
        # storing them as variables in the class.

        # Defining list of variables contained in data.
        variables = list(self.data.variables)

        # Set all variables in DataSet to attribute
        # variables of trajectories object.
        for var in variables:
            setattr(self, var, getattr(self.data, var))

##############################################################################
# Define use_datetime.

    def use_datetime(self, start_date):
        """
        Convert time attribute variable to datetime64 format.

        Parameters
        ----------
        start_date : string
            Starting date to use when converting time attribute
            variable to datetime64, formatted as 'YYYY-MM-DD'

        Returns
        --------
        trajectories object
            Original trajectories object is returned with transformed
            time attribute variable DataArray containing datetimes
            with dimensions (traj x obs).

        Examples
        --------
        Convert time in trajectories object to datetime with start
        date '2000-01-01.
        >>> trajectories.use_datetime('2000-01-01')
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(start_date, str) is False:
            raise TypeError("start_date must be specified as a string")

        # -----------------------------------
        # Set start_date in datetime format.
        # -----------------------------------
        # Using np.datetime64 to convert start_date.
        start_date = np.datetime64(start_date)

        # --------------------------------------------------
        # Convert time to datetime64 format with start_date.
        # --------------------------------------------------
        # Redefining time variable in datetime64 format.
        self.data.time.values = start_date + self.data.time.values

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define filter_between() method.

    def filter_between(self, variable, min_val, max_val):
        """
        Filter trajectories between two values of an attribute variable.

        Filtering returns the complete trajectories where the specified
        attribute variable takes a value between a specified min
        and max (including these values).

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
        trajectories object
            Complete trajectories, including all attribute variables,
            which meet the filter specification.

        Examples
        --------
        Filtering all trajectories where Latitude is between 0 N - 20 N.
        >>> trajectories.filter_between('lat', 0, 20)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if (isinstance(min_val, int) or isinstance(min_val, float)) is False:
            raise TypeError("min must be specified as integer or float")

        if (isinstance(max_val, int) or isinstance(max_val, float)) is False:
            raise TypeError("max must be specified as integer or float")

        # ----------------------------------
        # Defining ds, the filtered DataSet.
        # ----------------------------------
        ds = filter_traj(self, filt_type='between', variable=variable, min_val=min_val, max_val=max_val)

        # Returning the subsetted xarray DataSet as a trajectories object -
        # this enables multiple filtering to take place.
        return trajectories(ds)

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
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if (isinstance(min_val, int) or isinstance(min_val, float)) is False:
            raise TypeError("min must be specified as integer or float")

        if (isinstance(max_val, int) or isinstance(max_val, float)) is False:
            raise TypeError("max must be specified as integer or float")

        # ----------------------------------------------------
        # Return indices between min and max with find_traj().
        # ----------------------------------------------------
        # Define indices, storing the indices of trajectory points between
        # min and max.
        indices = find_traj(self, find_type='between', variable=variable, min_val=min_val, max_val=max_val)

        # Returning the tuple of arrays containing indices
        # for satisfactory trajectory points.
        return indices

##############################################################################
# Define filter_equal() method.

    def filter_equal(self, variable, val):
        """
        Filter trajectories with attribute variable equal to value.

        Filtering returns the complete trajectories where the specified
        attribute variable takes the value specified by val.

        Parameters
        ----------
        variable : string
            Name of the variable in the trajectories object.
        val : numeric
            Value variable should equal.

        Returns
        -------
        trajectories object
            Complete trajectories, including all attribute variables,
            which meet the filter specification.

        Examples
        --------
        Filtering all trajectories where Latitude equals 0 N.
        >>> trajectories.filter_equal('lat', 0)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if (isinstance(val, int) or isinstance(val, float)) is False:
            raise TypeError("val must be specified as integer or float")

        # ----------------------------------
        # Defining ds, the filtered DataSet.
        # ----------------------------------
        ds = filter_traj(self, filt_type='equal', variable=variable, val=val)

        # Returning the subsetted xarray DataSet as a trajectories object -
        # this enables multiple filtering to take place.
        return trajectories(ds)

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
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if (isinstance(val, int) or isinstance(val, float)) is False:
            raise TypeError("val must be specified as integer or float")

        # ---------------------------------------------
        # Return indices equal to val with find_traj().
        # ----------------------------------------------
        # Define indices, storing the indices of trajectory points equal to
        # val.
        indices = find_traj(self, find_type='equal', variable=variable, val=val)

        # Returning the tuple of arrays containing indices
        # for satisfactory trajectory points.
        return indices

##############################################################################
# Define compute_dx() method.

    def compute_dx(self, unit='km'):
        """
        Compute particle zonal displacements from trajectories.

        Zonal (x) displacements between particle positions for
        all trajectories are returned as a new DataArray, dx,
        within the trajectories object.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.
        unit : string
            Unit for particle displacement output - default 'km' -
            alternative option - 'm'.

        Returns
        -------
        trajectories object
            Original trajectories object is returned with appended attribute
            variable DataArray containing particle zonal displacements
            with dimensions (traj x obs).

            The first observation (obs) for all trajectories
            (traj) is NaN since the zonal distance from the origin
            of a particle at the origin is not defined.

        Examples
        --------
        Computing zonal displacements for all trajectories.
        >>> trajectories.compute_dx()
        """
        # ------------------
        # Raise exceptions.
        # ------------------
        # Define np.array with available options for particle
        # zonal displacement output units.
        unit_options = ['m', 'km']

        # Raising exception when unavailable unit is specified.
        if unit not in unit_options:
            raise ValueError("invalid unit - use \'m\' or \'km\'")

        # -----------------------------------------
        # Computing dx with compute_displacement().
        # -----------------------------------------
        dx = compute_displacement(self, axis='x', unit=unit)

        # ---------------------
        # Adding dx to DataSet.
        # ---------------------
        # Append zonal displacement DataArray to original DataSet.
        self.data['dx'] = xr.DataArray(dx, dims=["traj", "obs"])
        # Adding attributes to zonal displacement DataArray.
        self.data.dx.attrs = {
                             'long_name': "zonal displacement",
                             'standard_name': "dx",
                             'units': unit,
                             'positive': "eastward"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define compute_dy() method.

    def compute_dy(self, unit='km'):
        """
        Compute particle meridional displacements from trajectories.

        Meridional (y) displacements between particle positions for
        all trajectories are returned as a new DataArray, dy,
        within the trajectories object.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.
        unit : string
            Unit for particle displacement output - default 'km' -
            alternative option - 'm'.

        Returns
        -------
        trajectories object
            Original trajectories object is returned with appended attribute
            variable DataArray containing particle meridional displacements
            with dimensions (traj x obs).

            The first observation (obs) for all trajectories
            (traj) is NaN since the meridional distance from the origin
            of a particle at the origin is not defined.

        Examples
        --------
        Computing meridional displacements for all trajectories.
        >>> trajectories.compute_dy()
        """
        # ------------------
        # Raise exceptions.
        # ------------------
        # Define np.array with available options for particle
        # meridional displacement output units.
        unit_options = ['m', 'km']

        # Raising exception when unavailable unit is specified.
        if unit not in unit_options:
            raise ValueError("invalid unit - use \'m\' or \'km\'")

        # -----------------------------------------
        # Computing dy with compute_displacement().
        # -----------------------------------------
        dy = compute_displacement(self, axis='y', unit=unit)

        # ---------------------
        # Adding dy to DataSet.
        # ---------------------
        # Append meridional displacement DataArray to original DataSet.
        self.data['dy'] = xr.DataArray(dy, dims=["traj", "obs"])
        # Adding attributes to meridional displacement DataArray.
        self.data.dy.attrs = {
                                'long_name': "meridional displacement",
                                'standard_name': "dy",
                                'units': unit,
                                'positive': "northward"
                                }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define compute_dz() method.

    def compute_dz(self, unit='m'):
        """
        Compute particle vertical displacements from trajectories.

        Vertical (z) displacements between particle positions for
        all trajectories are returned as a new DataArray, dz,
        within the trajectories object.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.
        unit : string
            Unit for particle displacement output - default 'm' -
            alternative option - 'km'.

        Returns
        -------
        trajectories object.
        Original trajectories object is returned with appended attribute
        variable DataArray containing particle vertical displacements
        with dimensions (traj x obs).

        The first observation (obs) for all trajectories
        (traj) is NaN since the vertical distance from the origin
        of a particle at the origin is not defined.

        Examples
        --------
        Computing vertical displacements for all trajectories.
        >>> trajectories.compute_dz()
        """
        # ------------------
        # Raise exceptions.
        # ------------------
        # Define np.array with available options for particle
        # vertical displacement output units.
        unit_options = ['m', 'km']

        # Raising exception when unavailable unit is specified.
        if unit not in unit_options:
            raise ValueError("invalid unit - use \'m\' or \'km\'")

        # -----------------------------------------
        # Computing dz with compute_displacement().
        # -----------------------------------------
        dz = compute_displacement(self, axis='z', unit=unit)

        # ---------------------
        # Adding dz to DataSet.
        # ---------------------
        # Append vertical displacement DataArray to original DataSet.
        self.data['dz'] = xr.DataArray(dz, dims=["traj", "obs"])
        # Adding attributes to vertical displacement DataArray.
        self.data.dz.attrs = {
                             'long_name': "vertical displacement",
                             'standard_name': "dz",
                             'units': unit,
                             'positive': "upward"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define compute_u() method.

    def compute_u(self, unit='m/s'):
        """
        Compute particle zonal velocity from trajectories
        zonal displacements.

        Lagrangian zonal (x) velocity components for
        all trajectories are returned as a new DataArray, u,
        within the trajectories object.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.
        unit : string
            Unit for Lagrangian velocity output - default 'm/s' -
            alternative options - 'm/day', 'km/day'.

        Returns
        -------
        trajectories object.
        Original trajectories object is returned with appended attribute
        variable DataArray containing the zonal component of each particle's
        Lagrangian velocity with dimensions (traj x obs).

        The first observation (obs) for all trajectories
        (traj) is NaN since the zonal velocity from the origin
        of a particle at the origin is not defined.

        Examples
        --------
        Computing zonal velocities for all trajectories.
        >>> trajectories.compute_u()
        """
        # ------------------
        # Raise exceptions.
        # ------------------
        # Define np.array with available options for Lagrangian
        # velocity output units.
        unit_options = ['m/s', 'm/day', 'km/day']

        # Raising exception when unavailable unit is specified.
        if unit not in unit_options:
            raise ValueError("invalid unit - use \'m/s\', \'m/day\', \'km/day\'")

        # -----------------------------------------
        # Computing u with compute_velocity().
        # -----------------------------------------
        u = compute_velocity(self, axis='x', unit=unit)

        # ---------------------
        # Adding u to DataSet.
        # ---------------------
        # Append zonal velocity DataArray to original DataSet.
        self.data['u'] = xr.DataArray(u, dims=["traj", "obs"])
        # Adding attributes to zonal velocity DataArray.
        self.data.u.attrs = {
                             'long_name': "zonal velocity",
                             'standard_name': "u",
                             'units': "m/s",
                             'positive': "eastward"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define compute_v() method.

    def compute_v(self, unit='m/s'):
        """
        Compute particle meridional velocity from trajectories
        meridional displacements.

        Lagrangian meridional (y) velocity components for
        all trajectories are returned as a new DataArray, v,
        within the trajectories object.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.
        unit : string
            Unit for Lagrangian velocity output - default 'm/s' -
            alternative options - 'm/day', 'km/day'.

        Returns
        -------
        trajectories object.
        Original trajectories object is returned with appended attribute
        variable DataArray containing the meridional component of each
        particle's Lagrangian velocity with dimensions (traj x obs).

        The first observation (obs) for all trajectories
        (traj) is NaN since the meridional velocity from the origin
        of a particle at the origin is not defined.

        Examples
        --------
        Computing meridional velocities for all trajectories.
        >>> trajectories.compute_v()
        """
        # ------------------
        # Raise exceptions.
        # ------------------
        # Define np.array with available options for Lagrangian
        # velocity output units.
        unit_options = ['m/s', 'm/day', 'km/day']

        # Raising exception when unavailable unit is specified.
        if unit not in unit_options:
            raise ValueError("invalid unit - use \'m/s\', \'m/day\', \'km/day\'")

        # -----------------------------------------
        # Computing v with compute_velocity().
        # -----------------------------------------
        v = compute_velocity(self, axis='y', unit=unit)

        # ---------------------
        # Adding v to DataSet.
        # ---------------------
        # Append meridional velocity DataArray to original DataSet.
        self.data['v'] = xr.DataArray(v, dims=["traj", "obs"])
        # Adding attributes to meridional velocity DataArray.
        self.data.v.attrs = {
                             'long_name': "meridional velocity",
                             'standard_name': "v",
                             'units': "m/s",
                             'positive': "northward"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define compute_w() method.

    def compute_w(self, unit='m/s'):
        """
        Compute particle vertical velocity from trajectories
        vertical displacements.

        Lagrangian vertical (z) velocity components for
        all trajectories are returned as a new DataArray, w,
        within the trajectories object.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.
        unit : string
            Unit for Lagrangian velocity output - default 'm/s' -
            alternative options - 'm/day', 'km/day'.

        Returns
        -------
        trajectories object.
        Original trajectories object is returned with appended attribute
        variable DataArray containing the vertical component of each
        particle's Lagrangian velocity with dimensions (traj x obs).

        The first observation (obs) for all trajectories
        (traj) is NaN since the vertical velocity from the origin
        of a particle at the origin is not defined.

        Examples
        --------
        Computing vertical velocities for all trajectories.
        >>> trajectories.compute_w()
        """
        # ------------------
        # Raise exceptions.
        # ------------------
        # Define np.array with available options for Lagrangian
        # velocity output units.
        unit_options = ['m/s', 'm/day', 'km/day']

        # Raising exception when unavailable unit is specified.
        if unit not in unit_options:
            raise ValueError("invalid unit - use \'m/s\', \'m/day\', \'km/day\'")

        # -----------------------------------------
        # Computing w with compute_velocity().
        # -----------------------------------------
        w = compute_velocity(self, axis='z', unit=unit)

        # ---------------------
        # Adding w to DataSet.
        # ---------------------
        # Append vertical velocity DataArray to original DataSet.
        self.data['w'] = xr.DataArray(w, dims=["traj", "obs"])
        # Adding attributes to vertical velocity DataArray.
        self.data.w.attrs = {
                             'long_name': "vertical velocity",
                             'standard_name': "w",
                             'units': unit,
                             'positive': "upwards"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define compute_dist() method.

    def compute_dist(self, cumsum_dist=False, unit='km'):
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
            Unit for distance travelled output - default is 'km' -
            alternative option 'm'.

        Returns
        -------
        trajectories object.
        Original trajectories object is returned with appended attribute
        variable DataArray containing the distance travelled by each
        particle along it's trajectory with dimensions (traj x obs).

        The first observation (obs) for all trajectories
        (traj) is NaN since the (cumulative) distance
        from the origin of a particle at the origin is
        not defined.

        Examples
        --------
        Computing distance travelled by particles for all trajectories,
        specifying cumulative distance as False and unit as default 'km'.
        >>> trajectories.compute_dist()
        """
        # ------------------
        # Raise exceptions.
        # ------------------
        # Define np.array with available options for
        # distance output units.
        unit_options = ['m', 'km']

        # Raising exception when unavailable unit is specified.
        if unit not in unit_options:
            raise ValueError("invalid unit - use \'m\', \'km\'")

        # -------------------------------------------
        # Computing distance with compute_distance().
        # -------------------------------------------
        dist = compute_distance(self, cumsum_dist=cumsum_dist, unit=unit)

        # -----------------------
        # Adding dist to DataSet.
        # -----------------------
        if cumsum_dist is True:
            # Append distance DataArray to original DataSet.
            self.data['cumdist'] = xr.DataArray(dist, dims=["traj", "obs"])

            # Adding attributes to cumdist DataArray.
            self.data.cumdist.attrs = {
                                'long_name': "cumulative distance",
                                'standard_name': "cumdist",
                                'units': unit,
                                }
        else:
            # Append distance DataArray to original DataSet.
            self.data['dist'] = xr.DataArray(dist, dims=["traj", "obs"])

            # Adding attributes to dist DataArray.
            self.data.dist.attrs = {
                                'long_name': "distance",
                                'standard_name': "dist",
                                'units': unit,
                                }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define get_start_time() method.

    def get_start_time(self):
        """
        Returns times when particles are released (start of trajectory).

        The start time (ns) is given as the time elapsed since the begining
        of the simulation and is returned for all trajectories as a new
        DataArray.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.

        Returns
        -------
        DataSet.
            Original DataSet is returned with appended attribute
            variable t_start (ns) DataArray containing the
            release time of each particle with dimension (traj).

        Examples
        --------
        Get release times for all trajectories.
        >>> trajectories.get_start_time()
        """
        # ------------------------------------------
        # Return start times with get_start_time().
        # ------------------------------------------
        t_start = get_start_time(self)

        # ---------------------------
        # Adding t_start to DataSet.
        # ---------------------------
        # Append t_start DataArray to original DataSet.
        self.data['t_start'] = xr.DataArray(t_start, dims=["traj"])
        # Adding attributes to t_start DataArray.
        self.data.t_start.attrs = {
                             'long_name': "release time",
                             'standard_name': "t_start",
                             'units': "ns"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define get_start_loc() method.

    def get_start_loc(self):
        """
        Returns locations where particles are released (start of trajectory).

        The start locations are divided into lon_start and lat_start which
        are returned for all trajectories as new DataArrays.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.

        Returns
        -------
        DataSet.
            Original DataSet is returned with appended attribute
            variables lat_start and lon_start DataArray containing
            the release latitude and longitude of each particle
            with dimension (traj).

        Examples
        --------
        Get release locations for all trajectories.
        >>> trajectories.get_start_loc()
        """
        # ----------------------------------------------
        # Return start locations with get_start_loc().
        # ----------------------------------------------
        lat_start, lon_start = get_start_loc(self)

        # --------------------------------
        # Adding lat/lon_start to DataSet.
        # --------------------------------
        # Append lat_start DataArray to original DataSet.
        self.data['lat_start'] = xr.DataArray(lat_start, dims=["traj"])
        # Adding attributes to lat_start DataArray.
        self.data.lat_start.attrs = {
                             'long_name': "release latitude",
                             'standard_name': "lat_start",
                             'units': "degrees north"
                             }

        # Append lon_start DataArray to original DataSet.
        self.data['lon_start'] = xr.DataArray(lon_start, dims=["traj"])
        # Adding attributes to lon_start DataArray.
        self.data.lon_start.attrs = {
                             'long_name': "release longitude",
                             'standard_name': "lon_start",
                             'units': "degrees east"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define get_end_time() method.

    def get_end_time(self):
        """
        Returns times when particles exit the system (end of trajectory).

        The end time (ns) is given as the time elapsed since the begining
        of the simulation and is returned for all trajectories as a new
        DataArray.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.

        Returns
        -------
        DataSet.
            Original DataSet is returned with appended attribute
            variable t_end (ns) DataArray containing the
            exit time of each particle with dimension (traj).

        Examples
        --------
        Get exit times for all trajectories.
        >>> trajectories.get_end_time()
        """
        # ------------------------------------------
        # Return exit times with get_end_time().
        # ------------------------------------------
        t_end = get_end_time(self)

        # ---------------------------
        # Adding t_end to DataSet.
        # ---------------------------
        # Append t_end DataArray to original DataSet.
        self.data['t_end'] = xr.DataArray(t_end, dims=["traj"])
        # Adding attributes to t_end DataArray.
        self.data.t_end.attrs = {
                             'long_name': "exit time",
                             'standard_name': "t_end",
                             'units': "ns"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define get_end_loc() method.

    def get_end_loc(self):
        """
        Returns locations where particles exit the system
        (start of trajectory).

        The end locations are divided into lon_end and lat_end which
        are returned for all trajectories as new DataArrays.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.

        Returns
        -------
        DataSet.
            Original DataSet is returned with appended attribute
            variables lat_end and lon_end DataArray containing
            the exit latitude and longitude of each particle
            with dimension (traj).

        Examples
        --------
        Get exit locations for all trajectories.
        >>> trajectories.get_end_loc()
        """
        # -----------------------------------------
        # Return exit locations with get_end_loc().
        # -----------------------------------------
        lat_end, lon_end = get_end_loc(self)

        # --------------------------------
        # Adding lat/lon_end to DataSet.
        # --------------------------------
        # Append lat_end DataArray to original DataSet.
        self.data['lat_end'] = xr.DataArray(lat_end, dims=["traj"])
        # Adding attributes to lat_end DataArray.
        self.data.lat_end.attrs = {
                             'long_name': "exit latitude",
                             'standard_name': "lat_end",
                             'units': "degrees north"
                             }

        # Append lon_end DataArray to original DataSet.
        self.data['lon_end'] = xr.DataArray(lon_end, dims=["traj"])
        # Adding attributes to lon_end DataArray.
        self.data.lon_end.attrs = {
                             'long_name': "exit longitude",
                             'standard_name': "lon_end",
                             'units': "degrees east"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define get_duration() method.

    def get_duration(self):
        """
        Returns time taken for particles to exit the system following release
        (duration of trajectory).

        The duration (ns) is given as the time elapsed between the release of a
        particle and it's exit from the sytem and is returned for all
        trajectories as a new DataArray.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.

        Returns
        -------
        DataSet.
            Original DataSet is returned with appended attribute
            variable t_total (ns) DataArray containing the
            duration each particle is present in the system,
            with dimension (traj).

        Examples
        --------
        Get duration for all trajectories.
        >>> trajectories.get_duration()
        """
        # ------------------------------------
        # Return duration with get_duration().
        # ------------------------------------
        t_total = get_duration(self)

        # ---------------------------
        # Adding t_total to DataSet.
        # ---------------------------
        # Append t_total DataArray to original DataSet.
        self.data['t_total'] = xr.DataArray(t_total, dims=["traj"])
        # Adding attributes to t_total DataArray.
        self.data.t_total.attrs = {
                             'long_name': "duration",
                             'standard_name': "t_total",
                             'units': "ns"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define get_seed() method.

    def get_seed(self):
        """
        Returns seeding level when particles are released (start
        of trajectory).

        The seeding level, an integer between 1 and the total no. of seeding
        levels, marks when a particle is released into the system and is
        returned for all trajectories as a new DatatArray.

        Parameters
        ----------
        self : trajectories object
        Trajectories object passed from trajectories class method.

        Returns
        -------
        DataSet.
            Original DataSet is returned with appended attribute
            variable seed_level DataArray containing the seed
            level for each particle released, with dimension (traj).

        Examples
        --------
        Get seed levels for all trajectories.
        >>> trajectories.get_seed().
        """
        # ------------------------------------
        # Return seed levels with get_seed().
        # ------------------------------------
        seed_level = get_seed(self)

        # -----------------------------
        # Adding seed_level to DataSet.
        # -----------------------------
        # Append seed_level DataArray to original DataSet.
        self.data['seed_level'] = xr.DataArray(seed_level, dims=["traj"])
        # Adding attributes to seed_level DataArray.
        self.data.seed_level.attrs = {
                             'long_name': "seeding level",
                             'standard_name': "seed_level",
                             'units': "none"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define get_max() function.

    def get_max(self, variable):
        """
        Returns maximum value of a specified variable for each trajectory.

        The maximum value of the variable is returned for all trajectories
        as an ndarray.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.
        variable : string
            Name of the variable in the trajectories object.

        Returns
        -------
        DataSet.
            Original DataSet is returned with appended attribute
            variable {variable}_max DataArray containing the max
            values along each trajectory, with dimension (traj).

        Examples
        --------
        Get the maximum temperature along each trajectory.
        >>>  trajectories.get_max('temp').
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        # -------------------------------------------------
        # Returning max values of variable with get_vals().
        # -------------------------------------------------
        max_values = get_val(self=self, variable=variable, get_max=True)

        # -----------------------------
        # Adding max_values to DataSet.
        # -----------------------------
        # Defining std. name of max_values using specified variable.
        std_name = variable + "_max"
        # Defining long name of max_values using specified variable.
        long_name = "maximum " + variable
        # Append max_values DataArray to original DataSet.
        self.data[std_name] = xr.DataArray(max_values, dims=["traj"])
        # Adding attributes to max_values DataArray.
        self.data[variable].attrs = {
                                'long_name': long_name,
                                'standard_name': std_name,
                                'units': self.data[variable].attrs['units']
                                }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define get_min() function.

    def get_min(self, variable):
        """
        Returns minimum value of a specified variable for each trajectory.

        The minimum value of the variable is returned for all trajectories
        as an ndarray.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.
        variable : string
            Name of the variable in the trajectories object.

        Returns
        -------
        DataSet.
            Original DataSet is returned with appended attribute
            variable {variable}_max DataArray containing the min
            values along each trajectory, with dimension (traj).

        Examples
        --------
        Get the maximum temperature along each trajectory.
        >>>  trajectories.get_max('temp').
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        # -------------------------------------------------
        # Returning min values of variable with get_vals().
        # -------------------------------------------------
        min_values = get_val(self=self, variable=variable, get_max=False)

        # -----------------------------
        # Adding min_values to DataSet.
        # -----------------------------
        # Defining std. name of min_values using specified variable.
        std_name = variable + "_min"
        # Defining long name of min_values using specified variable.
        long_name = "minimum " + variable
        # Append min_values DataArray to original DataSet.
        self.data[std_name] = xr.DataArray(min_values, dims=["traj"])
        # Adding attributes to min_values DataArray.
        self.data[variable].attrs = {
                                'long_name': long_name,
                                'standard_name': std_name,
                                'units': self.data[variable].attrs['units']
                                }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)


##############################################################################
# Testing with ORCA01 Preliminary Data.
traj = trajectories(xr.open_dataset('ORCA1-N406_TRACMASS_output_run.nc'))
plot_trajectories(traj.get_seed().filter_equal('seed_level', val=1), col_var='temp')