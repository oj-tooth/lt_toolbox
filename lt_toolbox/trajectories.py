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
import matplotlib.pyplot as plt
from get_utils import get_start_time, get_start_loc, get_end_time, get_end_loc, get_duration, get_minmax, get_val
from add_utils import add_seed, add_id, add_var
from filter_utils import filter_traj
from find_utils import find_traj
from compute_utils import compute_displacement, compute_velocity, compute_distance
from plot_utils import plot_timeseries, plot_ts_diagram, plot_variable
from map_utils import map_trajectories

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

        The start locations are divided into lon_start, lat_start and z_start
        which are returned for all trajectories as new DataArrays.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.

        Returns
        -------
        DataSet.
            Original DataSet is returned with appended attribute
            variables lat_start, lon_start, z_start DataArray containing
            the release latitude, longitude and depth of each particle
            with dimension (traj).

        Examples
        --------
        Get release locations for all trajectories.
        >>> trajectories.get_start_loc()
        """
        # ----------------------------------------------
        # Return start locations with get_start_loc().
        # ----------------------------------------------
        lat_start, lon_start, z_start = get_start_loc(self)

        # -----------------------------------
        # Adding lat/lon/z_start to DataSet.
        # -----------------------------------
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

        # Append z_start DataArray to original DataSet.
        self.data['z_start'] = xr.DataArray(z_start, dims=["traj"])
        # Adding attributes to z_start DataArray.
        self.data.z_start.attrs = {
                             'long_name': "release depth",
                             'standard_name': "z_start",
                             'units': "meters",
                             'positive': "upwards"
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

        The end locations are divided into lon_end, lat_end and
        z_end which are returned for all trajectories as new
        DataArrays.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.

        Returns
        -------
        DataSet.
            Original DataSet is returned with appended attribute
            variables lat_end, lon_end and z_end DataArray containing
            the exit latitude, longitude and depth of each particle
            with dimension (traj).

        Examples
        --------
        Get exit locations for all trajectories.
        >>> trajectories.get_end_loc()
        """
        # -----------------------------------------
        # Return exit locations with get_end_loc().
        # -----------------------------------------
        lat_end, lon_end, z_end = get_end_loc(self)

        # --------------------------------
        # Adding lat/lon/z_end to DataSet.
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

        # Append z_end DataArray to original DataSet.
        self.data['z_end'] = xr.DataArray(z_end, dims=["traj"])
        # Adding attributes to z_end DataArray.
        self.data.z_end.attrs = {
                             'long_name': "exit depth",
                             'standard_name': "z_end",
                             'units': "meters",
                             'positive': "upwards"
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
# Define get_value() function.

    def get_value(self, variable, time_level):
        """
        Returns the value of a specified variable at a specified
        time level for each trajectory.

        The values of the specified variable are returned for all
        trajectories for a time level specified in the form
        'YYYY-MM-DD' as a new DataArray.

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
        Get the value of temperature for each trajectory at time
        level 2000-01-31. Note that we must convert time to datetime64
        format before using .get_value().
        >>>  trajectories.use_datetime(start_time='2000-01-01').get_value('temp', '2000-01-31')
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if isinstance(time_level, str) is False:
            raise TypeError("time_level must be specified as a string in the format YYYY-MM-DD")

        # ----------------------------------------------------------
        # Returning values of variable at time level with get_val().
        # ----------------------------------------------------------
        values = get_val(self=self, variable=variable, time_level=time_level)

        # -------------------------
        # Adding values to DataSet.
        # -------------------------
        # Defining std. name of values using specified variable.
        std_name = variable + "_i"
        # Defining long name of values using specified variable.
        long_name = variable + " at " + time_level
        # Append min_values DataArray to original DataSet.
        self.data[std_name] = xr.DataArray(values, dims=["traj"])
        # Adding attributes to min_values DataArray.
        self.data[std_name].attrs = {
                                'long_name': long_name,
                                'standard_name': std_name,
                                'units': self.data[variable].attrs['units']
                                }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define get_max() function.

    def get_max(self, variable):
        """
        Returns maximum value of a specified variable for each trajectory.

        The maximum value of the variable is returned for all trajectories
        as a new DataArray.

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

        # ---------------------------------------------------
        # Returning max values of variable with get_minmax().
        # ---------------------------------------------------
        max_values = get_minmax(self=self, variable=variable, get_max=True)

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
        self.data[std_name].attrs = {
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
        as a new DataArray.

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

        # ---------------------------------------------------
        # Returning min values of variable with get_minmax().
        # ---------------------------------------------------
        min_values = get_minmax(self=self, variable=variable, get_max=False)

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
        self.data[std_name].attrs = {
                                'long_name': long_name,
                                'standard_name': std_name,
                                'units': self.data[variable].attrs['units']
                                }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define add_seed() method.

    def add_seed(self):
        """
        Adds seeding level when particles are released (start
        of trajectory) as a new attribute variable.

        The seeding level, an integer between 1 and the total no. of seeding
        levels, marks when a particle is released into the system and is
        returned for all trajectories as a new DataArray.

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
        >>> trajectories.add_seed().
        """
        # ------------------------------------
        # Return seed levels with add_seed().
        # ------------------------------------
        seed_level = add_seed(self)

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
# Define add_id() method.

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
        DataSet.
            Original DataSet is returned with appended attribute
            variable seed_level DataArray containing the id
            for each particle released, with dimension (traj).

        Examples
        --------
        Get trajectory id for all trajectories.
        >>> trajectories.add_id().
        """
        # -----------------------------------
        # Return trajectory id with add_id().
        # -----------------------------------
        traj_id = add_id(self)

        # --------------------------
        # Adding traj_id to DataSet.
        # --------------------------
        # Append traj_id DataArray to original DataSet.
        self.data['id'] = xr.DataArray(traj_id, dims=["traj"])
        # Adding attributes to traj_id DataArray.
        self.data.id.attrs = {
                             'long_name': "trajectory id",
                             'standard_name': "id",
                             'units': "none"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define add_variable() method.

    def add_variable(self, data, attributes):
        """
        Adds a new variable to the existing trajectories object.

        The variable data must be provided as an ndarray with dimensions
        (traj) / (obs) / (traj x obs) and the attributes provided as a
        dictionary.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.
        data : ndarray
            values of new variable to be added to the trajectories object
            DataSet.
        attributes : dict
            the attributes of the new variable, at a minimum -'long_name',
            'standard_name' and 'units' should be included. The standard
            name will be assigned as the attribute variable name.

        Returns
        -------
        trajectories object
            Original trajectories object is returned with new attribute
            variable DataArray appended, dimensions are either (traj) /
            (obs) / (traj x obs).
        """
        # -----------------------------------------
        # Returning updated DataSet with add_var().
        # -----------------------------------------
        self.data = add_var(self=self, data=data, attrs=attributes)

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define map_trajectories() method.

    def map_trajectories(self, col_variable=None):
        """
        Maps surface trajectories (latitudes and longitudes) of
        particles on an orthographic projection of Earth's surface.

        Latitudes and longitudes of particle positions are connected
        to visualise surface trajectories. Trajectories can also be
        optionally coloured according to a specified scalar variable
        given by col_variable.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.
        col_variable : string
            Name of variable in the trajectories object to colour
            mapped trajectories, default is None.

        Returns
        -------
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(col_variable, str) is False:
            raise TypeError("col_variable must be specified as a string")

        # ------------------------------------------------
        # Return trajectories map with map_trajectories().
        # ------------------------------------------------
        map_trajectories(self=self, col_variable=col_variable)

        return

##############################################################################
# Define plot_timeseries() method.

    def plot_timeseries(self, variable, col_variable=None):
        """
        Plots time series of specified attribute variable as it
        evolves along each particle's trajectory.

        Time series can also be optionally coloured according to
        a specified (1-dimensional) scalar variable given by
        col_variable.

        When col_variable is not specified, the trajectory id of
        each time series is included in a legend.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.
        variable : string
            Name of the variable in the trajectories object.
        col_variable : string
            Name of variable in the trajectories object to colour
            plotted trajectories - must be 1-dimensional - default
            is None.

        Returns
        -------
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if (isinstance(col_variable, str) or col_variable is None) is False:
            raise TypeError("col_variable must be specified as a string")

        # ------------------------
        # Return time series plot.
        # ------------------------
        plot_timeseries(self=self, variable=variable, col_variable=col_variable)

        return

##############################################################################
# Define plot_ts_diagram() method.

    def plot_ts_diagram(self, col_variable=None):
        """
        Plots temperature-salinity diagram as a scatter plot of
        temp (y) and salinity (x) for every point along each
        particle's trajectory.

        Plotted points can be optionally coloured according to
        a specified (1-dimensional) scalar variable given by
        col_variable.

        When col_variable is not specified, points are coloured
        according to their trajectory id with an accompanying legend.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.
        col_variable : string
            Name of variable in the trajectories object to colour
            scatter points - must be 1-dimensional - default
            is None.

        Returns
        -------
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if (isinstance(col_variable, str) or col_variable is None) is False:
            raise TypeError("col_variable must be specified as a string")

        # -----------------------------
        # Return temp-salinity diagram.
        # -----------------------------
        plot_ts_diagram(self=self, col_variable=col_variable)

        return

##############################################################################
# Define plot_variable() method.

    def plot_variable(self, variable, plane, seed_level, time_level, cmap='coolwarm'):
        """
        2-dimensional Cartesian contour plot of a specified variable
        at a specific time along particle trajectories.

        Follows the specification of the trajectory map of Betten et
        al. (2017); values of the variable are displayed on particle
        initial grid locations at the time of seeding.

        When cmap is not specified, the default colour map is 'coolwarm'
        - a diverging colormap.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.
        variable : string
            Name of the variable in the trajectories object.
        plane : string
            Seeding plane from which particles are released - options
            are 'xz' zonal-vertical and 'yz' meridional-vertical.
        seed_level : integer
            Seeding level when particles are released.
        time_level : string
            Time level along trajectories to plot variable.
        cmap : string
            A colormap instance or registered colormap name.

        Returns
        -------
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")
        if isinstance(plane, str) is False:
            raise TypeError("plan must be specified as a string - options are \'xz\' or \'yz\'")
        if isinstance(seed_level, int) is False:
            raise TypeError("seed_level must be specified as an integer")
        if isinstance(time_level, str) is False:
            raise TypeError("time_level must be specified as a string in the format \'YYYY-MM-DD\'")
        if isinstance(cmap, str) is False:
            raise TypeError("cmap must be specified as a string")

        # ------------------------
        # Return 2-D contour plot.
        # ------------------------
        plot_variable(self=self, variable=variable, plane=plane, seed_level=seed_level, time_level=time_level, cmap=cmap)

        return

##############################################################################
# Testing with ORCA01 Preliminary Data.
traj = trajectories(xr.open_dataset('ORCA1-N406_TRACMASS_complete.nc'))
traj = traj.use_datetime('2000-01-01')
traj.plot_variable('temp', 'xz', 11, '2007-07-23')
