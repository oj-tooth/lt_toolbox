##############################################################################
# trajectories.py
#
# Description:
# Defines the trajectories Class from a .nc file containing atmosphere
# ocean parcel trajectories (and accompanying tracers), stored following
# CF-conventions implemented with the NCEI trajectory template.
#
# Last Edited:
# 2021/01/06
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.

import xarray as xr
import numpy as np
from .get_utils import get_start_time, get_start_loc, get_end_time, get_end_loc, get_duration, get_minmax, get_val
from .add_utils import add_seed, add_id, add_var
from .filter_utils import filter_traj
from .compute_utils import compute_displacement, compute_velocity, compute_distance, compute_probability_distribution


##############################################################################
# Define trajectories Class.


class trajectories:

    # Importing methods for finding indices in trajectories object.
    from .find_utils import find_between, find_equal, find_polygon
    # Importing methods for cartesian plotting with matplotlib.
    from .plot_utils import plot_timeseries, plot_ts_diagram, plot_variable
    # Importing methods for geospatial mapping with Cartopy.
    from .map_utils import map_trajectories, map_probability, map_property

    def __init__(self, ds):
        """
        Create a trajectories object from an xarray DataSet.

        Parameters
        ----------
        ds: DataSet
            Trajectories stored in xarray DataSet following
            CF-conventions implemented with the NCEI trajectory
            template.

        Returns
        --------
        trajectories object
            Complete trajectories, including all attribute variables
            contained in DataSet, ds.

        Examples
        --------
        Creating trajectories object, traj, with output_file.nc file.

        >>> trajectories = trajectories(xr.open_dataset('output_file.nc'))

        Note
        ----
        For NCEI trajectory template see:
        https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl
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
        if np.issubdtype(self.data['time'].values.dtype, np.datetime64) is True:
            raise TypeError("time already exists with dtype = \'datetime64[ns]\'")

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

    def filter_between(self, variable, min_val, max_val, drop=False):
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
        variable : string
            Name of the variable in the trajectories object.
        min_val : numeric
            Minimum value variable should equal or be greater than.
        max_val : numeric
            Maximum value variable should equal or be less than.
        drop : boolean
            Determines if fitered trajectories should be returned as a
            new trajectories object (False) or instead dropped from the
            existing trajectories object (True).

        Returns
        -------
        trajectories object
            Complete trajectories, including all attribute variables,
            which meet the filter specification.

        Examples
        --------
        Filtering all trajectories where Latitude is between 0 N - 20 N.

        >>> trajectories.filter_between('lat', 0, 20, drop=False)

        Filtering trajectory observations between two dates using numpy
        datetime64.

        >>> tmin = np.datetime64('2000-01-01')
        >>> tmax = np.datetime64('2000-03-01')
        >>> trajectories.filter_between('time', tmin, tmax)

        Filtering trajectory observations with time using numpy timedelta64.

        >>> tmin = np.timedelta64(0, 'D')
        >>> tmax = np.timedelta64(90, 'D')
        >>> trajectories.filter_between('time', tmin, tmax)
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

        if isinstance(drop, bool) is False:
            raise TypeError("drop must be specified as a boolean")

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

        # ----------------------------------
        # Defining ds, the filtered DataSet.
        # ----------------------------------
        ds = filter_traj(self, filt_type='between', variable=variable, val=None, min_val=min_val, max_val=max_val, polygon=None, drop=drop)

        # Returning the subsetted xarray DataSet as a trajectories object -
        # this enables multiple filtering to take place.
        return trajectories(ds)

##############################################################################
# Define filter_equal() method.

    def filter_equal(self, variable, val, drop=False):
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
        trajectories object
            Complete trajectories, including all attribute variables,
            which meet the filter specification.

        Examples
        --------
        Filtering all trajectories where Latitude equals 0 N.

        >>> trajectories.filter_equal('lat', 0, drop=False)

        Filtering trajectory observations for one date using numpy
        datetime64.

        >>> tval = np.datetime64('2000-03-01')
        >>> trajectories.filter_equal('time', tval)

        Filtering trajectory observations for multiple times using
        numpy timedelta64.

        >>> tval = [np.timedelta64(10, 'D'), np.timedelta64(20, 'D')]
        >>> trajectories.filter_equal('time', tval)
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

        if isinstance(drop, bool) is False:
            raise TypeError("drop must be specified as a boolean")

        # For non-time variables.
        if variable != 'time':
            # For list of values, val, elements must be int or float type.
            if isinstance(val, list) is True:
                if all(isinstance(value, (int, float, np.int64, np.float64)) for value in val) is False:
                    raise TypeError("contents of val must be specified as integers or floats")

            # For single value, val.
            else:
                if isinstance(val, (int, float, np.int64, np.float64)) is False:
                    raise TypeError("val must be specified as integer or float")

        # For time variable.
        else:
            # For list of values, val, elements must be numpy datetime64 or
            # timedelta64 format only.
            if isinstance(val, list) is True:
                if all(isinstance(value, (np.datetime64, np.timedelta64)) for value in val) is False:
                    raise TypeError("contents of val must be specified as datetime64 or timedelta64")

            # For single value, val.
            else:
                if isinstance(val, (np.datetime64, np.timedelta64)) is False:
                    raise TypeError("val must be specified as datetime64 or timedelta64")

        # ----------------------------------
        # Defining ds, the filtered DataSet.
        # ----------------------------------
        ds = filter_traj(self, filt_type='equal', variable=variable, val=val, min_val=None, max_val=None, polygon=None, drop=drop)

        # Returning the subsetted xarray DataSet as a trajectories object -
        # this enables multiple filtering to take place.
        return trajectories(ds)

##############################################################################
# Define filter_polygon() method.

    def filter_polygon(self, polygon, drop=False):
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
        drop : boolean
            Determines if fitered trajectories should be returned as a
            new trajectories object (False) or instead dropped from the
            existing trajectories object (True).

        Returns
        -------
        trajectories object
            Complete trajectories, including all attribute variables,
            which meet the filter specification.

        Examples
        --------
        Filtering all trajectories which intersect a simple polygon, square.

        >>> square = [(-40, 30), (-40, 35), (-30, 35), (-30, 30), (-40, 30)]
        >>> trajectories.filter_polygon(square, drop=False)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(drop, bool) is False:
            raise TypeError("drop must be specified as a boolean")

        if isinstance(polygon, list) is False:
            raise TypeError("polygon must be specified as a list of tuples")

        # ----------------------------------
        # Defining ds, the filtered DataSet.
        # ----------------------------------
        ds = filter_traj(self, filt_type='polygon', variable=None, val=None, min_val=None, max_val=None, polygon=polygon, drop=drop)

        # Returning the subsetted xarray DataSet as a trajectories object -
        # this enables multiple filtering to take place.
        return trajectories(ds)

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
# Define compute_distance() method.

    def compute_distance(self, cumsum_dist=False, unit='km'):
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

        >>> trajectories.compute_distance()
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
# Define compute_probability() method.

    def compute_probability(self, bin_res, method, gf_sigma=None, group_by=None):
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

        Examples
        --------
        Computing the Lagrangian probability distribution using all
        particle positions for particles released at seed_level 1.

        >>> trajectories.filter_equal('seed_level', 1).compute_probability(bin_res=1, method='pos')

        Computing the Lagrangian probability density distribution using
        all particle positions with a Gaussian filter for particles released
        at seed_levels 1 to 5.

        >>> trajectories.filter_between('seed_level', 1, 5).compute_probability(bin_res=1, method='pos-gauss', gf_sigma=1, group_by='seed_level')
        """
        # ------------------
        # Raise exceptions.
        # ------------------
        if group_by is not None:
            # Defining list of variables contained in data.
            variables = list(self.data.variables)

            if group_by not in variables:
                raise ValueError("variable: \'" + group_by + "\' not found in Dataset")

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
            nav_lon, nav_lat, probability = compute_probability_distribution(self, lat_lims=lat_lims, lon_lims=lon_lims, bin_res=bin_res, method=method, gf_sigma=gf_sigma)

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
                nav_lon, nav_lat, p = compute_probability_distribution(traj, lat_lims=lat_lims, lon_lims=lon_lims, bin_res=bin_res, method=method, gf_sigma=gf_sigma)

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
        # Defining list of variables contained in data.
        variables = list(self.data.variables)

        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if variable not in variables:
            raise ValueError("variable: \'" + variable + "\' not found in Dataset")

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
        # Defining list of variables contained in data.
        variables = list(self.data.variables)

        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if variable not in variables:
            raise ValueError("variable: \'" + variable + "\' not found in Dataset")

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
        # Defining list of variables contained in data.
        variables = list(self.data.variables)

        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if variable not in variables:
            raise ValueError("variable: \'" + variable + "\' not found in Dataset")

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
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(data, np.ndarray) is False:
            raise TypeError("data must be provided as an ndarray")
        if isinstance(attributes, dict) is False:
            raise TypeError("variable attributes must be provided as a dictionary")

        # -----------------------------------------
        # Returning updated DataSet with add_var().
        # -----------------------------------------
        self.data = add_var(self=self, data=data, attrs=attributes)

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)
