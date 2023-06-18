##############################################################################
# trajectory_frame.py
#
# Description:
# Defines the TrajFrame Class from a .csv or .parquet file containing atmosphere
# ocean parcel trajectories (and accompanying tracers) stored in a tabular format.
#
# Last Edited:
# 2023/17/06
#
# Last Edited By:
# Ollie Tooth
#
# Contact:
# oliver.tooth@env-res.ox.ac.uk
#
##############################################################################
# Importing relevant packages.

import polars as pl
import numpy as np

# from .filter_frame_utils import filter_traj_between, filter_traj_equal, filter_traj_polygon
# from .compute_frame_utils import compute_displacement, compute_velocity, compute_distance, compute_probability_distribution, compute_res_time, compute_trans_time


##############################################################################
# Define TrajFrame Class.


class TrajFrame:

    # Importing methods for finding indices in TrajFrame object.
    # from .find_frame_utils import find_between, find_equal, find_polygon
    # Importing methods for cartesian plotting with matplotlib.
    # from .plot_frame_utils import plot_timeseries, plot_ts_diagram, plot_variable
    # Importing methods for geospatial mapping with Cartopy.
    # from .map_frame_utils import map_trajectories, map_probability, map_property
    # Importing methods for sensitivity analysis.
    # from .compute_frame_utils import compute_fuv

    def __init__(self, df, attrs, df_summary=None):
        """
        Create a TrajFrame object from a polars DataFrame.

        Parameters
        ----------
        df: DataFrame
            Complete Lagrangian trajectories stored in polars DataFrame.
        attrs: dictionary
            Attributes metadata for columns variables contained in the trajectory
            DataFrame, df.
        df_summary : DataFrame
            Output data stored for each trajectory in trajectory DataFrame, df.
            Default value is None since original DataFrame is assumed to
            contain multiple rows for each trajectory. An exception is made
            when the original DataFrame contains only a single row per
            trajectory. In this case, df and df_summary are equivalent.

        Returns
        --------
        TrajFrame object
            Complete trajectories, including all column variables contained
            in trajectory DataFrame, df. Summary data stored for each trajectory
            in summary DataFrame, df_summary.

        Examples
        --------
        Creating TrajFrame object, traj, with output_file.csv file.

        >>> attrs = {"id": "water parcel unique ID",
                     "lon": "longitude (deg E)",
                     "lat": "latitude (deg N)",
                     "depth": "depth (m)",
                     "temp": "potential temperature (C)",
                     }
        >>> df = pl.read_csv('output_file.csv')
        >>> trajectories = TrajFrame(df=df, attrs={"id"})

        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Raise error if df is not an polars DataFrame.
        if (str(type(df)) == "<class 'polars.internals.dataframe.frame.DataFrame'>") is False:
            raise TypeError("df must be specified as a polars DataFrame")

        # Raise error if attrs is not a dictionary.
        if isinstance(attrs, dict) is False:
            raise TypeError("attrs must be specified as a dictionary")

        # Defining list of column variables contained in trajectory DataFrame.
        col_variables = df.columns

        # Raise error if attrs does not contain all column variables.
        if list(attrs.keys()) != col_variables:
            raise ValueError("variable(s) missing from attrs")

        # Raise error if the any core column variable is absent from trajectory DataFrame.
        if 'id' not in col_variables:
            raise ValueError("core variable missing from Dataset: \'id\'")
        if 'subvol' not in col_variables:
            raise ValueError("core variable missing from Dataset: \'subvol\'")
        if 'time' not in col_variables:
            raise ValueError("core variable missing from Dataset: \'time\'")
        if ('y' not in col_variables) & ('lat' not in col_variables):
            raise ValueError("core variable missing from Dataset: \'y\' or \'lat\'")
        if ('x' not in col_variables) & ('lon' not in col_variables):
            raise ValueError("core variable missing from Dataset: \'x\' or \'lon\'")
        if ('z' not in col_variables) & ('depth' not in col_variables):
            raise ValueError("core variable missing from Dataset: \'z\' or \'depth\'")

        # -----------------------------------------------------
        # Storing input DataFrame as trajectory_data attribute.
        # -----------------------------------------------------
        # Defining data as input DataFrame containing one or more
        # rows per trajectory.
        self.trajectory_data = df

        # -----------------------------------------------
        # Metadata for input DataFrame column variables.
        # -----------------------------------------------
        # For improved useability, store column variables metadata,
        # storing them as an attribute in the class.

        # Defining attributes as metadata for column variables of DataFrame.
        self.attrs = attrs

        # -------------------------------------------------
        # Storing ouput DataFrame as output_data attribute.
        # -------------------------------------------------
        # Exception when input DataFrame contains only a single row per trajectory.
        if self.trajectory_data.shape[0] == self.trajectory_data["id"].unique().shape[0]:
            # In this case input DataFrame df is equivalent to output DataFrame df_summary.
            self.summary_data = df
        elif df_summary is not None:
            # Defining output_data as output DataFrame containing a single
            # row per trajectory.
            self.summary_data = df_summary
        else:
            # Add default summary DataFrame including ID and initial subvol variables to summary_data.
            self.summary_data = self.trajectory_data.groupby(pl.col("id")).agg(pl.col('subvol').first())

##############################################################################
# Define print() method.

    def __str__(self):
        # Return Dimension and Data Variables contained within
        # TrajFrame object.
        return f"<TrajFrame object>\n\n------------------------------------------\n[Trajectory DataFrame]\nObservations: {self.trajectory_data.shape[0]}\nVariables: {self.trajectory_data.shape[1]}\n------------------------------------------ \n{self.trajectory_data.glimpse}\n"

##############################################################################
# Define len() method.

    def __len__(self):
        # Return the no. trajectories containined in TrajFrame object.
        return self.trajectory_data.shape[0]

##############################################################################
# Define use_datetime.

    def use_datetime(self, start_date, fmt="%Y-%m-%d"):
        """
        Convert time attribute variable to Datetime format.

        Parameters
        ----------
        start_date : string
            Starting date to use when converting time attribute
            variable to Datetime.
        fmt : string
            Datetime format of specified start data. Default
            format is YYYY-MM-DD.

        Returns
        --------
        TrajFrame object
            Original TrajFrame object is returned with transformed
            time attribute variable Series containing datetimes.

        Examples
        --------
        Convert time in TrajFrame object to datetime with start
        date '2000-01-01 using default Datetime format.

        >>> trajectories.use_datetime(start_date='2000-01-01')
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(start_date, str) is False:
            raise TypeError("start_date must be specified as a string")
        if isinstance(self.trajectory_data['time'].dtype, pl.Datetime) is True:
            raise TypeError("time already exists with dtype = \'Datetime\'")

        # --------------------------------------------------
        # Convert time to Datetime format with start_date.
        # --------------------------------------------------
        # Redefining time variable in Datetime format.
        self.trajectory_data = self.trajectory_data.with_column(
            (pl.col("time")*np.timedelta64(1, 's')).cast(pl.Duration) + pl.lit(start_date).str.strptime(pl.Datetime, fmt)
        )

        # Return TrajFrame object with updated main DataFrame and original attributes and summary DataFrame.
        return TrajFrame(df=self.trajectory_data, attrs=self.attrs, df_summary=self.summary_data)

##############################################################################
# Define filter() method.

    def filter(self, expr, drop=False):
        """
        Filter trajectories using coniditional on a single column variable
        specified with a string expression.

        Filtering returns a reduced trajectory DataFrame where only the
        complete trajectories meeting the specified condition are retained.
        The expection is when users specify drop=True, in which case
        trajectories meeting the specified condition are dropped from the
        trajectory DataFrame.

        Filtering also reduces the summary DataFrame, retaining only the
        summary statistics available for trajectories meeting the specfied
        condition. The expection is when users specify drop=True.

        When the expression variable is specified as 'time' only the
        observations (obs) meeting the specified date conditions
        are retained for all trajectories.

        Parameters
        ----------
        variable : string
            Name of the variable in the TrajFrame object.
        expr : string
            String expression of the form "{variable} {operator} {value}",
            where {variable} represents the column variable contained in
            the trajectory DataFrame used to filter trajectories, {operator}
            represents one of the six standard comparison operators and
            {value} represents the value with which to compare the {variable}
            to.
        drop : boolean
            Inidcates if fitered trajectories should be retained in the
            new TrajFrame object (False) or instead dropped from the
            existing TrajFrame object (True).

        Returns
        -------
        TrajFrame object
            Complete TrajFrame, including all attribute variables,
            which meet (do not meet) the specified filter specification.

        Examples
        --------
        Filtering all trajectories where latitude is greater than 50 N.

        >>> trajectories.filter(expr='lat > 50', drop=False)

        Filtering trajectory observations between two dates using Datetime strings.

        >>> trajectories.filter('time > 2000-01-01').filter('time <= 2000-12-31')
        """
        # ------------------------
        # Split string expression.
        # ------------------------
        # Split string expression into three arguments seperated by spaces
        expr_split = expr.split(sep=' ')
        # Defining arguments from string expression:
        variable = expr_split[0]
        operator = expr_split[1]
        value = expr_split[2]

        # -------------------
        # Raising exceptions.
        # -------------------
        if len(expr_split) != 3:
            raise ValueError("string expression contains too many arguments. Use format \'var op val\' to compare column variable to value.")

        # Defining list of variables contained in trajectory DataFrame.
        col_variables = list(self.trajectory_data.columns)

        # Defining list of standard operators.
        operator_list = ['==', '!=', '<', '>', '<=', '>=']

        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if variable not in col_variables:
            raise ValueError("variable: \'" + variable + "\' not found in Dataset")

        if isinstance(drop, bool) is False:
            raise TypeError("drop must be specified as a boolean")

        if operator not in operator_list:
            raise ValueError("unknown comparison operator specified: \'" + operator + "\'. Use one of the standard Python comparison operators: ==, !=, <, >, <=, >=")

        # -----------------------------------------------
        # Determine value dtype from specified variable.
        # ----------------------------------------------
        value_dtype = self.trajectory_data[variable].dtype

        # --------------------------------------------------
        # Applying specified filter to trajectory DataFrame.
        # --------------------------------------------------
        # Apply filter according to specfied comparison operator:
        # Case 1. Equal
        if operator == '==':
            # Filter trajectory DataFrame according to drop argument:
            if drop is True:
                self.trajectory_data = self.trajectory_data.filter(~(pl.col(variable) == pl.lit(value).cast(value_dtype)))
            else:
                self.trajectory_data = self.trajectory_data.filter(pl.col(variable) == pl.lit(value).cast(value_dtype))

        # Case 2. Not Equal
        elif operator == '!=':
            # Filter trajectory DataFrame according to drop argument:
            if drop is True:
                self.trajectory_data = self.trajectory_data.filter(~(pl.col(variable) != pl.lit(value).cast(value_dtype)))
            else:
                self.trajectory_data = self.trajectory_data.filter(pl.col(variable) != pl.lit(value).cast(value_dtype))

        # Case 3. Less Than
        elif operator == '<':
            # Filter trajectory DataFrame according to drop argument:
            if drop is True:
                self.trajectory_data = self.trajectory_data.filter(~(pl.col(variable) < pl.lit(value).cast(value_dtype)))
            else:
                self.trajectory_data = self.trajectory_data.filter(pl.col(variable) < pl.lit(value).cast(value_dtype))

        # Case 4. Greater Than
        elif operator == '>':
            # Filter trajectory DataFrame according to drop argument:
            if drop is True:
                self.trajectory_data = self.trajectory_data.filter(~(pl.col(variable) > pl.lit(value).cast(value_dtype)))
            else:
                self.trajectory_data = self.trajectory_data.filter(pl.col(variable) > pl.lit(value).cast(value_dtype))

        # Case 5. Less Than or Equal
        elif operator == '<=':
            # Filter trajectory DataFrame according to drop argument:
            if drop is True:
                self.trajectory_data = self.trajectory_data.filter(~(pl.col(variable) <= pl.lit(value).cast(value_dtype)))
            else:
                self.trajectory_data = self.trajectory_data.filter(pl.col(variable) <= pl.lit(value).cast(value_dtype))

        # Case 6. Greater Than or Equal
        elif operator == '>=':
            # Filter trajectory DataFrame according to drop argument:
            if drop is True:
                self.trajectory_data = self.trajectory_data.filter(~(pl.col(variable) >= pl.lit(value).cast(value_dtype)))
            else:
                self.trajectory_data = self.trajectory_data.filter(pl.col(variable) >= pl.lit(value).cast(value_dtype))

        # -----------------------------------------------
        # Applying specified filter to summary DataFrame.
        # -----------------------------------------------
        # Filter summary DataFrame to include only trajectories contained in trajectory DataFrame:
        self.summary_data = self.summary_data.filter(pl.col('id').is_in(self.trajectory_data['id']))

        # Return TrajFrame object with updated trajectory and summary DataFrames & original attributes.
        return TrajFrame(df=self.trajectory_data, attrs=self.attrs, df_summary=self.summary_data)

##############################################################################
# Define filter_polygon() method.

    # def filter_polygon(self, polygon, method, drop=False):
    #     """
    #     Filter trajectories which intersect a specified polygon.

    #     Filtering returns the complete trajectories of particles
    #     which have been inside the boundary of a given polygon at
    #     any point in their lifetime.

    #     Parameters
    #     ----------
    #     polygon : list
    #         List of coordinates, specified as an ordered sequence of tuples
    #         (Lon, Lat), representing the boundary of the polygon.
    #     method : string
    #         Method to filter trajectories using polygon - 'pos' considers
    #         trajectories as a collection of points, 'traj' considers
    #         trajectories as connected lines.
    #     drop : boolean
    #         Determines if fitered trajectories should be returned as a
    #         new TrajArray object (False) or instead dropped from the
    #         existing TrajArray object (True).

    #     Returns
    #     -------
    #     TrajArray object
    #         Complete trajectories, including all attribute variables,
    #         which meet the filter specification.

    #     Examples
    #     --------
    #     Filtering all trajectories which intersect a simple polygon, square,
    #     using 'pos' method to consider each trajectory as a collection of
    #     points.

    #     >>> square = [[[-40, 30], [-40, 35], [-30, 35], [-30, 30], [-40, 30]]]
    #     >>> trajectories.filter_polygon(square, method='pos' drop=False)
    #     """
    #     # -------------------
    #     # Raising exceptions.
    #     # -------------------
    #     if isinstance(drop, bool) is False:
    #         raise TypeError("drop must be specified as a boolean")

    #     if isinstance(method, str) is False:
    #         raise TypeError("method must be specified as a string - options: \'pos\' or \'traj\'")

    #     if isinstance(polygon, list) is False:
    #         raise TypeError("polygon must be specified as a list of lists")

    #     if len(polygon) > 1:
    #         if len(polygon) != np.shape(self.data['time'].values)[1]:
    #             raise ValueError("one polygon must be specified per observation (obs) when filtering with multiple polygons")

    #     # ----------------------------------
    #     # Defining ds, the filtered DataSet.
    #     # ----------------------------------
    #     ds = filter_traj_polygon(self, polygon=polygon, method=method, drop=drop)

    #     # Returning the subsetted xarray DataSet as a TrajArray object -
    #     # this enables multiple filtering to take place.
    #     return TrajArray(ds)

##############################################################################
# Define compute_dx() method.

    # def compute_dx(self, unit='km'):
    #     """
    #     Compute particle zonal displacements from trajectories.

    #     Zonal (x) displacements between particle positions for
    #     all trajectories are returned as a new DataArray, dx,
    #     within the TrajArray object.

    #     Parameters
    #     ----------
    #     self : TrajArray object
    #         TrajArray object passed from TrajArray class method.
    #     unit : string
    #         Unit for particle displacement output - default 'km' -
    #         alternative option - 'm'.

    #     Returns
    #     -------
    #     TrajArray object
    #         Original TrajArray object is returned with appended attribute
    #         variable DataArray containing particle zonal displacements
    #         with dimensions (traj x obs).

    #         The first observation (obs) for all trajectories
    #         (traj) is NaN since the zonal distance from the origin
    #         of a particle at the origin is not defined.

    #     Examples
    #     --------
    #     Computing zonal displacements for all trajectories.

    #     >>> trajectories.compute_dx()
    #     """
    #     # ------------------
    #     # Raise exceptions.
    #     # ------------------
    #     # Define np.array with available options for particle
    #     # zonal displacement output units.
    #     unit_options = ['m', 'km']

    #     # Raising exception when unavailable unit is specified.
    #     if unit not in unit_options:
    #         raise ValueError("invalid unit - options: \'m\' or \'km\'")

    #     # -----------------------------------------
    #     # Computing dx with compute_displacement().
    #     # -----------------------------------------
    #     dx = compute_displacement(self, axis='x', unit=unit)

    #     # ---------------------
    #     # Adding dx to DataSet.
    #     # ---------------------
    #     # Append zonal displacement DataArray to original DataSet.
    #     self.data['dx'] = xr.DataArray(dx, dims=["traj", "obs"])
    #     # Adding attributes to zonal displacement DataArray.
    #     self.data.dx.attrs = {
    #                          'long_name': "zonal displacement",
    #                          'standard_name': "dx",
    #                          'units': unit,
    #                          'positive': "eastward"
    #                          }

    #     # Return TrajArray object with updated DataSet.
    #     return TrajArray(self.data)

##############################################################################
# Define compute_dy() method.

    # def compute_dy(self, unit='km'):
    #     """
    #     Compute particle meridional displacements from trajectories.

    #     Meridional (y) displacements between particle positions for
    #     all trajectories are returned as a new DataArray, dy,
    #     within the TrajArray object.

    #     Parameters
    #     ----------
    #     self : TrajArray object
    #         TrajArray object passed from TrajArray class method.
    #     unit : string
    #         Unit for particle displacement output - default 'km' -
    #         alternative option - 'm'.

    #     Returns
    #     -------
    #     TrajArray object
    #         Original TrajArray object is returned with appended attribute
    #         variable DataArray containing particle meridional displacements
    #         with dimensions (traj x obs).

    #         The first observation (obs) for all trajectories
    #         (traj) is NaN since the meridional distance from the origin
    #         of a particle at the origin is not defined.

    #     Examples
    #     --------
    #     Computing meridional displacements for all trajectories.

    #     >>> trajectories.compute_dy()
    #     """
    #     # ------------------
    #     # Raise exceptions.
    #     # ------------------
    #     # Define np.array with available options for particle
    #     # meridional displacement output units.
    #     unit_options = ['m', 'km']

    #     # Raising exception when unavailable unit is specified.
    #     if unit not in unit_options:
    #         raise ValueError("invalid unit - options: \'m\' or \'km\'")

    #     # -----------------------------------------
    #     # Computing dy with compute_displacement().
    #     # -----------------------------------------
    #     dy = compute_displacement(self, axis='y', unit=unit)

    #     # ---------------------
    #     # Adding dy to DataSet.
    #     # ---------------------
    #     # Append meridional displacement DataArray to original DataSet.
    #     self.data['dy'] = xr.DataArray(dy, dims=["traj", "obs"])
    #     # Adding attributes to meridional displacement DataArray.
    #     self.data.dy.attrs = {
    #                             'long_name': "meridional displacement",
    #                             'standard_name': "dy",
    #                             'units': unit,
    #                             'positive': "northward"
    #                             }

    #     # Return TrajArray object with updated DataSet.
    #     return TrajArray(self.data)

##############################################################################
# Define compute_dz() method.

    # def compute_dz(self, unit='m'):
    #     """
    #     Compute particle vertical displacements from trajectories.

    #     Vertical (z) displacements between particle positions for
    #     all trajectories are returned as a new DataArray, dz,
    #     within the trajectories object.

    #     Parameters
    #     ----------
    #     self : TrajArray object
    #         TrajArray object passed from TrajArray class method.
    #     unit : string
    #         Unit for particle displacement output - default 'm' -
    #         alternative option - 'km'.

    #     Returns
    #     -------
    #     TrajArray object.
    #     Original TrajArray object is returned with appended attribute
    #     variable DataArray containing particle vertical displacements
    #     with dimensions (traj x obs).

    #     The first observation (obs) for all trajectories
    #     (traj) is NaN since the vertical distance from the origin
    #     of a particle at the origin is not defined.

    #     Examples
    #     --------
    #     Computing vertical displacements for all trajectories.

    #     >>> trajectories.compute_dz()
    #     """
    #     # ------------------
    #     # Raise exceptions.
    #     # ------------------
    #     # Define np.array with available options for particle
    #     # vertical displacement output units.
    #     unit_options = ['m', 'km']

    #     # Raising exception when unavailable unit is specified.
    #     if unit not in unit_options:
    #         raise ValueError("invalid unit - options: \'m\' or \'km\'")

    #     # -----------------------------------------
    #     # Computing dz with compute_displacement().
    #     # -----------------------------------------
    #     dz = compute_displacement(self, axis='z', unit=unit)

    #     # ---------------------
    #     # Adding dz to DataSet.
    #     # ---------------------
    #     # Append vertical displacement DataArray to original DataSet.
    #     self.data['dz'] = xr.DataArray(dz, dims=["traj", "obs"])
    #     # Adding attributes to vertical displacement DataArray.
    #     self.data.dz.attrs = {
    #                          'long_name': "vertical displacement",
    #                          'standard_name': "dz",
    #                          'units': unit,
    #                          'positive': "upward"
    #                          }

    #     # Return TrajArray object with updated DataSet.
    #     return TrajArray(self.data)

##############################################################################
# Define compute_u() method.

    # def compute_u(self, unit='m/s'):
    #     """
    #     Compute particle zonal velocity from trajectories
    #     zonal displacements.

    #     Lagrangian zonal (x) velocity components for
    #     all trajectories are returned as a new DataArray, u,
    #     within the trajectories object.

    #     Parameters
    #     ----------
    #     self : TrajArray object
    #         TrajArray object passed from TrajArray class method.
    #     unit : string
    #         Unit for Lagrangian velocity output - default 'm/s' -
    #         alternative options - 'm/day', 'km/day'.

    #     Returns
    #     -------
    #     TrajArray object.
    #     Original TrajArray object is returned with appended attribute
    #     variable DataArray containing the zonal component of each particle's
    #     Lagrangian velocity with dimensions (traj x obs).

    #     The first observation (obs) for all trajectories
    #     (traj) is NaN since the zonal velocity from the origin
    #     of a particle at the origin is not defined.

    #     Examples
    #     --------
    #     Computing zonal velocities for all trajectories.

    #     >>> trajectories.compute_u()
    #     """
    #     # ------------------
    #     # Raise exceptions.
    #     # ------------------
    #     # Define np.array with available options for Lagrangian
    #     # velocity output units.
    #     unit_options = ['m/s', 'm/day', 'km/day']

    #     # Raising exception when unavailable unit is specified.
    #     if unit not in unit_options:
    #         raise ValueError("invalid unit - options: \'m/s\', \'m/day\', \'km/day\'")

    #     # -----------------------------------------
    #     # Computing u with compute_velocity().
    #     # -----------------------------------------
    #     u = compute_velocity(self, axis='x', unit=unit)

    #     # ---------------------
    #     # Adding u to DataSet.
    #     # ---------------------
    #     # Append zonal velocity DataArray to original DataSet.
    #     self.data['u'] = xr.DataArray(u, dims=["traj", "obs"])
    #     # Adding attributes to zonal velocity DataArray.
    #     self.data.u.attrs = {
    #                          'long_name': "zonal velocity",
    #                          'standard_name': "u",
    #                          'units': "m/s",
    #                          'positive': "eastward"
    #                          }

    #     # Return TrajArray object with updated DataSet.
    #     return TrajArray(self.data)

##############################################################################
# Define compute_v() method.

    # def compute_v(self, unit='m/s'):
    #     """
    #     Compute particle meridional velocity from trajectories
    #     meridional displacements.

    #     Lagrangian meridional (y) velocity components for
    #     all trajectories are returned as a new DataArray, v,
    #     within the TrajArray object.

    #     Parameters
    #     ----------
    #     self : TrajArray object
    #         TrajArray object passed from TrajArray class method.
    #     unit : string
    #         Unit for Lagrangian velocity output - default 'm/s' -
    #         alternative options - 'm/day', 'km/day'.

    #     Returns
    #     -------
    #     TrajArray object.
    #     Original TrajArray object is returned with appended attribute
    #     variable DataArray containing the meridional component of each
    #     particle's Lagrangian velocity with dimensions (traj x obs).

    #     The first observation (obs) for all trajectories
    #     (traj) is NaN since the meridional velocity from the origin
    #     of a particle at the origin is not defined.

    #     Examples
    #     --------
    #     Computing meridional velocities for all trajectories.

    #     >>> trajectories.compute_v()
    #     """
    #     # ------------------
    #     # Raise exceptions.
    #     # ------------------
    #     # Define np.array with available options for Lagrangian
    #     # velocity output units.
    #     unit_options = ['m/s', 'm/day', 'km/day']

    #     # Raising exception when unavailable unit is specified.
    #     if unit not in unit_options:
    #         raise ValueError("invalid unit - options: \'m/s\', \'m/day\', \'km/day\'")

    #     # -----------------------------------------
    #     # Computing v with compute_velocity().
    #     # -----------------------------------------
    #     v = compute_velocity(self, axis='y', unit=unit)

    #     # ---------------------
    #     # Adding v to DataSet.
    #     # ---------------------
    #     # Append meridional velocity DataArray to original DataSet.
    #     self.data['v'] = xr.DataArray(v, dims=["traj", "obs"])
    #     # Adding attributes to meridional velocity DataArray.
    #     self.data.v.attrs = {
    #                          'long_name': "meridional velocity",
    #                          'standard_name': "v",
    #                          'units': "m/s",
    #                          'positive': "northward"
    #                          }

    #     # Return TrajArray object with updated DataSet.
    #     return TrajArray(self.data)

##############################################################################
# Define compute_w() method.

    # def compute_w(self, unit='m/s'):
    #     """
    #     Compute particle vertical velocity from trajectories
    #     vertical displacements.

    #     Lagrangian vertical (z) velocity components for
    #     all trajectories are returned as a new DataArray, w,
    #     within the TrajArray object.

    #     Parameters
    #     ----------
    #     self : TrajArray object
    #         TrajArray object passed from TrajArray class method.
    #     unit : string
    #         Unit for Lagrangian velocity output - default 'm/s' -
    #         alternative options - 'm/day', 'km/day'.

    #     Returns
    #     -------
    #     TrajArray object.
    #     Original TrajArray object is returned with appended attribute
    #     variable DataArray containing the vertical component of each
    #     particle's Lagrangian velocity with dimensions (traj x obs).

    #     The first observation (obs) for all trajectories
    #     (traj) is NaN since the vertical velocity from the origin
    #     of a particle at the origin is not defined.

    #     Examples
    #     --------
    #     Computing vertical velocities for all trajectories.

    #     >>> trajectories.compute_w()
    #     """
    #     # ------------------
    #     # Raise exceptions.
    #     # ------------------
    #     # Define np.array with available options for Lagrangian
    #     # velocity output units.
    #     unit_options = ['m/s', 'm/day', 'km/day']

    #     # Raising exception when unavailable unit is specified.
    #     if unit not in unit_options:
    #         raise ValueError("invalid unit - options: \'m/s\', \'m/day\', \'km/day\'")

    #     # -----------------------------------------
    #     # Computing w with compute_velocity().
    #     # -----------------------------------------
    #     w = compute_velocity(self, axis='z', unit=unit)

    #     # ---------------------
    #     # Adding w to DataSet.
    #     # ---------------------
    #     # Append vertical velocity DataArray to original DataSet.
    #     self.data['w'] = xr.DataArray(w, dims=["traj", "obs"])
    #     # Adding attributes to vertical velocity DataArray.
    #     self.data.w.attrs = {
    #                          'long_name': "vertical velocity",
    #                          'standard_name': "w",
    #                          'units': unit,
    #                          'positive': "upwards"
    #                          }

    #     # Return TrajArray object with updated DataSet.
    #     return TrajArray(self.data)

##############################################################################
# Define compute_distance() method.

    # def compute_distance(self, cumsum_dist=False, unit='km'):
    #     """
    #     Compute distance travelled by particles along their
    #     of trajectories.

    #     Either the distance travelled between particle positions
    #     or the cumulative distance travelled is computed
    #     and returned for all trajectories as a new DataArray.

    #     Parameters
    #     ----------
    #     self : TrajArray object
    #         TrajArray object passed from TrajArray class method.
    #     cumsum_dist : logical
    #         Compute the cumulative distance travelled by each particle -
    #         default is False.
    #     unit : string
    #         Unit for distance travelled output - default is 'km' -
    #         alternative option 'm'.

    #     Returns
    #     -------
    #     TrajArray object.
    #     Original TrajArray object is returned with appended attribute
    #     variable DataArray containing the distance travelled by each
    #     particle along it's trajectory with dimensions (traj x obs).

    #     The first observation (obs) for all trajectories
    #     (traj) is NaN since the (cumulative) distance
    #     from the origin of a particle at the origin is
    #     not defined.

    #     Examples
    #     --------
    #     Computing distance travelled by particles for all trajectories,
    #     specifying cumulative distance as False and unit as default 'km'.

    #     >>> trajectories.compute_distance()
    #     """
    #     # ------------------
    #     # Raise exceptions.
    #     # ------------------
    #     # Define np.array with available options for
    #     # distance output units.
    #     unit_options = ['m', 'km']

    #     # Raising exception when unavailable unit is specified.
    #     if unit not in unit_options:
    #         raise ValueError("invalid unit - options: \'m\', \'km\'")

    #     # -------------------------------------------
    #     # Computing distance with compute_distance().
    #     # -------------------------------------------
    #     dist = compute_distance(self, cumsum_dist=cumsum_dist, unit=unit)

    #     # -----------------------
    #     # Adding dist to DataSet.
    #     # -----------------------
    #     if cumsum_dist is True:
    #         # Append distance DataArray to original DataSet.
    #         self.data['cumdist'] = xr.DataArray(dist, dims=["traj", "obs"])

    #         # Adding attributes to cumdist DataArray.
    #         self.data.cumdist.attrs = {
    #                             'long_name': "cumulative distance",
    #                             'standard_name': "cumdist",
    #                             'units': unit,
    #                             }
    #     else:
    #         # Append distance DataArray to original DataSet.
    #         self.data['dist'] = xr.DataArray(dist, dims=["traj", "obs"])

    #         # Adding attributes to dist DataArray.
    #         self.data.dist.attrs = {
    #                             'long_name': "distance",
    #                             'standard_name': "dist",
    #                             'units': unit,
    #                             }

    #     # Return TrajArray object with updated DataSet.
    #     return TrajArray(self.data)

##############################################################################
# Define compute_residence_time() method.

    # def compute_residence_time(self, polygon):
    #     """
    #     Compute the maximum duration (days) that particles spend
    #     within a specified polygon (residence time).

    #     Residence time is defined as the longest continuous
    #     period of time that a particle is contained within the
    #     limits of the specified polygon/s.

    #     Parameters
    #     ----------
    #     self : TrajArray object
    #         TrajArray object passed from TrajArray class method.
    #     polygon : list
    #         List of coordinates, specified as an ordered sequence of tuples
    #         (Lon, Lat), representing the boundary of the polygon.

    #     Returns
    #     -------
    #     TrajArray object.
    #     Original TrajArray object is returned with appended attribute
    #     variable DataArray containing the residence time (days) of each
    #     trajectory with dimensions (traj).

    #     Examples
    #     --------
    #     Computing the residence time of trajectories which intersect a simple
    #     polygon, square. Below we filter trajectories using 'pos' method to
    #     consider each trajectory as a collection of points.

    #     >>> square = [[[-40, 30], [-40, 35], [-30, 35], [-30, 30], [-40, 30]]]
    #     >>> trajectories.filter_polygon(square, method='pos', drop=False).compute_residence_time(polygon=square)
    #     """
    #     # ------------------
    #     # Raise exceptions.
    #     # ------------------
    #     if isinstance(polygon, list) is False:
    #         raise TypeError("polygon must be specified as a list of lists")

    #     if len(polygon) > 1:
    #         if len(polygon) != np.shape(self.data['time'].values)[1]:
    #             raise ValueError("one polygon must be specified per observation (obs) when filtering with multiple polygons")

    #     # --------------------------------------------------
    #     # Computing residence times with .compute_res_time().
    #     # --------------------------------------------------
    #     max_residence_time = compute_res_time(self, polygon)

    #     # Append DataArray to original DataSet.
    #     self.data['residence_time'] = xr.DataArray(max_residence_time, dims=["traj"])

    #     # -----------------------------------
    #     # Adding variable attributes DataSet.
    #     # -----------------------------------
    #     self.data.residence_time.attrs = {
    #                         'long_name': "maximum residence time",
    #                         'standard_name': "residence_time",
    #                         'units': "days",
    #                         }

    #     # Return TrajArray object with updated DataSet.
    #     return TrajArray(self.data)

##############################################################################
# Define compute_transit_time() method.

    # def compute_transit_time(self, polygon):
    #     """
    #     Compute the time taken (days) for filtered particles
    #     to first intersect a specified polygon (transit time).

    #     Transit time is defined as the time taken for each
    #     particle to enter the limits of the specified polygon/s.

    #     Parameters
    #     ----------
    #     self : TrajArray object
    #         TrajArray object passed from TrajArray class method.
    #     polygon : list
    #         List of coordinates, specified as an ordered sequence of tuples
    #         (Lon, Lat), representing the boundary of the polygon.

    #     Returns
    #     -------
    #     TrajArray object.
    #     Original TrajArray object is returned with appended attribute
    #     variable DataArray containing the transit time (days) of each
    #     trajectory with dimensions (traj).

    #     Examples
    #     --------
    #     Computing the transit time of trajectories which intersect a simple
    #     polygon, square. Below we filter trajectories using 'pos' method to
    #     consider each trajectory as a collection of points.

    #     >>> square = [[[-40, 30], [-40, 35], [-30, 35], [-30, 30], [-40, 30]]]
    #     >>> trajectories.filter_polygon(square, method='pos', drop=False).compute_transit_time(polygon=square)
    #     """
    #     # ------------------
    #     # Raise exceptions.
    #     # ------------------
    #     if isinstance(polygon, list) is False:
    #         raise TypeError("polygon must be specified as a list of lists")

    #     if len(polygon) > 1:
    #         if len(polygon) != np.shape(self.data['time'].values)[1]:
    #             raise ValueError("one polygon must be specified per observation (obs) when filtering with multiple polygons")

    #     # -----------------------------------------------------
    #     # Computing residence times with .compute_trans_time().
    #     # -----------------------------------------------------
    #     transit_time = compute_trans_time(self, polygon)

    #     # Append DataArray to original DataSet.
    #     self.data['transit_time'] = xr.DataArray(transit_time, dims=["traj"])

    #     # -----------------------------------
    #     # Adding variable attributes DataSet.
    #     # -----------------------------------
    #     self.data.transit_time.attrs = {
    #                         'long_name': "transit time",
    #                         'standard_name': "transit_time",
    #                         'units': "days",
    #                         }

    #     # Return TrajArray object with updated DataSet.
    #     return TrajArray(self.data)

##############################################################################
# Define compute_probability() method.

    # def compute_probability(self, bin_res, method, gf_sigma=None, group_by=None):
    #     """
    #     Compute 2-dimensional binned Lagrangian probability
    #     distributions using particle positions or particle
    #     pathways.

    #     Particle positions are binned into a 2-dimensional
    #     (x-y) histogram and normalised by the total number
    #     of particle positions ('pos') or the total number
    #     of particles ('traj').

    #     A Gaussian filter with a specified radius may also
    #     be included to smooth the distribution.

    #     Parameters
    #     ----------
    #     self : TrajArray object
    #         TrajArray object passed from TrajArray class method.
    #     bin_res : numeric
    #         The resolution (degrees) of the grid on to which particle
    #         positions will be binned.
    #     method : string
    #         The type of probability to be computed. 'pos' - particle
    #         positions are binned and then normalised by the total number
    #         of particle positions. 'traj' - for each particle positions
    #         are counted once per bin and then normalised by the total
    #         number of particles. To include a Gaussian filter modify the
    #         methods above to 'pos-gauss' or 'traj-gauss'.
    #     gf_sigma : numeric
    #         The standard deviation of the Gaussian filter (degrees) with
    #         which to smooth the Lagrangian probability distribution.
    #     group_by : string
    #         Grouping variable to compute Lagrangian probability
    #         distributions - one distribution is computed for every
    #         unique member of variable. See example below.

    #     Returns
    #     -------
    #     TrajArray object.
    #     Original TrajArray object is returned with appended attribute
    #     variable DataArrays containing the binned 2-dimensional
    #     Lagrangian probability distribution and the coordinates of the
    #     centre points of the grid with dimensions (x - y). Where group_by
    #     is used the Lagrangian probability distributions will have
    #     dimensions (samples - x - y).

    #     Examples
    #     --------
    #     Computing the Lagrangian probability distribution using all
    #     particle positions for particles released at seed_level 1.

    #     >>> trajectories.filter_equal('seed_level', 1).compute_probability(bin_res=1, method='pos')

    #     Computing the Lagrangian probability density distribution using
    #     all particle positions with a Gaussian filter for particles released
    #     at seed_levels 1 to 5.

    #     >>> trajectories.filter_between('seed_level', 1, 5).compute_probability(bin_res=1, method='pos-gauss', gf_sigma=1, group_by='seed_level')
    #     """
    #     # ------------------
    #     # Raise exceptions.
    #     # ------------------
    #     if group_by is not None:
    #         # Defining list of variables contained in data.
    #         variables = list(self.data.variables)

    #         if group_by not in variables:
    #             raise ValueError("variable: \'" + group_by + "\' not found in Dataset")

    #     if isinstance(method, str) is False:
    #         raise TypeError("method must be specified as a string - options: \'pos\', \'traj\', \'pos-gauss\', \'traj-gauss\'")

    #     if isinstance(bin_res, (int, float)) is False:
    #         raise TypeError("bin_res must be specified as integer or float")

    #     # ----------------------------------------------
    #     # Computing Lagrangian probability distribution.
    #     # ----------------------------------------------
    #     nav_lon, nav_lat, probability = compute_probability_distribution(self, bin_res=bin_res, method=method, gf_sigma=gf_sigma, group_by=group_by)

    #     # Append DataArrays to original DataSet.
    #     self.data['nav_lat'] = xr.DataArray(nav_lat, dims=["y", "x"])
    #     self.data['nav_lon'] = xr.DataArray(nav_lon, dims=["y", "x"])

    #     if group_by is not None:
    #         self.data['probability'] = xr.DataArray(probability, dims=["sample", "y", "x"])
    #     else:
    #         self.data['probability'] = xr.DataArray(probability, dims=["y", "x"])

    #     # -----------------------------------
    #     # Adding variable attributes DataSet.
    #     # -----------------------------------
    #     self.data.nav_lat.attrs = {
    #                         'long_name': "Latitude",
    #                         'standard_name': "latitude",
    #                         'units': "degrees_north",
    #                         }
    #     self.data.nav_lon.attrs = {
    #                         'long_name': "Longitude",
    #                         'standard_name': "longitude",
    #                         'units': "degrees_east",
    #                         }
    #     self.data.probability.attrs = {
    #                         'long_name': "Lagrangian probability",
    #                         'standard_name': "probability",
    #                         }

    #     # Return TrajArray object with updated DataSet.
    #     return TrajArray(self.data)

##############################################################################
# Define get_start_time() method.

    def get_start_time(self):
        """
        Returns time when water parcels are seeded (start of trajectory).

        The start time is given in the format of the {time} core variable
        and stored for each trajectory in the summary DataFrame.

        Parameters
        ----------
        self : TrajFrame object
            TrajFrame object passed from TrajFrame class method.

        Returns
        -------
        TrajFrame.
            Original TrajFrame is returned with appended column
            variable {start_time} in the summary DataFrame
            containing the times each trajectory was seeded.

        Examples
        --------
        Get seeding times for all trajectories.

        >>> trajectories.get_start_time()
        """
        # -------------------------------------------
        # Calculate seeding time for each trajectory.
        # -------------------------------------------
        start_time_series = (self.trajectory_data
                             .groupby(pl.col("id"))
                             .agg(pl.col('time').min().alias('start_time'))
                             )['start_time']

        # ----------------------------------
        # Adding {start_time} to DataFrame.
        # ----------------------------------
        # Appending start time values as new column variable in summary DataFrame.
        self.summary_data = self.summary_data.with_column(start_time_series)

        # Return TrajFrame object with updated summary DataFrame & original trajectory DataFrame and attributes.
        return TrajFrame(df=self.trajectory_data, attrs=self.attrs, df_summary=self.summary_data)

##############################################################################
# Define get_start_loc() method.

    def get_start_loc(self):
        """
        Returns locations where water parcels are seeded (start of trajectory).

        The start locations of each trajectory are added in the form of three
        new column variables in the summary DataFrame: {lon_start}, {lat_start}
        and {depth_start}.

        Parameters
        ----------
        self : TrajFrame object
            TrajFrame object passed from TrajFrame class method.

        Returns
        -------
       TrajFrame.
            Original TrajFrame is returned with appended column
            variables {lon_start}, {lat_start}, {depth_start} in
            the summary DataFrame.

        Examples
        --------
        Get seeding locations for all trajectories.

        >>> trajectories.get_start_loc()
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Defining list of column variables contained in trajectory DataFrame.
        col_variables = list(self.trajectory_data.columns)

        if ('lon' not in col_variables) | ('lat' not in col_variables) | ('depth' not in col_variables):
            raise ValueError("required variable missing from trajectory DataFrame: \'lon\', \'lat\', \'depth\'")

        # --------------------------------------------------
        # Return start longitude, latitude and depth values.
        # --------------------------------------------------
        lon_start_series = (self.trajectory_data.groupby('id')
                             .agg([pl.col('lon').sort_by('t').first().alias('lon_start')])
                             )['lon_start']
        lat_start_series = (self.trajectory_data.groupby('id')
                             .agg([pl.col('lat').sort_by('t').first().alias('lat_start')])
                             )['depth_start']
        depth_start_series = (self.trajectory_data.groupby('id')
                             .agg([pl.col('depth').sort_by('t').first().alias('depth_start')])
                             )['depth_start']

        # ----------------------------------------
        # Adding lat/lon/depth_start to DataFrame.
        # ----------------------------------------
        # Appending all new column variables in summary DataFrame.
        self.summary_data = self.summary_data.with_columns(lon_start_series, lat_start_series, depth_start_series)

        # Return TrajFrame object with updated summary DataFrame & original trajectory DataFrame and attributes.
        return TrajFrame(df=self.trajectory_data, attrs=self.attrs, df_summary=self.summary_data)

##############################################################################
# Define get_end_time() method.

    def get_end_time(self):
        """
        Returns time when water parcels are terminated (end of trajectory).

        The end time is given in the format of the {time} core variable
        and stored for each trajectory in the summary DataFrame.

        Parameters
        ----------
        self : TrajFrame object
            TrajFrame object passed from TrajFrame class method.

        Returns
        -------
        TrajFrame.
            Original TrajFrame is returned with appended column
            variable {end_time} in the summary DataFrame
            containing the times each trajectory was terminate.

        Examples
        --------
        Get end times for all trajectories.

        >>> trajectories.get_end_time()
        """
        # ----------------------------------------------
        # Calculate termination time of each trajectory.
        # ----------------------------------------------
        end_time_series = (self.trajectory_data
                             .groupby(pl.col("id"))
                             .agg(pl.col('time').max().alias('end_time'))
                             )['end_time']

        # -------------------------------
        # Adding {end_time} to DataFrame.
        # -------------------------------
        # Appending end time values as new column variable in summary DataFrame.
        self.summary_data = self.summary_data.with_column(end_time_series)

        # Return TrajFrame object with updated summary DataFrame & original trajectory DataFrame and attributes.
        return TrajFrame(df=self.trajectory_data, attrs=self.attrs, df_summary=self.summary_data)

##############################################################################
# Define get_end_loc() method.

    def get_end_loc(self):
        """
        Returns locations where water parcels are terminated (end of trajectory).

        The end locations of each trajectory are added in the form of three
        new column variables in the summary DataFrame: {lon_end}, {lat_end}
        and {depth_end}.

        Parameters
        ----------
        self : TrajFrame object
            TrajFrame object passed from TrajFrame class method.

        Returns
        -------
       TrajFrame.
            Original TrajFrame is returned with appended column
            variables {lon_end}, {lat_end}, {depth_end} in
            the summary DataFrame.

        Examples
        --------
        Get end locations for all trajectories.

        >>> trajectories.get_end_loc()
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Defining list of column variables contained in trajectory DataFrame.
        col_variables = list(self.trajectory_data.columns)

        if ('lon' not in col_variables) | ('lat' not in col_variables) | ('depth' not in col_variables):
            raise ValueError("required variable missing from trajectory DataFrame: \'lon\', \'lat\', \'depth\'")

        # -------------------------------------------------
        # Return end longitude, latitude and depth values.
        # -------------------------------------------------
        lon_end_series = (self.trajectory_data.groupby('id')
                             .agg([pl.col('lon').sort_by('t').last().alias("lon_end")])
                             )['lon_end']
        lat_end_series = (self.trajectory_data.groupby('id')
                             .agg([pl.col('lat').sort_by('t').last().alias("lat_end")])
                             )['lat_end']
        depth_end_series = (self.trajectory_data.groupby('id')
                             .agg([pl.col('depth').sort_by('t').last().alias("depth_end")])
                             )['depth_end']

        # ----------------------------------------
        # Adding {lat/lon/depth_end} to DataFrame.
        # ----------------------------------------
        # Appending all new column variables in summary DataFrame.
        self.summary_data = self.summary_data.with_columns(lon_end_series, lat_end_series, depth_end_series)

        # Return TrajFrame object with updated summary DataFrame & original trajectory DataFrame and attributes.
        return TrajFrame(df=self.trajectory_data, attrs=self.attrs, df_summary=self.summary_data)

##############################################################################
# Define get_duration() method.

    def get_duration(self):
        """
        Returns time taken for particles to be terminated following seeding
        (duration of trajectory).

        The duration of each trajectory is stored in the polars Duration
        dtype format.

        Parameters
        ----------
        self : TrajFrame object
            TrajFrame object passed from TrajFrame class method.

        Returns
        -------
        TrajFrame.
            Original TrajFrame is returned with appended column
            variable {total_time} in the summary DataFrame.

        Examples
        --------
        Get duration of all trajectories.

        >>> trajectories.get_duration()
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Raise exception if {time} column variable is not stored as polar Datetime dtype.
        if isinstance(self.trajectory_data['time'].dtype, pl.Datetime) is False:
            raise TypeError("times must be stored as polars Datetime dtype")

        # -------------------------------------
        # Calculate duration of each trajectory.
        # -------------------------------------
        total_time_series =(self.trajectory_data.groupby(pl.col('id'))
                            .agg((pl.col('time').max() - pl.col('time').min()).cast(pl.Duration).alias("total_time"))
                            )

        # ---------------------------------
        # Adding {total_time} to DataFrame.
        # ---------------------------------
        # Appending trajectory duration values as new column variable in summary DataFrame.
        self.summary_data = self.summary_data.with_column(total_time_series)

        # Return TrajFrame object with updated summary DataFrame & original trajectory DataFrame and attributes.
        return TrajFrame(df=self.trajectory_data, attrs=self.attrs, df_summary=self.summary_data)

##############################################################################
# Define get_values() function.

    def get_values(self, variable, date, fmt="%Y-%m-%d"):
        """
        Returns the values of a specified variable at a specified
        date for each trajectory.

        The values of the specified variable are returned for all
        trajectories in the summary DataFrame.

        Parameters
        ----------
        self : TrajArray object
            TrajArray object passed from TrajArray class method.
        variable : string
            Name of the variable in the TrajArray object.
        date : string
            date on which to get values of specified variable.
        fmt : string
            Datetime format of specified date. Default
            format is YYYY-MM-DD.

        Returns
        -------
        TrajFrame.
            Original TrajFrame is returned with appended column
            variable {variable}_i in the summary DataFrame.

        Examples
        --------
        Get the value of temperature for each trajectory on date 2000-01-31.
        Note that we must convert time to polars Datetime format before using
        .get_values().

        >>>  trajectories.use_datetime(start_time='2000-01-01').get_values(variable='temp', date='2000-01-31')
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Defining list of column variables contained in trajectory DataFrame.
        variables = list(self.trajectory_data.columns)

        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if variable not in variables:
            raise ValueError("variable: \'" + variable + "\' not found in trajectory DataFrame")

        if isinstance(date, str) is False:
            raise TypeError("date must be specified as a string in format givent to fmt")

        # ----------------------------------------------------------
        # Extract values of specfied variable at specified date.
        # ----------------------------------------------------------
        # Filtering trajectory DataFrame for specified date and then selecting specified variable.
        values_series = (self.trajectory_data
                         .filter(pl.col('time') == pl.lit(date).str.strptime(pl.Datetime, fmt))
                         .select(pl.col(variable).alias(variable + "_i"))
                         )

        # ---------------------------------
        # Adding {variable}_i to DataFrame.
        # ---------------------------------
        # Appending variable values at specified date as new column variable in summary DataFrame.
        # Here, we use concatenation (hstack) since not all trajectories may have values of
        # {variable} at the specfied date, so this method fills with new column with null values
        # where values are missing to allow join to summary DataFrame.
        self.summary_data = pl.concat([self.summary_data, values_series], how='horizontal')

        # Return TrajFrame object with updated summary DataFrame & original trajectory DataFrame and attributes.
        return TrajFrame(df=self.trajectory_data, attrs=self.attrs, df_summary=self.summary_data)

##############################################################################
# Define get_max() function.

    def get_max(self, variable):
        """
        Returns maximum value of a specified variable for each trajectory.

        The maximum value of the variable is returned for all trajectories
        as a new column variable in the summar DataFrame.

        Parameters
        ----------
        self : TrajFrame object
            TrajFrame object passed from TrajFrame class method.
        variable : string
            Name of the variable in the TrajFrame object.

        Returns
        -------
        TrajFrame.
            Original TrajFrame is returned with appended column
            variable {variable}_max in the summary DataFrame
            containing the max values along each trajectory.

        Examples
        --------
        Get the maximum temperature along each trajectory.

        >>>  trajectories.get_max(variable='temp').
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Defining list of variables contained in data.
        col_variables = self.trajectory_data.columns

        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if variable not in col_variables:
            raise ValueError("variable: \'" + variable + "\' not found in main DataFrame")

        # -----------------------------------------------
        # Calculate maximum values of variable as Series.
        # -----------------------------------------------
        # Defining std. name of max_values using specified variable.
        max_variable = variable + "_max"

        # Calculate maximum value of {variable} along each trajectory.
        max_series = self.trajectory_data.groupby(pl.col("id")).agg(pl.col(variable).max().alias(max_variable))[max_variable]

        # ---------------------------------------
        # Adding max_values to summary DataFrame.
        # ---------------------------------------
        # Appending maximum values as new variable in summary DataFrame.
        self.summary_data = self.summary_data.with_column(max_series)

        # Return TrajFrame object with updated summary DataFrame & original trajectory DataFrame and attributes.
        return TrajFrame(df=self.trajectory_data, attrs=self.attrs, df_summary=self.summary_data)

##############################################################################
# Define get_min() function.

    def get_min(self, variable):
        """
        Returns minimum value of a specified variable for each trajectory.

        The minimum value of the variable is returned for all trajectories
        as a new column variable in the summary DataFrame.

        Parameters
        ----------
        self : TrajFrame object
            TrajFrame object passed from TrajFrame class method.
        variable : string
            Name of the variable in the TrajFrame object.

        Returns
        -------
        TrajFrame.
            Original TrajFrame is returned with appended column
            variable {variable}_min in the summary DataFrame
            containing the min values along each trajectory.

        Examples
        --------
        Get the minimum temperature along each trajectory.

        >>>  trajectories.get_min(variable='temp').
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Defining list of variables contained in data.
        col_variables = self.trajectory_data.columns

        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if variable not in col_variables:
            raise ValueError("variable: \'" + variable + "\' not found in main DataFrame")

        # -----------------------------------------------
        # Calculate minimum values of variable as Series.
        # -----------------------------------------------
        # Defining std. name of min_values using specified variable.
        min_variable = variable + "_min"

        # Calculate minimum value of {variable} along each trajectory.
        min_series = self.trajectory_data.groupby(pl.col("id")).agg(pl.col(variable).min().alias(min_variable))[min_variable]

        # ---------------------------------------
        # Adding min_values to summary DataFrame.
        # ---------------------------------------
        # Appending minimum values as new variable in summary DataFrame.
        self.summary_data = self.summary_data.with_column(min_series)

        # Return TrajFrame object with updated summary DataFrame & original attributes and trajectory DataFrame.
        return TrajFrame(df=self.trajectory_data, attrs=self.attrs, df_summary=self.summary_data)

##############################################################################
# Define add_variable() method.

    def add_variable(self, data, attrs, summary_var=False):
        """
        Adds a new variable to the existing TrajFrame object.

        The variable data must be provided as 1-dimensional ndarray
        with the variable attributes provided as a string.

        Parameters
        ----------
        self : TrajFrame object
            TrajFrame object passed from TrajFrame class method.
        data : ndarray
            values of new variable to be added to the TrajFrame object
            DataFrame.
        attributes : dict
            metadata attributes describing the new variable.
        summary_var : bool
            indicates if new column variable should be added to the
            summary DataFrame (True) or trajectory DataFrame (False).
            The Default value is False.

        Returns
        -------
        TrajFrame object
            Original TrajFrame object is returned with the new column variable
            appended to the trajectory DataFrame or summary DataFrame.

        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(data, np.ndarray) is False:
            raise TypeError("data must be specified as an ndarray")
        if isinstance(attrs, dict) is False:
            raise TypeError("column variable attributes must be specified as a dictionary")
        if isinstance(summary_var, bool) is False:
            raise TypeError("summary_var must be specified as a boolean value")

        # -----------------------------------------------------
        # Returning updated DataFrame with new column variable.
        # -----------------------------------------------------
        # Defining new column variable name.
        variable = list(attrs.keys())[0]

        # Updating DataFrame conditional on summary_var boolean value:
        if summary_var is True:
            # Append new column variable to summary DataFrame.
            self.summary_data = self.summary_data.with_column(pl.Series(name=variable, values=data))
        elif summary_var is False:
            # Append new column variable to trajectory DataFrame.
            self.trajectory_data = self.trajectory_data.with_column(pl.Series(name=variable, values=data))
            # Append new attributes to attrs dictionary.
            self.attrs.update(attrs)

        # Return TrajFrame object with updated summary DataFrame or trajectory DataFrame with updated attributes.
        return TrajFrame(df=self.trajectory_data, attrs=self.attrs, df_summary=self.summary_data)
