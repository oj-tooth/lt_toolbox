##############################################################################
# trajstore.py
#
# Description:
# Defines the TrajStore Class from a .csv or .parquet file containing atmosphere
# ocean parcel trajectories (and accompanying tracers) stored in a tabular format.
#
# Last Edited:
# 2023/11/05
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

# Importing utility functions
from utils.filter_frame_utils import filter_traj_polygon, filter_traj

##############################################################################
# Define TrajStore Class.


class TrajStore:

    def __init__(self, source, read_mode=None, summary_source=None, rename_cols=None, read_options=None, **kwargs):
        """
        Create a TrajStore from a single filepath, list of filepaths, Polars DataFrame
        or Polars LazyFrame.

        Parameters
        ----------
        source: str | list
            Path or list of paths to .csv or .parquet file(s) containing Lagrangian
            trajectories in tabular format. Alternatively, a Polars DataFrame or
            LazyFrame may be specified.
        read_mode : str
            Select either 'eager' or 'lazy' mode to read source files.
            Default is to use Polars lazy API, only executing full queries
            when specified.
        summary_source : DataFrame | LazyFrame
            Summary data stored for each trajectory in TrajFrame, df.
            Default value is None since TrajFrame is assumed to
            contain multiple rows for each trajectory. An exception is made
            when the TrajFrame contains only a single row per trajectory.
            In this case, TrajFrame and SummaryFrame are equivalent.
        rename_cols : dict
            Rename columns using key value pairs that map from old name to new
            name.
        read_options : dict
            Additional keyword arguments to be passed to Polars read_csv(),
            read_parquet(), scan_csv() or scan_parquet() function when
            constructing TrajStore object.

        Returns
        --------
        TrajStore object
            Complete trajectories, including all column variables contained
            in TrajectoryFrame. Summary data stored for each
            trajectory in SummaryFrame.

        Examples
        --------
        Creating TrajStore object, trajectories, with example_trajectories.csv file in eager mode.

        >>> filename = 'example_trajectories.csv'
        >>> trajectories = TrajStore(source=filename, SummaryFrame=None, mode='eager')

        Creating TrajStore object, traj, with multiple parquet files in lazy mode.

        >>> filenames = [ 'example_trajectories1.parquet', 'example_trajectories2.parquet']
        >>> trajectories = TrajStore(source=filenames, SummaryFrame=None, mode='lazy')

        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Raising exceptions where a TrajectoryFrame source is specified:
        if source is not None:
            # Determine if source is Polars DataFrame:
            is_DataFrame = isinstance(source, pl.DataFrame)
            # Determine if source is Polars LazyFrame:
            is_LazyFrame = isinstance(source, pl.LazyFrame)

            # Raise error if source is not a string, list or Polars DataFrame / LazyFrame:
            if (isinstance(source, str) | isinstance(source, list) | is_DataFrame | is_LazyFrame) is False:
                raise TypeError("source file path(s) must be specified as string or list of strings")

            # Raise error if source list does not contain strings:
            if isinstance(source, list):
                if any(isinstance(file, str) for file in source) is False:
                    raise TypeError("source file paths must be specified as list of strings")

            # Raise error if source filepath does not contain .csv or .parquet extension:
            if isinstance(source, str):
                if (source.endswith('.csv')) | (source.endswith('.parquet')) is False:
                    raise TypeError("source file must be specified as either .csv or .parquet")
            if isinstance(source, list):
                if any(((file.endswith('.csv')) | (file.endswith('.parquet'))) for file in source) is False:
                    raise TypeError("source files must be specified as either .csv or .parquet")

        # Raise error if read_mode is not a string:
        if read_mode is not None:
            if isinstance(read_mode, str) is False:
                raise TypeError("mode must be specified as string")
            if read_mode not in ['eager', 'lazy']:
                raise TypeError("mode must be specified as either \'eager\' or \'lazy\'")

        # Raise error if summary_source is not a Polars DataFrame:
        if summary_source is not None:
            if isinstance(summary_source, pl.DataFrame) is False:
                raise TypeError("summary_source must be specified a Polars DataFrame")

        # Raise error if attrs is not a dictionary:
        if rename_cols is not None:
            if isinstance(rename_cols, dict) is False:
                raise TypeError("rename columns mapping specified as a dictionary")

        # ------------------------------------------------------
        # Constructing Lagrangian trajectory frame from source:
        # ------------------------------------------------------
        if source is not None:
            # Case 1. From specified Polars DataFrame or LazyFrame:
            if (is_DataFrame | is_LazyFrame) is True:
                df = source

            # Case 2. Reading input file(s) as Polars DataFrame or LazyFrame.
            else:
                # Eager reading of source file(s).
                if read_mode == 'eager':
                    # Sub Case 1. Single file:
                    if isinstance(source, str):
                        if source.endswith('.csv'):
                            # Reading DataFrame from source .csv file:
                            df = pl.read_csv(source, **(read_options or {}))
                        elif source.endswith('.parquet'):
                            # Reading DataFrame from source .parquet file:
                            df = pl.read_parquet(source, **(read_options or {}))
                    # Sub Case 2. Multiple files:
                    if isinstance(source, list):
                        if source[0].endswith('.csv'):
                            # Reading DataFrame from source .csv file:
                            df = pl.concat([pl.read_csv(file, **(read_options or {})) for file in source])
                        elif source[0].endswith('.parquet'):
                            # Reading DataFrame from source .parquet file:
                            df = pl.concat([pl.read_parquet(file, **(read_options or {})) for file in source])

                # Lazy reading of source file(s).
                if read_mode == 'lazy':
                    # Sub Case 1. Single file:
                    if isinstance(source, str):
                        if source.endswith('.csv'):
                            # Scan LazyFrame from source .csv file:
                            df = pl.scan_csv(source, **(read_options or {}))
                        elif source.endswith('.parquet'):
                            # Scan LazyFrame from source .parquet file:
                            df = pl.scan_parquet(source, **(read_options or {}))
                    # Sub Case 2. Multiple files:
                    if isinstance(source, list):
                        if source[0].endswith('.csv'):
                            # Scan LazyFrame from source .csv file:
                            df = pl.concat([pl.scan_csv(file, **(read_options or {})) for file in source])
                        elif source[0].endswith('.parquet'):
                            # Scan LazyFrame from source .parquet file:
                            df = pl.concat([pl.scan_parquet(file, **(read_options or {})) for file in source])

        # Renaming columns of trajectory frame:
        if rename_cols is not None:
            # Rename columns using key value pairs that map from
            # old name to new name:
            df = df.rename(rename_cols)

        # ------------------------------------------------------
        # Storing input trajectory frame as TrajStore attribute.
        # ------------------------------------------------------
        if source is not None:
            # Defining TrajFrame as input DataFrame or LazyFrame
            # containing one or more rows per trajectory.
            self.TrajFrame = df
        else:
            self.TrajFrame = None

        # ---------------------------------------
        # Defining modes as TrajStore attributes:
        # ---------------------------------------
        # Defining mode of TrajFrame:
        if self.TrajFrame is None:
            self.traj_mode = None
        elif isinstance(self.TrajFrame, pl.DataFrame):
            self.traj_mode = 'eager'
        elif isinstance(self.TrajFrame, pl.LazyFrame):
            self.traj_mode = 'lazy'

        # --------------------------------------------
        # Storing SummaryFrame as TrajStore attribute.
        # --------------------------------------------
        # Where user specifies a SummaryFrame from Polars DataFrame or LazyFrame:
        if summary_source is not None:
            # Defining summary LazyFrame SummaryFrame containing a single
            # row per trajectory.
            self.SummaryFrame = summary_source
        elif summary_source is None:
            # Defining SummaryFrame as None.
            self.SummaryFrame = None

        # --------------------------------------
        # Extracting TrajFrame column variables.
        # --------------------------------------
        if self.TrajFrame is not None:
            # Storing list of column variables contained in TrajFrame.
            self.columns = self.TrajFrame.columns

            # Raise error if any core column variables are absent from TrajFrame.
            if 'id' not in self.columns:
                raise ValueError("core variable missing from TrajFrame: \'id\'")
            if 'time' not in self.columns:
                raise ValueError("core variable missing from TrajFrame: \'time\'")
        else:
            # Storing list of column variables in SummaryFrame.
            self.columns = self.SummaryFrame.columns

        # ----------------------------------------------------
        # Storing LazyFrame query plan as TrajStore attribute.
        # ----------------------------------------------------
        # Query plan for TrajFrame:
        if self.traj_mode == 'lazy':
            # Store optimised query plan description as query_plan:
            self.traj_query_plan = self.TrajFrame.explain()
        else:
            # No query plan when using eager mode or without TrajectoryFrame:
            self.traj_query_plan = None

##############################################################################
# Define print() method.

    def __str__(self):
        # Return summary of TrajFrame and SummaryFrame
        # stored in TrajStore object.

        # Construct summary string for TrajFrame:
        if self.traj_mode is None:
            traj_str = ''
        elif self.traj_mode == 'eager':
            traj_str = f"<TrajStore object>\n\n----- Trajectory DataFrame -----\nObservations: {self.TrajFrame.shape[0]}\nVariables: {self.columns}\n{self.TrajFrame.glimpse}\n"
        elif self.traj_mode == 'lazy':
            traj_str = f"<TrajFrame object>\n\n----- Trajectory LazyFrame -----\nSchema: {self.TrajFrame.schema}\nOptimised Query Plan:\n{self.traj_query_plan}\n"

        # Construct summary string for TrajFrame:
        summary_str = f"\n----- Summary DataFrame -----\nObservations: {self.SummaryFrame.shape[0]}\nVariables: {self.SummaryFrame.columns}\n{self.SummaryFrame.glimpse}\n"

        return traj_str + summary_str

##############################################################################
# Define len() method.

    def __len__(self):
        # Return the total number of trajectories in SummaryFrame:
        return self.SummaryFrame.shape[0]

##############################################################################
# Define collect() method.

    def collect(self, streaming=False, **kwargs):
        """
        Execute LazyFrame optimised query and collect DataFrame.

        Parameters
        ----------
        streaming : bool
            Run parts of the query in a streaming fashion
            (this is in an alpha state). Default is False.
        **kwargs (optional)
            Additional keyword arguments to be passed to Polars
            collect() function.

        Returns
        --------
        TrajStore object
            TrajStore object is returned with one or more eager
            DataFrames following query execution.

        Examples
        --------
        Execute and collect the trajectories resulting from a simple
        filter without implementing streaming:

        >>> trajectories.filter('id < 100').collect(frame='all', streaming=False)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(streaming, bool) is False:
            raise TypeError("streaming must be specified as a boolean")

        # ------------------------------------------------------------
        # Collect LazyFrame(s) as DataFrame following query execution.
        # ------------------------------------------------------------
        # Execute optimised query plan on trajectory frame.
        trajectory_data = self.TrajFrame.collect(streaming=streaming, **kwargs)

        # Return TrajStore object including eager TrajFrame and/or SummaryFrame.
        return TrajStore(source=trajectory_data, summary_source=self.SummaryFrame)

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
        TrajStore object
            Original TrajStore object is returned with transformed
            time attribute variable Series containing datetimes.

        Examples
        --------
        Convert time in TrajStore object to datetime with start
        date '2000-01-01 using default Datetime format.

        >>> trajectories.use_datetime(start_date='2000-01-01')
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(start_date, str) is False:
            raise TypeError("start_date must be specified as a string")
        if isinstance(self.TrajFrame.schema['time'], pl.Datetime) is True:
            raise TypeError("time already exists with dtype = \'Datetime\'")

        # --------------------------------------------------
        # Convert time to Datetime format with start_date.
        # --------------------------------------------------
        # Redefining time variable in Datetime format.
        trajectory_data = self.TrajFrame.with_column(
            (pl.col("time")*np.timedelta64(1, 's')).cast(pl.Duration) + pl.lit(start_date).str.strptime(pl.Datetime, format=fmt)
        )

        # Return TrajStore object with updated TrajFrame.
        return TrajStore(source=trajectory_data, summary_source=self.SummaryFrame)

##############################################################################
# Define filter() method.

    def filter(self, expr, drop=False):
        """
        Filter trajectories using conditional on a single column variable
        specified with a string expression.

        Filtering returns a reduced TrajectoryStore where only the
        complete trajectories meeting the specified condition are retained.
        The exception is when users specify drop=True, in which case
        trajectories meeting the specified condition are dropped from the
        TrajFrame.

        Filtering also reduces the SummaryFrame, retaining only the
        summary statistics available for trajectories meeting the specfied
        condition. The exception is when users specify drop=True.

        When the expression variable is specified as 'time' only the
        observations (obs) meeting the specified date conditions
        are retained for all trajectories.

        Parameters
        ----------
        variable : string
            Name of the variable in the TrajStore.
        expr : string
            String expression of the form "{variable} {operator} {value}",
            where {variable} represents the column variable contained in
            the TrajFrame used to filter trajectories, {operator}
            represents one of the six standard comparison operators and
            {value} represents the value with which to compare the {variable}
            to.
        drop : boolean
            Indcates if fitered trajectories should be retained in the
            new TrajStore (False) or instead dropped from the
            existing TrajStore (True).

        Returns
        -------
        TrajStore object
            Complete TrajStore, including the complete Lagrangian trajectories
            which meet (do not meet) the specified filter condition.

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

        # Defining list of variables contained in TrajFrame.
        col_variables = list(self.columns)

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

        # -------------------------------------------------------
        # Applying specified filter to TrajFrame & Summary Frame.
        # -------------------------------------------------------
        if self.TrajFrame is not None:
            # Determine dtype of filter column values:
            value_dtype = self.TrajFrame.schema[variable]
            # Filter TrajFrame using specified expression:
            trajectory_data = filter_traj(df=self.TrajFrame,
                                          variable=variable,
                                          operator=operator,
                                          value=value,
                                          value_dtype=value_dtype,
                                          drop=drop
                                          )
            if self.SummaryFrame is not None:
                # Update SummaryFrame:
                if  self.traj_mode == 'eager':
                    # Filter SummaryFrame using IDs in filtered eager TrajFrame:
                    traj_ids = trajectory_data['id'].unique()
                    summary_data = self.SummaryFrame.filter(pl.col('id').is_in(traj_ids))
                elif self.traj_mode == 'lazy':
                    # Filter SummaryFrame using IDs in filtered lazy TrajFrame:
                    traj_ids = trajectory_data['id'].unique().collect(streaming=True)
                    summary_data = self.SummaryFrame.filter(pl.col('id').is_in(traj_ids))
            else:
                # No need to update SummaryFrame:
                summary_data = self.SummaryFrame

        else:
            # Determine dtype of filter column values:
            value_dtype = self.SummaryFrame.schema[variable]
            # Filter only SummaryFrame using specified expression:
            summary_data = filter_traj(df=self.SummaryFrame,
                                          variable=variable,
                                          operator=operator,
                                          value=value,
                                          value_dtype=value_dtype,
                                          drop=drop
                                          )
            # No need to update TrajFrame:
            trajectory_data = self.TrajFrame

        # Return TrajStore object with updated TrajFrame & SummaryFrame.
        return TrajStore(source=trajectory_data, summary_source=summary_data)

##############################################################################
# Define filter_polygon() method.

    def filter_polygon(self, xy_vars, x_poly, y_poly, drop=False):
        """
        Filter trajectories which intersect a specified polygon.

        Filtering returns the complete trajectories of particles
        which have been inside the boundary of a given polygon at
        any point in their lifetime.

        Parameters
        ----------
        xy_vars : list(str)
            List of x and y coordinate variable names.
        x_poly : list
            List of x-coordinates representing the boundary of the polygon.
        y_poly : list
            List of y-coordinates representing the boundary of the polygon.
        drop : boolean
            Determines if fitered trajectories should be returned as a
            new TrajStore (False) or instead dropped from the
            existing TrajStore (True).

        Returns
        -------
        TrajStore object
            Complete TrajStore, including the complete Lagrangian trajectories
            which meet (do not meet) the specified filter condition.

        Examples
        --------
        Filtering all trajectories which intersect a simple square polygon:

        >>> x_square = [-40, -40, -30, -30, -40]
        >>> y_square = [30, 35, 35, 30, 30]
        >>> trajectories.filter_polygon(xy_vars=['x','y'], x_poly=x_square, y_poly=y_square, drop=False)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(drop, bool) is False:
            raise TypeError("drop must be specified as a boolean")

        if isinstance(xy_vars, list) is False:
            raise TypeError("x and y coordinate variables must be specified in a list")

        if isinstance(x_poly, list) is False:
            raise TypeError("polygon x coordinates must be given as a list")

        if isinstance(y_poly, list) is False:
            raise TypeError("polygon y coordinates must be given as a list")

        # --------------------------------
        # Defining the filtered TrajFrame.
        # --------------------------------
        if self.traj_mode == 'eager':
            trajectory_data = filter_traj_polygon(self.TrajFrame, xy_vars=xy_vars, x_poly=x_poly, y_poly=y_poly, drop=drop)
        elif self.traj_mode == 'lazy':
            trajectory_data = (self.TrajFrame
                            .map_batches(lambda df : filter_traj_polygon(df, xy_vars=xy_vars, x_poly=x_poly, y_poly=y_poly, drop=drop))
                            )

        # Return TrajStore object with updated TrajFrame & SummaryFrame.
        return TrajStore(source=trajectory_data, summary_source=self.SummaryFrame)

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
        and stored for each trajectory in the SummaryFrame.

        Parameters
        ----------
        self : TrajStore
            TrajStore passed from TrajStore class method.

        Returns
        -------
        TrajStore
            Original TrajStore is returned with appended column
            variable {start_time} in the SummaryFrame
            containing the times each trajectory was seeded.

        Examples
        --------
        Get seeding times for all trajectories.

        >>> trajectories.get_start_time()
        """
        # -------------------------------------------
        # Calculate seeding time for each trajectory.
        # -------------------------------------------
        start_data = (self.TrajFrame
                             .groupby(pl.col("id"))
                             .agg(pl.col('time').min().alias('start_time'))
                             )

        # -----------------------------------
        # Adding start_time to summary frame.
        # -----------------------------------
        if self.traj_mode == 'lazy':
            # Collecting start time values in DataFrame:
            start_data = start_data.collect(streaming=True)

        # Joining start time values to SummaryFrame:
        if self.SummaryFrame is None:
            summary_data = start_data.sort(by='id')
        else:
            summary_data = self.SummaryFrame.join(start_data, on='id', how='left').sort(by='id')

        # Return TrajStore with updated SummaryFrame.
        return TrajStore(source=self.TrajFrame, summary_source=summary_data)

##############################################################################
# Define get_start_loc() method.

    def get_start_loc(self):
        """
        Returns locations where water parcels are seeded (start of trajectory).

        The start locations of each trajectory are added in the form of three
        new column variables in the SummaryFrame: {lon_start}, {lat_start}
        and {depth_start}.

        Parameters
        ----------
        self : TrajStore
            TrajStore passed from TrajStore class method.

        Returns
        -------
        TrajStore
            Original TrajStore is returned with appended column
            variables {lon_start}, {lat_start}, {depth_start} in
            the SummaryFrame.

        Examples
        --------
        Get seeding locations for all trajectories.

        >>> trajectories.get_start_loc()
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Defining list of column variables contained in TrajFrame.
        col_variables = list(self.columns)

        if ('lon' not in col_variables) | ('lat' not in col_variables) | ('depth' not in col_variables):
            raise ValueError("required variable missing from TrajFrame: \'lon\', \'lat\', \'depth\'")

        # --------------------------------------------------
        # Return start longitude, latitude and depth values.
        # --------------------------------------------------
        start_data = (self.TrajFrame
                      .groupby('id')
                      .agg([
                          pl.col(['lon', 'lat', 'depth']).sort_by('time').first(),
                          ])
                      .rename({'lon':'lon_start', 'lat':'lat_start', 'depth':'depth_start'})
                      )

        # ----------------------------------------
        # Adding lat/lon/depth_start to DataFrame.
        # ----------------------------------------
        if self.traj_mode == 'lazy':
            # Collecting start locations values in DataFrame:
            start_data = start_data.collect(streaming=True)

        # Joining start locations values to SummaryFrame:
        if self.SummaryFrame is None:
            summary_data = start_data.sort(by='id')
        else:
            summary_data = self.SummaryFrame.join(start_data, on='id', how='left').sort(by='id')

        # Return TrajStore with updated SummaryFrame.
        return TrajStore(source=self.TrajFrame, summary_source=summary_data)

##############################################################################
# Define get_end_time() method.

    def get_end_time(self):
        """
        Returns time when water parcels are terminated (end of trajectory).

        The end time is given in the format of the {time} core variable
        and stored for each trajectory in the SummaryFrame.

        Parameters
        ----------
        self : TrajStore
            TrajStore passed from TrajStore class method.

        Returns
        -------
        TrajStore
            Original TrajStore is returned with appended column
            variable {end_time} in the SummaryFrame
            containing the times each trajectory was terminate.

        Examples
        --------
        Get end times for all trajectories.

        >>> trajectories.get_end_time()
        """
        # ----------------------------------------------
        # Calculate termination time of each trajectory.
        # ----------------------------------------------
        end_data = (self.TrajFrame
                           .groupby(pl.col("id"))
                           .agg(pl.col('time').max().alias('end_time'))
                           )

        # -------------------------------
        # Adding end_time to DataFrame.
        # -------------------------------
        if self.traj_mode == 'lazy':
            # Collecting end time values in DataFrame:
            end_data = end_data.collect(streaming=True)

        # Joining end time values to SummaryFrame:
        if self.SummaryFrame is None:
            summary_data = end_data.sort(by='id')
        else:
            summary_data = self.SummaryFrame.join(end_data, on='id', how='left').sort(by='id')

        # Return TrajStore with updated SummaryFrame.
        return TrajStore(source=self.TrajFrame, summary_source=summary_data)

##############################################################################
# Define get_end_loc() method.

    def get_end_loc(self):
        """
        Returns locations where water parcels are terminated (end of trajectory).

        The end locations of each trajectory are added in the form of three
        new column variables in the SummaryFrame: {lon_end}, {lat_end}
        and {depth_end}.

        Parameters
        ----------
        self : TrajStore
            TrajStore object passed from TrajStore class method.

        Returns
        -------
        TrajStore
            Original TrajStore is returned with appended column
            variables {lon_end}, {lat_end}, {depth_end} in
            the SummaryFrame.

        Examples
        --------
        Get end locations for all trajectories.

        >>> trajectories.get_end_loc()
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Defining list of column variables contained in TrajFrame.
        col_variables = list(self.columns)

        if ('lon' not in col_variables) | ('lat' not in col_variables) | ('depth' not in col_variables):
            raise ValueError("required variable missing from TrajFrame: \'lon\', \'lat\', \'depth\'")

        # --------------------------------------------------
        # Return end longitude, latitude and depth values.
        # --------------------------------------------------
        end_data = (self.TrajFrame
                    .groupby('id')
                    .agg([
                        pl.col(['lon', 'lat', 'depth']).sort_by('time').last(),
                        ])
                    .rename({'lon':'lon_end', 'lat':'lat_end', 'depth':'depth_end'})
                    )

        # ----------------------------------------
        # Adding lat/lon/depth_end to DataFrame.
        # ----------------------------------------
        if self.traj_mode == 'lazy':
            # Collecting end locations values in DataFrame:
            end_data = end_data.collect(streaming=True)

        # Joining end locations values to SummaryFrame:
        if self.SummaryFrame is None:
            summary_data = end_data.sort(by='id')
        else:
            summary_data = self.SummaryFrame.join(end_data, on='id', how='left').sort(by='id')

        # Return TrajStore with updated SummaryFrame.
        return TrajStore(source=self.TrajFrame, summary_source=summary_data)

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
        self : TrajStore
            TrajStore passed from TrajStore class method.

        Returns
        -------
        TrajStore
            Original TrajStore is returned with appended column
            variable {total_time} in the SummaryFrame.

        Examples
        --------
        Get duration of all trajectories.

        >>> trajectories.get_duration()
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Raise exception if {time} column variable is not stored as polar Datetime dtype.
        if isinstance(self.TrajFrame.schema["time"], pl.Datetime) is False:
            raise TypeError("times must be stored as polars Datetime dtype")

        # -------------------------------------
        # Calculate duration of each trajectory.
        # -------------------------------------
        duration_data =(self.TrajFrame
                        .groupby(pl.col("id"))
                        .agg((pl.col("time").max() - pl.col("time").min()).cast(pl.Duration).alias("duration"))
                        )

        # -----------------------------
        # Adding duration to DataFrame.
        # -----------------------------
        if self.traj_mode == 'lazy':
            # Collecting duration values in DataFrame:
            duration_data = duration_data.collect(streaming=True).sort('id')

        # Joining duration values to SummaryFrame:
        if self.SummaryFrame is None:
            summary_data = duration_data
        else:
            summary_data = self.SummaryFrame.join(duration_data, on='id')

        # Return TrajStore with updated SummaryFrame.
        return TrajStore(source=self.TrajFrame, summary_source=summary_data)

##############################################################################
# Define get_values() function.

    def get_values(self, var, date, fmt="%Y-%m-%d"):
        """
        Returns the values of a specified variable at a specified
        date for each trajectory.

        The values of the specified variable are returned for all
        trajectories in the SummaryFrame.

        Parameters
        ----------
        self : TrajStore
            TrajStore passed from TrajStore class method.
        var : string
            Name of the variable in the TrajFrame.
        date : string
            date on which to get values of specified variable.
        fmt : string
            Datetime format of specified date. Default
            format is YYYY-MM-DD.

        Returns
        -------
        TrajStore
            Original TrajStore is returned with appended column
            variable {var}_i in the SummaryFrame.

        Examples
        --------
        Get the value of temperature for each trajectory on date 2000-01-31.
        Note that we must convert time to polars Datetime format before using
        .get_values().

        >>>  trajectories.use_datetime(start_time='2000-01-01').get_values(var='temp', date='2000-01-31')
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Defining list of column variables contained in TrajFrame.
        variables = list(self.columns)

        if isinstance(var, str) is False:
            raise TypeError("variable must be specified as a string")

        if var not in variables:
            raise ValueError("variable: \'" + var + "\' not found in TrajFrame")

        if isinstance(date, str) is False:
            raise TypeError("date must be specified as a string in format givent to fmt")

        # ------------------------------------------------------
        # Extract values of specfied variable at specified date.
        # ------------------------------------------------------
        # Filtering TrajFrame for specified date and then selecting specified variable.
        values_data = (self.TrajFrame
                         .filter(pl.col("time") == pl.lit(date).str.strptime(pl.Datetime, format=fmt))
                         .select([pl.col('id'), pl.col(var).alias(var + "_i")])
                         )

        # ---------------------------------
        # Adding variable_i to DataFrame.
        # ---------------------------------
        # Appending variable values at specified date as new column variable in SummaryFrame.
        if self.traj_mode == 'lazy':
            # Collecting values in DataFrame:
            values_data = values_data.collect(streaming=True).sort('id')

        # Joining values to SummaryFrame:
        if self.SummaryFrame is None:
            summary_data = values_data 
        else:
            summary_data = self.SummaryFrame.join(values_data, on='id')

        # Return TrajStore with updated SummaryFrame.
        return TrajStore(source=self.TrajFrame, summary_source=summary_data)

##############################################################################
# Define get_max() function.

    def get_max(self, var):
        """
        Returns maximum value of a specified variable for each trajectory.

        The maximum value of the variable is returned for all trajectories
        as a new column variable in the SummaryFrame.

        Parameters
        ----------
        self : TrajStore
            TrajStore passed from TrajStore class method.
        var : string
            Name of the variable in the TrajFrame.

        Returns
        -------
        TrajStore
            Original TrajStore is returned with appended column
            variable {variable}_max in the SummaryFrame
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
        col_variables = self.columns

        if isinstance(var, str) is False:
            raise TypeError("variable must be specified as a string")

        if var not in col_variables:
            raise ValueError("variable: \'" + var + "\' not found in main DataFrame")

        # -----------------------------------------------
        # Calculate maximum values of variable as Series.
        # -----------------------------------------------
        # Defining std. name of max_values using specified variable.
        max_variable = var + "_max"

        # Calculate maximum value of {variable} along each trajectory.
        max_data = (self.TrajFrame
                    .groupby(pl.col("id"))
                    .agg(pl.col(var).max().alias(max_variable))
                    )

        # ----------------------------------
        # Adding max_values to SummaryFrame.
        # ----------------------------------
        if self.traj_mode == 'lazy':
            # Collecting maximum values in DataFrame:
            max_data = max_data.collect(streaming=True).sort('id')

        # Joining maximum values to SummaryFrame:
        if self.SummaryFrame is None:
            summary_data = max_data  
        else:
            summary_data = self.SummaryFrame.join(max_data, on='id')

        # Return TrajStore with updated SummaryFrame.
        return TrajStore(source=self.TrajFrame, summary_source=summary_data)

##############################################################################
# Define get_min() function.

    def get_min(self, var):
        """
        Returns minimum value of a specified variable for each trajectory.

        The minimum value of the variable is returned for all trajectories
        as a new column variable in the SummaryFrame.

        Parameters
        ----------
        self : TrajStore
            TrajStore passed from TrajStore class method.
        var : string
            Name of the variable in the TrajFrame.

        Returns
        -------
        TrajStore
            Original TrajStore is returned with appended column
            variable {variable}_min in the SummaryFrame
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
        col_variables = self.columns

        if isinstance(var, str) is False:
            raise TypeError("variable must be specified as a string")

        if var not in col_variables:
            raise ValueError("variable: \'" + var + "\' not found in main DataFrame")

        # -----------------------------------------------
        # Calculate minimum values of variable as Series.
        # -----------------------------------------------
        # Defining std. name of min_values using specified variable.
        min_variable = var + "_min"

        # Calculate minimum value of {variable} along each trajectory.
        min_data = (self.TrajFrame
                    .groupby(pl.col("id"))
                    .agg(pl.col(var).min().alias(min_variable))
                    )

        # ----------------------------------
        # Adding min_values to SummaryFrame.
        # ----------------------------------
        if self.traj_mode == 'lazy':
            # Collecting maximum values in DataFrame:
            min_data = min_data.collect(streaming=True).sort('id')

        # Joining minimum values to SummaryFrame:
        if self.SummaryFrame is None:
            summary_data = min_data
        else:
            summary_data = self.SummaryFrame.join(min_data, on='id')

        # Return TrajStore with updated SummaryFrame.
        return TrajStore(source=self.TrajFrame, summary_source=summary_data)

##############################################################################
# Define add_variable() method.

    def add_variable(self, data, var, summary_var=False):
        """
        Adds a new variable to the existing TrajStore object.

        The variable data must be provided as 1-dimensional ndarray
        with the variable attributes provided as a string.

        Parameters
        ----------
        self : TrajStore
            TrajStore object passed from TrajStore class method.
        data : ndarray
            values of new variable to be added to the TrajStore.
        var : str
            new variable name to be added to TrajStore.
        summary_var : bool
            indicates if new column variable should be added to the
            SummaryFrame (True) or TrajFrame (False).
            The Default value is False.

        Returns
        -------
        TrajStore
            Original TrajStore is returned with the new column variable
            appended to the TrajFrame or SummaryFrame.

        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(data, np.ndarray) is False:
            raise TypeError("data must be specified as an ndarray")
        if isinstance(var, str) is False:
            raise TypeError('variable name must be specfied as a string')
        if isinstance(summary_var, bool) is False:
            raise TypeError("summary_var must be specified as a boolean value")

        # -----------------------------------------------------------------------
        # Returning updated TrajFrame or SummaryFrame with new column variable.
        # -----------------------------------------------------------------------
        # Updating DataFrame conditional on summary_var boolean value:
        if summary_var is True:
            # If no SummaryFrame exists define eager DataFrame with unique IDs:
            if self.SummaryFrame is None:
                if self.traj_mode == 'eager':
                    self.SummaryFrame = self.TrajFrame['id'].unique()
                elif self.traj_mode == 'lazy':
                    self.SummaryFrame = self.TrajFrame['id'].unique().collect(streaming=True)
            # Append new column variable to SummaryFrame.
            summary_data = self.SummaryFrame.with_column(pl.Series(name=var, values=data))
            # Keep existing TrajFrame.
            trajectory_data = self.TrajFrame
        elif summary_var is False:
            # Append new column variable to TrajFrame.
            trajectory_data = self.TrajFrame.with_column(pl.Series(name=var, values=data))
            # Keep existing SummaryFrame.
            summary_data = self.SummaryFrame

        # Return TrajStore with updated TrajFrame or SummaryFrame.
        return TrajStore(source=trajectory_data, read_mode=None, summary_source=summary_data, rename_cols=None)
