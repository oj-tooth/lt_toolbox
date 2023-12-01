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
import xarray as xr

# Importing utility functions
from .utils.filter_frame_utils import filter_traj_polygon, filter_traj, filter_summary
from .utils.compute_frame_utils import haversine_dist, binned_statistic_1d, binned_statistic_2d, binned_group_statistic_1d, binned_group_statistic_2d
from .utils.transform_frame_utils import transform_coords

##############################################################################
# Define TrajStore Class.


class TrajStore:

    def __init__(self, traj_source, read_mode='eager', read_kwargs=None, rename_cols=None, summary_source=None, summary_array=None):
        """
        Create a TrajStore from a single filepath, list of filepaths, Polars DataFrame
        or Polars LazyFrame.

        Parameters
        ----------
        traj_source: str | list
            Path or list of paths to .csv or .parquet file(s) containing Lagrangian
            trajectories in tabular format. Alternatively, a Polars DataFrame or
            LazyFrame may be specified.
        read_mode : str
            Select either 'eager' or 'lazy' mode to read traj_source files.
            Default is to use Polars lazy API, only executing full queries
            when specified.
        read_kwargs : dict
            Additional keyword arguments to be passed to Polars read_csv(),
            read_parquet(), scan_csv() or scan_parquet() function when
            constructing TrajStore object.
        rename_cols : dict
            Rename columns using key value pairs that map from old name to new
            name.
        summary_source : DataFrame
            Summary data stored for each trajectory in TrajFrame, df.
            Default value is None since TrajFrame is assumed to
            contain multiple rows for each trajectory. An exception is made
            when the TrajFrame contains only a single row per trajectory.
            In this case, TrajFrame and SummaryFrame are equivalent.
        summary_array : DataSet
            DataSet storing summary statistics in the form of n-dimensional 
            DataArrays generated from Lagrangian trajectory data contained in
            the TrajStore.

        Returns
        --------
        TrajStore object
            Complete trajectories, including all column variables contained
            in TrajectoryFrame. Summary data stored for each trajectory in
            SummaryFrame. Summary statistics stored as n-dimensional
            arrays in SummaryArray.

        Examples
        --------
        Creating TrajStore object, trajectories, with example_trajectories.csv file in eager mode.

        >>> filename = 'example_trajectories.csv'
        >>> trajectories = TrajStore(traj_source=filename, summary_source=None, read_mode='eager')

        Creating TrajStore object, traj, with multiple parquet files in lazy mode.

        >>> filenames = [ 'example_trajectories1.parquet', 'example_trajectories2.parquet']
        >>> trajectories = TrajStore(traj_source=filenames, summary_source=None, read_mode='lazy')

        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Raising exceptions where a TrajectoryFrame source is specified:
        if traj_source is not None:
            # Determine if source is Polars DataFrame:
            is_DataFrame = isinstance(traj_source, pl.DataFrame)
            # Determine if source is Polars LazyFrame:
            is_LazyFrame = isinstance(traj_source, pl.LazyFrame)

            # Raise error if source is not a string, list or Polars DataFrame / LazyFrame:
            if (isinstance(traj_source, str) | isinstance(traj_source, list) | is_DataFrame | is_LazyFrame) is False:
                raise TypeError("source file path(s) must be specified as string or list of strings")

            # Raise error if source list does not contain strings:
            if isinstance(traj_source, list):
                if any(isinstance(file, str) for file in traj_source) is False:
                    raise TypeError("source file paths must be specified as list of strings")

            # Raise error if source filepath does not contain .csv or .parquet extension:
            if isinstance(traj_source, str):
                if (traj_source.endswith('.csv')) | (traj_source.endswith('.parquet')) is False:
                    raise TypeError("source file must be specified as either .csv or .parquet")
            if isinstance(traj_source, list):
                if any(((file.endswith('.csv')) | (file.endswith('.parquet'))) for file in traj_source) is False:
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

        # Raise error if summary_array is not an xarray DataSet:
        if summary_array is not None:
            if isinstance(summary_array, xr.Dataset) is False:
                raise TypeError("summary_source must be specified an xarray DataSet")

        # Raise error if attrs is not a dictionary:
        if rename_cols is not None:
            if isinstance(rename_cols, dict) is False:
                raise TypeError("rename columns mapping specified as a dictionary")

        # ----------------------------------------------------------
        # Constructing Lagrangian trajectory frame from traj_source:
        # ----------------------------------------------------------
        if traj_source is not None:
            # Case 1. From specified Polars DataFrame or LazyFrame:
            if (is_DataFrame | is_LazyFrame) is True:
                df = traj_source

            # Case 2. Reading input file(s) as Polars DataFrame or LazyFrame.
            else:
                # Eager reading of source file(s).
                if read_mode == 'eager':
                    # Sub Case 1. Single file:
                    if isinstance(traj_source, str):
                        if traj_source.endswith('.csv'):
                            # Reading DataFrame from source .csv file:
                            df = pl.read_csv(traj_source, **(read_kwargs or {}))
                        elif traj_source.endswith('.parquet'):
                            # Reading DataFrame from source .parquet file:
                            df = pl.read_parquet(traj_source, **(read_kwargs or {}))
                    # Sub Case 2. Multiple files:
                    if isinstance(traj_source, list):
                        if traj_source[0].endswith('.csv'):
                            # Reading DataFrame from source .csv file:
                            df = pl.concat([pl.read_csv(file, **(read_kwargs or {})) for file in traj_source])
                        elif traj_source[0].endswith('.parquet'):
                            # Reading DataFrame from source .parquet file:
                            df = pl.concat([pl.read_parquet(file, **(read_kwargs or {})) for file in traj_source])

                # Lazy reading of source file(s).
                if read_mode == 'lazy':
                    # Sub Case 1. Single file:
                    if isinstance(traj_source, str):
                        if traj_source.endswith('.csv'):
                            # Scan LazyFrame from source .csv file:
                            df = pl.scan_csv(traj_source, **(read_kwargs or {}))
                        elif traj_source.endswith('.parquet'):
                            # Scan LazyFrame from source .parquet file:
                            df = pl.scan_parquet(traj_source, **(read_kwargs or {}))
                    # Sub Case 2. Multiple files:
                    if isinstance(traj_source, list):
                        if traj_source[0].endswith('.csv'):
                            # Scan LazyFrame from source .csv file:
                            df = pl.concat([pl.scan_csv(file, **(read_kwargs or {})) for file in traj_source])
                        elif traj_source[0].endswith('.parquet'):
                            # Scan LazyFrame from source .parquet file:
                            df = pl.concat([pl.scan_parquet(file, **(read_kwargs or {})) for file in traj_source])

        # Renaming columns of trajectory frame:
        if rename_cols is not None:
            # Rename columns using key value pairs that map from
            # old name to new name:
            df = df.rename(rename_cols)

        # ------------------------------------------------------
        # Storing input trajectory frame as TrajStore attribute.
        # ------------------------------------------------------
        if traj_source is not None:
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

        # --------------------------------------------
        # Storing SummaryArray as TrajStore attribute.
        # --------------------------------------------
        if summary_array is not None:
            # Defining SummaryArray from specified Dataset.
            self.SummaryArray = summary_array
        elif summary_array is None:
            # Defining SummaryArray as empty Dataset.
            self.SummaryArray = xr.Dataset()

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
        if self.SummaryFrame is not None:
            summary_str = f"\n----- Summary DataFrame -----\nObservations: {self.SummaryFrame.shape[0]}\nVariables: {self.SummaryFrame.columns}\n{self.SummaryFrame.glimpse}\n"
        else:
            summary_str = ""

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
        return TrajStore(traj_source=trajectory_data, summary_source=self.SummaryFrame)

##############################################################################
# Define use_datetime.

    def use_datetime(self, start_date:str, fmt="%Y-%m-%d"):
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
        return TrajStore(traj_source=trajectory_data, summary_source=self.SummaryFrame)

##############################################################################
# Define transform_trajectory_coords() method.

    def transform_trajectory_coords(self, lon:xr.DataArray, lat:xr.DataArray, depth:xr.DataArray):
        """
        Transform Lagrangian trajectories from model grid coordinates {i,j,k}.
        to geographical coordinates {lon, lat, depth}.

        Lagrangian trajectory positions are (bi-)linearly interpolated from
        the specified ocean general circulation model grid.

        Parameters
        ----------
        self : TrajStore object
            TrajStore object containing Lagrangian trajectories in model coords
            {i, j, k}.
        lon : DataArray
            Longitudes associated with the center of each model grid cell.
        lat : DataArray
            Latitudes associated with the center of each model grid cell.
        depth : DataArray
            Depths associated with model vertical grid levels.

        Returns
        -------
        TrajStore object
            TrajStore containing Lagrangian trajectories in geographical
            coords {lon, lat, depth}.

        Examples
        --------
        Transforming Lagrangian trajectories with positions referenced to model
        coordinate system {x, y, z} to geographical coordinates {lon, lat, depth}
        using the ocean general circulation horizontal and vertical model grids.
        Here, we show a simple example for the Nucleus for European Modelling
        of the Ocean ORCA C-grid:

        >>> lon_mdl = ds_grid.nav_lon
        >>> lat_mdl = ds_grid.nav_lat
        >>> depth_mdl = ds_grid.nav_lev
        >>> trajectories.transform_trajectory_coords(lon=lon_mdl, lat=lat_mdl, depth=depth_mdl, drop=True)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(lon, xr.DataArray) is False:
            raise TypeError("longitude array must be specified as an xarray DataArray")

        if isinstance(lat, xr.DataArray) is False:
            raise TypeError("latitude array must be specified as an xarray DataArray")

        if isinstance(depth, xr.DataArray) is False:
            raise TypeError("depth array must be specified as an xarray DataArray")

        # ---------------------------------------------------------
        # Transforming Lagrangian Trajectories stored in TrajFrame.
        # ---------------------------------------------------------
        if self.traj_mode == 'eager':
            trajectory_data = self.TrajFrame.pipe(transform_coords,
                                                lon=lon,
                                                lat=lat,
                                                depth=depth
                                                )
        elif self.traj_mode == 'lazy':
            trajectory_data = (self.TrajFrame
                              .map_batches(lambda df : transform_coords(df, lon=lon, lat=lat, depth=depth))
                              )

        # Rename TrajFrame position columns to geographic coords:
        trajectory_data = trajectory_data.rename({'x':'lon', 'y':'lat', 'z':'depth'})  

        # Return TrajStore object with updated TrajFrame & SummaryFrame.
        return TrajStore(traj_source=trajectory_data, summary_source=self.SummaryFrame)

##############################################################################
# Define filter() method.

    def filter(self, expr:str, drop=False):
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

        # Defining list of standard operators.
        operator_list = ['==', '!=', '<', '>', '<=', '>=']

        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

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
            summary_data = filter_summary(df=self.SummaryFrame,
                                          variable=variable,
                                          operator=operator,
                                          value=value,
                                          value_dtype=value_dtype,
                                          drop=drop
                                          )
            # No need to update TrajFrame:
            trajectory_data = self.TrajFrame

        # Return TrajStore object with updated TrajFrame & SummaryFrame.
        return TrajStore(traj_source=trajectory_data, summary_source=summary_data)

##############################################################################
# Define filter_polygon() method.

    def filter_polygon(self, xy_vars:list, x_poly:list, y_poly:list, drop=False):
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
            trajectory_data = self.TrajFrame.pipe(filter_traj_polygon, xy_vars=xy_vars, x_poly=x_poly, y_poly=y_poly, drop=drop)
        elif self.traj_mode == 'lazy':
            trajectory_data = (self.TrajFrame
                            .map_batches(lambda df : filter_traj_polygon(df, xy_vars=xy_vars, x_poly=x_poly, y_poly=y_poly, drop=drop), streamable=True)
                            )

        # Return TrajStore object with updated TrajFrame & SummaryFrame.
        return TrajStore(traj_source=trajectory_data, summary_source=self.SummaryFrame)
    
##############################################################################
# Define filter_isin() method.

    def filter_isin(self, var:str, values:pl.Series, drop=False):
        """
        Filter trajectories with at least one variable observation
        in a given sequence of values.

        Filtering returns the complete trajectories of particles
        where one or more observations of the given variable are found
        in the given list or Series of values.

        Parameters
        ----------
        var : str
            Name of variable contained in TrajStore object.
        values : list | Series
            Values of variables used to filter trajectories.
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
        Filtering all trajectories with unique IDs in a given list:

        >>> id_group = [1, 2, 3, 4, 5]
        >>> trajectories.filter_isin(var='id', values=id_group, drop=False)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(drop, bool) is False:
            raise TypeError("drop must be specified as a boolean")

        if isinstance(var, str) is False:
            raise TypeError("variable name must be specified in a string")

        if (isinstance(values, list) | isinstance(values, pl.Series)) is False:
            raise TypeError("values of variable must be given as either a list or Series")

        # -------------------------------------------------------
        # Applying specified filter to TrajFrame & Summary Frame.
        # -------------------------------------------------------
        if self.TrajFrame is not None:
            if drop is False:
                # Filter TrajFrame to store only trajectories with obs in values for variable:
                trajectory_data = self.TrajFrame.filter(pl.col(var).is_in(values))
            else:
                # Filter TrajFrame to store only trajectories without obs in values for variable:
                trajectory_data = self.TrajFrame.filter(~pl.col(var).is_in(values))

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
            # Filter only SummaryFrame using specified expression:
            if drop is False:
                summary_data = self.SummaryFrame.filter(pl.col(var).is_in(values))
            else:
                summary_data = self.SummaryFrame.filter(~pl.col(var).is_in(values))
            # No need to update TrajFrame:
            trajectory_data = self.TrajFrame


        # Return TrajStore object with updated TrajFrame & SummaryFrame.
        return TrajStore(traj_source=trajectory_data, summary_source=summary_data)

##############################################################################
# Define compute_distance() method.

    def compute_distance(self, cum_dist=False, unit='km'):
        """
        Compute distance travelled by particles along their
        of trajectories.

        Either the distance travelled between particle positions
        or the cumulative distance travelled is computed
        and returned for all trajectories a new variable.

        Parameters
        ----------
        self : TrajStore object
            TrajStore object passed from TrajStore class method.
        cum_dist : logical
            Compute the cumulative distance travelled by each particle -
            default is False.
        unit : string
            Unit for distance travelled output - default is 'km' -
            alternative option 'm'.

        Returns
        -------
        TrajStore object.
        Original TrajStore object is returned with new column variable
        containing the distance travelled by each particle along it's
        trajectory.

        The first row for all trajectories is Null since the (cumulative)
        distance from the origin of a particle at the origin is not defined.

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
            raise ValueError("invalid unit - options: \'m\', \'km\'")

        # Raising exception if longitude or latitude variables are
        # not included in TrajFrame:
        if ('lon' not in self.columns) | ('lat' not in self.columns):
            raise ValueError("required variable missing from TrajFrame: \'lon\', \'lat\'")

        # ------------------------------------------
        # Computing distance with haversine_dist().
        # ------------------------------------------
        # Calculate the Haversine distance between neighbouring trajectory positions:
        dist_data = (self.TrajFrame
                           .group_by(by=pl.col('id'), maintain_order=True)
                           .agg(
                               pl.map_groups(exprs=['lon', 'lat'],
                                             function=lambda args: haversine_dist(args[0], args[1], cum_dist=cum_dist)
                                             ).alias('dist')
                                )
                            )
        
        # Explode distance DataFrame from one row per trajectory to one row per observation:
        dist_data = dist_data.explode(columns='dist')

        if cum_dist is True:
            # Add distance column to original TrajFrame as cum_dist:
            trajectory_data = self.TrajFrame.sort(by='id').with_columns(dist_data['dist'].alias('cum_dist'))
        else:
            # Add distance column to original TrajFrame as dist:
            trajectory_data = self.TrajFrame.sort(by='id').with_columns(dist_data['dist'])

        if unit == 'km':
            # Define conversion factor meters to kilometers:
            m_to_km = 1 / 1E3
            if cum_dist is True:
                # Transform distance values from m to km:
                trajectory_data = trajectory_data.with_columns(pl.col('cum_dist') * m_to_km)
            else:
                # Transform distance values from m to km:
                trajectory_data = trajectory_data.with_columns(pl.col('dist') * m_to_km)

        # Return TrajStore object with updated TrajFrame & SummaryFrame.
        return TrajStore(traj_source=trajectory_data, summary_source=self.SummaryFrame)

##############################################################################
# Define compute_binned_statistic_1d() method.

    def compute_binned_statistic_1d(self, var:str, values:str, statistic:str, bin_breaks:list, alias=None, group=None, summary_var=False):
        """
        Compute a 1-dimensional binned statistic using variables stored in
        a TrajStore.

        This is a generalization of a histogram function. A histogram divides
        the chosen column varaible into bins, and returns the count of the number
        of points in each bin. This function allows the computation of the sum,
        mean, median, or other statistic of the values within each bin.

        Parameters
        ----------
        var : str
            Name of variable whose values will binned.

        values : str
            Name of the variable on which the statistic will be computed.

        group : str
            Name of variable to group according to unique values using group_by()
            method. A 1-dimensional binned statistic will be computed for each
            group member.

        statistic: str
            The statistic to compute.
            The following statistics are available:

            * 'mean' : compute the mean of values for points within each bin.
                Empty bins will be represented by null.
            * 'std' : compute the standard deviation within each bin.
            * 'median' : compute the median of values for points within each
                bin. Empty bins will be represented by null.
            * 'count' : compute the count of points within each bin. This is
                identical to an unweighted histogram.
            * 'sum' : compute the sum of values for points within each bin.
                This is identical to a weighted histogram.
            * 'min' : compute the minimum of values for points within each bin.
                Empty bins will be represented by null.
            * 'max' : compute the maximum of values for point within each bin.
                Empty bins will be represented by null.

        bin_breaks: list
            List of bin edges used in the binning of var variable.

        alias : str
            New name of output statistics.

        summary_var: boolean
            Specify if variable to bin is contained in SummaryFrame rather than
            TrajFrame.

        Returns
        -------
        TrajStore
            Original TrajStore is returned with the 1-dimensional binned
            statistic included in the SummaryArray where the mid-points
            of the specified bins are given as the coordinate dimension.
        """
        # -----------------
        # Raise exceptions.
        # -----------------
        if isinstance(summary_var, bool) is False:
            raise TypeError('invalid type - summary_var must be specified as a boolean')
        if isinstance(bin_breaks, list) is False:
            raise TypeError('invalid type - bin_breaks must be specified as a list')
        if alias is not None:
            if isinstance(alias, str) is False:
                raise TypeError('invalid type - alias must be specified as a list')

        if summary_var is True:
            if var not in self.SummaryFrame.columns:
                raise ValueError(f'invalid variable - {var} is not contained in SummaryFrame')
            if values not in self.SummaryFrame.columns:
                raise ValueError(f'invalid variable - {values} is not contained in SummaryFrame')
            if group is not None:
                if group not in self.SummaryFrame.columns:
                    raise ValueError(f'invalid variable - {group} is not contained in SummaryFrame')
        else:
            if var not in self.columns:
                raise ValueError(f'invalid variable - {var} is not contained in TrajFrame')
            if values not in self.columns:
                raise ValueError(f'invalid variable - {values} is not contained in TrajFrame')
            if group is not None:
                if group not in self.columns:
                    raise ValueError(f'invalid variable - {group} is not contained in TrajFrame')

        # ---------------------------------
        # Calculating 1-D binned statistic.
        # ---------------------------------
        if summary_var is True:
            if group is None:
                # Calculate 1-dimensional statistic from SummaryFrame:
                result = binned_statistic_1d(var=self.SummaryFrame[var],
                                             values=self.SummaryFrame[values],
                                             statistic=statistic,
                                             bin_breaks=bin_breaks,
                                            )
            else:
                # Calculate 1-dimensional grouped statistic from SummaryFrame:
                result = binned_group_statistic_1d(var=self.SummaryFrame[var],
                                                   values=self.SummaryFrame[values],
                                                   groups=self.SummaryFrame[group],
                                                   statistic=statistic,
                                                   bin_breaks=bin_breaks,
                                                   )
        else:
            if group is None:
                # Calculate 1-dimensional statistic from TrajFrame:
                result = binned_statistic_1d(var=self.TrajFrame[var],
                                                    values=self.TrajFrame[values],
                                                    statistic=statistic,
                                                    bin_breaks=bin_breaks,
                                                    )
            else:
                # Calculate 1-dimensional grouped statistic from TrajFrame:
                result = binned_group_statistic_1d(var=self.TrajFrame[var],
                                                    values=self.TrajFrame[values],
                                                    groups=self.TrajFrame[group],
                                                    statistic=statistic,
                                                    bin_breaks=bin_breaks,
                                                    )

        # ----------------------------------------
        # Adding 1-D statistic to summary DataSet.
        # ----------------------------------------
        # Add result DataArray to DataSet as named variable:
        if alias is None:
            self.SummaryArray[result.name] = result
        else:
            self.SummaryArray[alias] = result

        # Return TrajStore object with updated TrajFrame & SummaryFrame.
        return TrajStore(traj_source=self.TrajFrame, summary_source=self.SummaryFrame, summary_array=self.SummaryArray)

##############################################################################
# Define compute_binned_statistic_2d() method.


    def compute_binned_statistic_2d(self, var_x:str, var_y:str, values:str, statistic:str, bin_breaks:list, alias=None, group=None, summary_var=False):
        """
        Compute a 2-dimensional binned statistic using the variables stored
        in a TrajStore.

        This is a generalization of a 2-D histogram function. A 2-D histogram divides
        the chosen column varaibles into bins, and returns the count of the number
        of points in each bin. This function allows the computation of the sum,
        mean, median, or other statistic of the values within each bin.

        Parameters
        ----------
        var_x : str
            Name of variable whose values to be binned along the first dimension.

        var_y : Series
            Name of variable whose values to be binned along the second dimension.

        values : Series
            Name of variable on which the statistic will be computed.
            This must be the same length as var_x & var_y.

        group : str
            Name of variable to group according to unique values using group_by()
            method. A 1-dimensional binned statistic will be computed for each
            group member.

        statistic: str
            The statistic to compute.
            The following statistics are available:

            * 'mean' : compute the mean of values for points within each bin.
                Empty bins will be represented by null.
            * 'std' : compute the standard deviation within each bin.
            * 'median' : compute the median of values for points within each
                bin. Empty bins will be represented by null.
            * 'count' : compute the count of points within each bin. This is
                identical to an unweighted histogram.
            * 'sum' : compute the sum of values for points within each bin.
                This is identical to a weighted histogram.
            * 'min' : compute the minimum of values for points within each bin.
                Empty bins will be represented by null.
            * 'max' : compute the maximum of values for point within each bin.
                Empty bins will be represented by null.

        bin_breaks: list
            List of lists including bin edges used in the binning of var_x
            and var_y variables.

        alias : str
            New name of output statistics.

        summary_var: boolean
            Specify if variable to bin is contained in SummaryFrame rather than
            TrajFrame.

        Returns
        -------
        statistic : DataArray
            DataArray containing values of the selected statistic in each bin.
        """
        # -----------------
        # Raise exceptions.
        # -----------------
        if isinstance(summary_var, bool) is False:
            raise TypeError('invalid type - summary_var must be specified as a boolean')
        if isinstance(bin_breaks, list) is False:
            raise TypeError('invalid type - bin_breaks must be specified as a list')
        if alias is not None:
            if isinstance(alias, str) is False:
                raise TypeError('invalid type - alias must be specified as a string')

        if summary_var is True:
            if var_x not in self.SummaryFrame.columns:
                raise ValueError(f'invalid variable - {var_x} is not contained in SummaryFrame')
            if var_y not in self.SummaryFrame.columns:
                raise ValueError(f'invalid variable - {var_y} is not contained in SummaryFrame')
            if values not in self.SummaryFrame.columns:
                raise ValueError(f'invalid variable - {values} is not contained in SummaryFrame')
            if group is not None:
                if group not in self.SummaryFrame.columns:
                    raise ValueError(f'invalid variable - {group} is not contained in SummaryFrame')
        else:
            if var_x not in self.columns:
                raise ValueError(f'invalid variable - {var_x} is not contained in TrajFrame')
            if var_y not in self.columns:
                raise ValueError(f'invalid variable - {var_y} is not contained in TrajFrame')
            if values not in self.columns:
                raise ValueError(f'invalid variable - {values} is not contained in TrajFrame')
            if group is not None:
                if group not in self.columns:
                    raise ValueError(f'invalid variable - {group} is not contained in TrajFrame')

        # ---------------------------------
        # Calculating 2-D binned statistic.
        # ---------------------------------
        if summary_var is True:
            if group is None:
                # Calculate 2-dimensional statistic from SummaryFrame:
                result = binned_statistic_2d(var_x=self.SummaryFrame[var_x],
                                            var_y=self.SummaryFrame[var_y],
                                            values=self.SummaryFrame[values],
                                            statistic=statistic,
                                            bin_breaks=bin_breaks,
                                            )
            else:
                # Calculate 2-dimensional grouped statistic from SummaryFrame:
                result = binned_group_statistic_2d(var_x=self.SummaryFrame[var_x],
                                                   var_y=self.SummaryFrame[var_y],
                                                   values=self.SummaryFrame[values],
                                                   groups=self.SummaryFrame[group],
                                                   statistic=statistic,
                                                   bin_breaks=bin_breaks,
                                                   )
        else:
            if group is None:
                # Calculate 2-dimensional statistic from TrajFrame:
                result = binned_statistic_2d(var_x=self.TrajFrame[var_x],
                                             var_y=self.TrajFrame[var_y],
                                             values=self.TrajFrame[values],
                                             statistic=statistic,
                                             bin_breaks=bin_breaks,
                                             )
            else:
                # Calculate 2-dimensional grouped statistic from TrajFrame:
                result = binned_group_statistic_2d(var_x=self.TrajFrame[var_x],
                                                   var_y=self.TrajFrame[var_y],
                                                   values=self.TrajFrame[values],
                                                   groups=self.TrajFrame[group],
                                                   statistic=statistic,
                                                   bin_breaks=bin_breaks,
                                                   )

        # ----------------------------------------
        # Adding 2-D statistic to summary DataSet.
        # ----------------------------------------
        # Add result DataArray to DataSet as named variable:
        if alias is None:
            self.SummaryArray[result.name] = result
        else:
            self.SummaryArray[alias] = result

        # Return TrajStore object with updated TrajFrame & SummaryFrame.
        return TrajStore(traj_source=self.TrajFrame, summary_source=self.SummaryFrame, summary_array=self.SummaryArray)

##############################################################################
# Define compute_property_lof() method.

    def compute_property_lof(self, subvol:str, prop:str, bin_breaks:list, alias=None, direction='+1', group=None):
        """
        Compute Lagrangian Overturning Function in discrete property-space.

        The net volume transport distribution in the chosen property
        space is accumulated in specified direction.

        Parameters
        ----------
        subvol : str
            Name of the variable storing water parcel volume transport.

        prop : str
            Name of the property variable prefix to bin volume transports.

        bin_breaks: list
            List of bin edges used in the binning volume transports.

        alias: str
            New name for Lagrangian overturning funtion in property-coordinates.

        direction : str
            direction of accumulation: '+1' is smallest to largest, 
            '-1' is largest to smallest.

        group : str
            Name of variable to group according to unique values using group_by()
            method. A Lagrangian overturning function will be computed for each
            group member.

        Returns
        -------
        TrajStore
            Original TrajStore is returned with Lagrangian overturning
            functions included in the SummaryArray where the mid-points
            of the specified bins are given as the coordinate dimension.
        """
        # -----------------
        # Raise exceptions.
        # -----------------
        if isinstance(bin_breaks, list) is False:
            raise TypeError('invalid type - bin_breaks must be specified as a list')
        if subvol not in self.SummaryFrame.columns:
            raise ValueError(f'invalid variable - {subvol} is not contained in SummaryFrame')
        if alias is not None:
            if isinstance(alias, str) is False:
                raise TypeError('invalid type - alias must be specified as a string')
        
        # Define inflow and outflow property variable names:
        prop_in = prop + '_in'
        prop_out = prop + '_out'

        if prop_in not in self.SummaryFrame.columns:
            raise ValueError(f'invalid variable - {prop_in} is not contained in SummaryFrame')
        if prop_out not in self.SummaryFrame.columns:
            raise ValueError(f'invalid variable - {prop_out} is not contained in SummaryFrame')
        if group is not None:
            if group not in self.SummaryFrame.columns:
                raise ValueError(f'invalid variable - {group} is not contained in SummaryFrame')

        # ---------------------------------------------------
        # Calculating 1-D binned statistic in property-space.
        # ---------------------------------------------------
        # Binning Volume Transport according to Inflow Properties:
        if group is None:
            # Calculate 1-dimensional statistic from SummaryFrame:
            result_in = binned_statistic_1d(var=self.SummaryFrame[prop_in],
                                            values=self.SummaryFrame[subvol],
                                            statistic='sum',
                                            bin_breaks=bin_breaks,
                                        )
        else:
            # Calculate 1-dimensional grouped statistic from SummaryFrame:
            result_in = binned_group_statistic_1d(var=self.SummaryFrame[prop_in],
                                                values=self.SummaryFrame[subvol],
                                                groups=self.SummaryFrame[group],
                                                statistic='sum',
                                                bin_breaks=bin_breaks,
                                                )

        # Binning Volume Transport according to Outflow Properties:
        if group is None:
            # Calculate 1-dimensional statistic from SummaryFrame:
            result_out = binned_statistic_1d(var=self.SummaryFrame[prop_out],
                                            values=self.SummaryFrame[subvol],
                                            statistic='sum',
                                            bin_breaks=bin_breaks,
                                        )
        else:
            # Calculate 1-dimensional grouped statistic from SummaryFrame:
            result_out = binned_group_statistic_1d(var=self.SummaryFrame[prop_out],
                                                values=self.SummaryFrame[subvol],
                                                groups=self.SummaryFrame[group],
                                                statistic='sum',
                                                bin_breaks=bin_breaks,
                                                )
       
        # ---------------------------------------------
        # Calculating Lagrangian Overturning Functions.
        # ---------------------------------------------
        # Rename property dimension names & replace NaN -> 0 for arithmetic:
        result_in = result_in.rename({prop_in:prop}).fillna(value=0)
        result_out = result_out.rename({prop_out:prop}).fillna(value=0)
        # Rename the volume transport variables:
        result_in.name = 'subvol_in'
        result_out.name = 'subvol_out'

        # Join inflow and outflow volume transport DataArrays:
        result_net = xr.merge([result_in, result_out])
        result_net['subvol'] = result_net['subvol_in'] - result_net['subvol_out']

        # Calculate accumulative sum of net volume transport:
        # Case 1. Accumulate with increasing property values:
        if direction == '+1':
            # Accumlate along property dimension:
            result_lof = result_net['subvol'].cumsum(dim=prop, skipna=True)
        # Case 2. Accumulate with decreasing property values:
        elif direction == '-1':
            # Reverse DataArray along property dimension:
            result_net = result_net.reindex({prop:list(reversed(result_net[prop]))})
            # Accumlate along property dimension:
            result_lof = result_net['subvol'].cumsum(dim=prop, skipna=True)

        # ----------------------------------------
        # Adding LOF statistic to summary DataSet.
        # ----------------------------------------
        # Add result_lof DataArray to DataSet as named variable:
        if alias is None:
            self.SummaryArray['LOF_'+prop] = result_lof
        else:
            self.SummaryArray[alias] = result_lof

        # Return TrajStore object with updated TrajFrame & SummaryFrame.
        return TrajStore(traj_source=self.TrajFrame, summary_source=self.SummaryFrame, summary_array=self.SummaryArray)

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
        return TrajStore(traj_source=self.TrajFrame, summary_source=summary_data)

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
        return TrajStore(traj_source=self.TrajFrame, summary_source=summary_data)

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
        return TrajStore(traj_source=self.TrajFrame, summary_source=summary_data)

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
        return TrajStore(traj_source=self.TrajFrame, summary_source=summary_data)

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
        return TrajStore(traj_source=self.TrajFrame, summary_source=summary_data)

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
        return TrajStore(traj_source=self.TrajFrame, summary_source=summary_data)

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
        return TrajStore(traj_source=self.TrajFrame, summary_source=summary_data)

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
        return TrajStore(traj_source=self.TrajFrame, summary_source=summary_data)

##############################################################################
# Define add_variable() method.

    def add_variable(self, name, values, summary_var=False):
        """
        Adds a new variable to the existing TrajStore object.

        The variable data must be provided as 1-dimensional ndarray
        with the variable attributes provided as a string.

        Parameters
        ----------
        self : TrajStore
            TrajStore object passed from TrajStore class method.
        values : list
            values of new variable to be added to the TrajStore.
        name : str
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
        if isinstance(values, list) is False:
            raise TypeError("inavlid type - values must be specified as a list")
        if isinstance(name, str) is False:
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
            summary_data = self.SummaryFrame.with_columns(pl.Series(name=name, values=values))
            # Keep existing TrajFrame.
            trajectory_data = self.TrajFrame
        elif summary_var is False:
            # Append new column variable to TrajFrame.
            trajectory_data = self.TrajFrame.with_columns(pl.Series(name=name, values=values))
            # Keep existing SummaryFrame.
            summary_data = self.SummaryFrame

        # Return TrajStore with updated TrajFrame or SummaryFrame.
        return TrajStore(traj_source=trajectory_data, summary_source=summary_data)
