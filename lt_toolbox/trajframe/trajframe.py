##############################################################################
# trajframe.py
#
# Description:
# Defines the TrajFrame Class from a DataFrame or LazyFrame containing atmosphere
# ocean parcel trajectories (and accompanying tracers) stored in a tabular format.
#
# Last Edited:
# 2023/12/09
#
# Last Edited By:
# Ollie Tooth
#
# Contact:
# oliver.tooth@env-res.ox.ac.uk
#
##############################################################################
# Importing relevant packages.

import os
import numpy as np
import polars as pl
import xarray as xr
import plotly.express as px

# Importing utility functions
from .utils.filter_frame_utils import filter_traj_polygon, filter_traj, filter_summary
from .utils.compute_frame_utils import binned_statistic_1d, binned_statistic_2d, binned_group_statistic_1d, binned_group_statistic_2d, binned_lazy_group_statistic_1d, binned_lazy_group_statistic_2d
from .utils.interpolate_frame_utils import interpolation_1d, interpolation_2d

##############################################################################
# Define TrajFrame Class.


class TrajFrame:

    def __init__(self, source:pl.DataFrame | pl.LazyFrame, condense=False, rename_cols=None, summary_source=None):
        """
        Create a TrajFrame from a Polars DataFrame or LazyFrame.

        Parameters
        ----------
        source: DataFrame | LazyFrame
            Lagrangian trajectories to be stored in TrajFrame. Each trajectory's
            data should be contained in a single row with the positions and
            properties recorded along-stream stored in columns of Polars list
            dtype and can be of variable length.
        condense: bool
            Transform DataFrame or LazyFrame from long-format to condensed
            format where data is stored in list columns.
        rename_cols : dict
            Rename columns variables using key value pairs that map from old
            name to new name.
        summary_source : DataSet
            DataSet storing summary statistics in the form of n-dimensional 
            DataArrays generated from Lagrangian trajectory data contained in
            the TrajFrame.

        Returns
        --------
        TrajFrame object
            Complete trajectories, including all column variables contained
            in .data attribute. Summary statistics stored as n-dimensional
            arrays in .summary_data.

        Examples
        --------
        Creating TrajFrame object, trajectories, with example_trajectories.csv file in eager mode.

        >>> filename = 'example_trajectories.csv'
        >>> data = pl.read_csv(filename)
        >>> trajectories = TrajFrame(source=data)

        Creating TrajFrame object, traj, with multiple parquet files in lazy mode.

        >>> filenames = [ 'example_trajectories1.parquet', 'example_trajectories2.parquet']
        >>> data = pl.concat([pl.scan_csv(file) for file in filenames])
        >>> trajectories = TrajFrame(source=data)

        """
        # -------------------
        # Raising exceptions.
        # -------------------
        # Determine if source is Polars DataFrame:
        is_dataframe = isinstance(source, pl.DataFrame)
        # Determine if source is Polars LazyFrame:
        is_lazyframe = isinstance(source, pl.LazyFrame)

        # Raise error if source is not a Polars DataFrame / LazyFrame:
        if (is_dataframe | is_lazyframe) is False:
            raise TypeError("source must be specified as a Polars DataFrame or LazyFrame")

        # Raise error if summary_source is not an xarray DataSet:
        if summary_source is not None:
            if isinstance(summary_source, xr.Dataset) is False:
                raise TypeError("summary_source must be specified an xarray DataSet")

        # Raise error if attrs is not a dictionary:
        if rename_cols is not None:
            if isinstance(rename_cols, dict) is False:
                raise TypeError("rename columns mapping specified as a dictionary")

        # ----------------------------------------------
        # Constructing Lagrangian TrajFrame from source:
        # ----------------------------------------------
        # Renaming column variables of TrajFrame:
        if rename_cols is not None:
            # Rename column variables using key value pairs that map from
            # old name to new name:
            data = source.rename(rename_cols)
        else:
            data = source

        # Condense DataFrame / LazyFrame into list columns:
        if condense:
            data = (data
                    .group_by(pl.col('id'), maintain_order=True)
                    .agg([
                        pl.all()
                        ])
                    )

        # -----------------------------------------------------
        # Storing input DataFrame/LazyFrame as data attribute.
        # -----------------------------------------------------
        # Defining TrajFrame as input DataFrame or LazyFrame
        # containing one or more rows per trajectory.
        self.data = data
        # Add schema attribute to TrajFrame:
        self.schema = data.schema

        # ---------------------------------------
        # Defining modes as TrajFrame attributes:
        # ---------------------------------------
        # Defining mode of TrajFrame:
        if is_dataframe:
            self.traj_mode = 'eager'
        elif is_lazyframe:
            self.traj_mode = 'lazy'

        # --------------------------------------------
        # Storing summary_data as TrajFrame attribute.
        # --------------------------------------------
        if summary_source is not None:
            # Defining summary_data from specified Dataset.
            self.summary_data = summary_source
        elif summary_source is None:
            # Defining summary_data as empty Dataset.
            self.summary_data = xr.Dataset()

        # --------------------------------------
        # Extracting TrajFrame column variables.
        # --------------------------------------
        # Storing list of column variables contained in TrajFrame.
        self.columns = self.data.columns

        # Raise error if any core column variables are absent from TrajFrame.
        if 'id' not in self.columns:
            raise ValueError("invalid value: core variable missing from TrajFrame: \'id\'")
        if 'time' not in self.columns:
            raise ValueError("invalid value: core variable missing from TrajFrame: \'time\'")

        # ----------------------------------------------------
        # Storing LazyFrame query plan as TrajFrame attribute:
        # ----------------------------------------------------
        # Query plan for TrajFrame:
        if self.traj_mode == 'lazy':
            # Store optimised query plan description as query_plan:
            self.traj_query_plan = self.data.explain()
        else:
            # No query plan when using eager mode:
            self.traj_query_plan = None

##############################################################################
# Define print() method.

    def __str__(self):
        # Return summary of TrajFrame and SummaryFrame
        # stored in TrajFrame object.

        # Construct summary string for TrajFrame:
        if self.traj_mode == 'eager':
            traj_str = f"<TrajFrame object>\n\n----- Trajectory DataFrame -----\nTrajectories: {self.data.shape[0]}\nVariables: {self.columns}\n{self.data.glimpse}\n"
        elif self.traj_mode == 'lazy':
            traj_str = f"<TrajFrame object>\n\n----- Trajectory LazyFrame -----\nSchema: {self.data.schema}\nOptimised Query Plan:\n{self.traj_query_plan}\n"

        return traj_str

##############################################################################
# Define len() method.

    def __len__(self):
        # Return the total number of trajectories in SummaryFrame:
        if self.traj_mode == 'eager':
            len_str = self.data.shape[0]
        elif self.traj_mode == 'lazy':
            len_str = self.data.select(pl.count()).collect(streaming=True)

        return len_str

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
        TrajFrame object
            TrajFrame object is returned with one or more eager
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
        if isinstance(self.data, pl.LazyFrame) is False:
            raise TypeError("invalid type: Lagrangian trajectories are not stored in a LazyFrame")
        if isinstance(streaming, bool) is False:
            raise TypeError("invalid type: streaming must be specified as a boolean")

        # ------------------------------------------------------------
        # Collect LazyFrame(s) as DataFrame following query execution.
        # ------------------------------------------------------------
        # Execute optimised query plan on trajectory frame.
        trajectory_data = self.data.collect(streaming=streaming, **kwargs)

        # Return TrajFrame object including eager DataFrame:
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define use_datetime.

    def use_datetime(self, start_date:str, unit='s', fmt="%Y-%m-%d"):
        """
        Convert time column variable to Datetime format.

        Parameters
        ----------
        start_date : string
            Starting date to use when converting time column
            variable to Datetime.
        unit : string
            Unit time variable is stored as (e.g., 's', 'd', 'w' etc.).
            Default is seconds, 's'.
        fmt : string
            Datetime format of specified start data. Default
            format is YYYY-MM-DD.

        Returns
        --------
        TrajFrame object
            Original TrajFrame object is returned with transformed
            time column variable Series containing datetimes.

        Examples
        --------
        Convert time in seconds in TrajFrame object to datetime with 
        start date '2000-01-01 using default Datetime format.

        >>> trajectories.use_datetime(start_date='2000-01-01')
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(start_date, str) is False:
            raise TypeError("invalid type: start_date must be specified as a string")
        if isinstance(unit, str) is False:
            raise TypeError("invalid type: unit must be specified as a string")
        if unit not in ['w', 'd', 'h', 'm', 's']:
            raise ValueError("invalid value: unit must be specified as one of \'s\', \'m\', \'h\', \'D\', \'W\'")
        if isinstance(self.data.schema['time'], pl.Datetime) is True:
            raise TypeError("invalid type: time already exists with dtype = \'Datetime\'")

        # ------------------------------------------------
        # Convert time to Datetime format with start_date.
        # ------------------------------------------------
        # Define all time parameter as zero initially:
        dt_dict = {'s':0, 'm':0, 'h':0, 'd':0, 'w':0}

        # Iterate over available durations to find unit:
        for key in dt_dict:
            if key == unit:
                # Modify value for specified unit:
                dt_dict[key] =+1

        # Define time duration increment from specified unit:
        dt = pl.duration(weeks=dt_dict['w'], days=dt_dict['d'], hours=dt_dict['h'], minutes=dt_dict['m'], seconds=dt_dict['s'])

        # Replacing time column with datetime values:
        trajectory_data = (self.data
                           .with_columns(
                           pl.col('time').list.eval(pl.element()*dt + pl.lit(start_date).str.strptime(pl.Datetime, format=fmt))
                           .cast(pl.List(pl.Datetime)))
                           )

        # Return TrajFrame object with updated trajectory data.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define transform_trajectory_coords() method.

    def transform_trajectory_coords(self, lon:np.ndarray, lat:np.ndarray, depth:np.ndarray):
        """
        Transform Lagrangian trajectories from model grid coordinates {i,j,k}.
        to geographical coordinates {lon, lat, depth}.

        Lagrangian trajectory positions are (bi-)linearly interpolated from
        the specified ocean general circulation model grid using xarray.

        Parameters
        ----------
        self : TrajFrame object
            TrajFrame object containing Lagrangian trajectories in model coords
            {i, j, k}.
        lon : ndarray
            Longitudes associated with the center of each model grid cell.
        lat : ndarray
            Latitudes associated with the center of each model grid cell.
        depth : ndarray
            Depths associated with model vertical grid levels.

        Returns
        -------
        TrajFrame object
            TrajFrame containing Lagrangian trajectories in geographical
            coords {lon, lat, depth}.

        Examples
        --------
        Transforming Lagrangian trajectories with positions referenced to model
        coordinate system {x, y, z} to geographical coordinates {lon, lat, depth}
        using the ocean general circulation horizontal and vertical model grids.
        Here, we show a simple example for the Nucleus for European Modelling
        of the Ocean ORCA C-grid:

        >>> lon_mdl = ds_grid.nav_lon.values
        >>> lat_mdl = ds_grid.nav_lat.values
        >>> depth_mdl = ds_grid.nav_lev.values
        >>> trajectories.transform_trajectory_coords(lon=lon_mdl, lat=lat_mdl, depth=depth_mdl)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(lon, np.ndarray) is False:
            raise TypeError("invalid type: longitude array must be specified as an ndarray")

        if isinstance(lat, np.ndarray) is False:
            raise TypeError("invalid type: latitude array must be specified as an ndarray")

        if isinstance(depth, np.ndarray) is False:
            raise TypeError("invalid type: depth array must be specified as an ndarray")

        # ---------------------------------------------------------
        # Transforming Lagrangian Trajectories stored in TrajFrame.
        # ---------------------------------------------------------
        # Determine column names with List dtype:
        list_cols = [col for col in self.columns if self.schema[col] == pl.List]
        # Explode positions from condensed format to long format
        # one observation per row:
        df_exp = self.data.explode(columns=list_cols)

        if self.traj_mode == 'eager':
            # Apply coordinate transformation:
            df_exp = (df_exp
                      .pipe(interpolation_2d, fields=[lon, lat], dims=['x', 'y'], aliases=['x', 'y'])
                      .pipe(interpolation_1d, field=depth, dim='z', alias='z')
                      )

        elif self.traj_mode == 'lazy':
            # Apply coordinate transformation:
            df_exp = (df_exp
                .map_batches(lambda df : interpolation_2d(df, fields=[lon, lat], dims=['x', 'y'], aliases=['x', 'y']),
                            streamable=True
                            )
                .map_batches(lambda df : interpolation_1d(df, field=depth, dim='z', alias='z'),
                            streamable=True
                            )
                )

        # Return output from exploded to Polars list dtypes:
        df_exp = (df_exp
                    .group_by(by=pl.col('id'), maintain_order=True)
                    .agg([
                        pl.col(list_cols),
                        ])
                    )
        # Update the original DataFrame / LazyFrame positions:
        trajectory_data = self.data.update(df_exp, on='id', how='inner')
        # Rename TrajFrame position columns to geographic coords:
        trajectory_data = trajectory_data.rename({'x':'lon', 'y':'lat', 'z':'depth'})

        # Return TrajFrame object with updated trajectory data:
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define filter() method.

    def filter(self, expr:str | pl.Expr, drop=False):
        """
        Filter trajectories using conditional on a single column variable
        specified with a string or Polars expression.

        Filtering returns a reduced TrajectoryFrame where only the
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
        expr : string | Expression
            String expression of the form "{variable} {operator} {value}",
            where {variable} represents the column variable contained in
            the TrajFrame used to filter trajectories, {operator}
            represents one of the six standard comparison operators and
            {value} represents the value with which to compare the {variable}
            to. Users can alternatively specify a Polars expression.
        drop : boolean
            Indcates if fitered trajectories should be retained in the
            new TrajFrame (False) or instead dropped from the
            existing TrajFrame (True).

        Returns
        -------
        TrajFrame object
            Complete TrajFrame, including the complete Lagrangian trajectories
            which meet (do not meet) the specified filter condition.

        Examples
        --------
        Filtering all trajectories where latitude is greater than 50 N.

        >>> trajectories.filter(expr='lat > 50', drop=False)

        Filtering trajectory observations between two dates using Datetime strings.

        >>> trajectories.filter('time > 2000-01-01').filter('time <= 2000-12-31')
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(expr, str):
            # Split string expression into three arguments seperated by spaces
            expr_split = expr.split(sep=' ')
            # Defining arguments from string expression:
            variable = expr_split[0]
            operator = expr_split[1]
            value = expr_split[2]

            if len(expr_split) != 3:
                raise ValueError("string expression contains too many arguments. Use format \'var op val\' to compare column variable to value.")

            # Defining list of standard operators.
            operator_list = ['==', '!=', '<', '>', '<=', '>=']

            if isinstance(variable, str) is False:
                raise TypeError("inavalid type - variable must be specified as a string")

            if isinstance(drop, bool) is False:
                raise TypeError("drop must be specified as a boolean")

            if operator not in operator_list:
                raise ValueError("unknown comparison operator specified: \'" + operator + "\'. Use one of the standard Python comparison operators: ==, !=, <, >, <=, >=")
        else:
            if isinstance(expr, pl.Expr) is False:
                raise TypeError('invalid type - expr must be specified as either a str or polars expression')

        # ----------------------------------------------
        # Applying specified filter to Trajectory Frame.
        # ----------------------------------------------
        if isinstance(expr, str):
            # Determine dtype of filter column values:
            value_dtype = self.data.schema[variable]
            if value_dtype == pl.List:
                # Determine inner dtype of List:
                inner_value_dtype = value_dtype.inner
                # Filter List variable using specified expression:
                trajectory_data = filter_traj(df=self.data,
                                            variable=variable,
                                            operator=operator,
                                            value=value,
                                            value_dtype=inner_value_dtype,
                                            drop=drop
                                            )
            else:
                # Filter non-List variable using specified expression:
                trajectory_data = filter_summary(df=self.data,
                                                variable=variable,
                                                operator=operator,
                                                value=value,
                                                value_dtype=value_dtype,
                                                drop=drop
                                                )
        else:
            # Filter TrajFrame using specified polars expression:
            if drop:
                trajectory_data = self.data.filter(~expr)
            else:
                trajectory_data = self.data.filter(expr)

        # Return TrajFrame object with updated trajectory data.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

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
            new TrajFrame (False) or instead dropped from the
            existing TrajFrame (True).

        Returns
        -------
        TrajFrame object
            Complete TrajFrame, including the complete Lagrangian trajectories
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
        # Determine column names with List dtype:
        list_cols = [col for col in self.columns if self.schema[col] == pl.List]
        # Explode positions from condensed format to long format
        # one observation per row:
        df_exp = self.data.explode(columns=list_cols)

        if self.traj_mode == 'eager':
            # Apply polygon filter:
            df_exp = df_exp.pipe(filter_traj_polygon,
                                 xy_vars=xy_vars,
                                 x_poly=x_poly,
                                 y_poly=y_poly,
                                 drop=drop,
                                 )

        elif self.traj_mode == 'lazy':
            # Apply polygon filter:
            df_exp = (df_exp
                      .map_batches(lambda data : filter_traj_polygon(data, xy_vars=xy_vars, x_poly=x_poly, y_poly=y_poly, drop=drop), streamable=True)
                      )

        # Return output from exploded to Polars list dtypes:
        df_exp = (df_exp
                    .group_by(by=pl.col('id'), maintain_order=True)
                    .agg([
                        pl.col(list_cols).explode(),
                        ])
                    )
        # Update the original DataFrame / LazyFrame using
        # filtered IDs:
        trajectory_data = self.data.update(df_exp, on='id', how='inner')

        # Return TrajFrame object with updated trajectory data.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define filter_isin() method.

    def filter_isin(self, var:str, values:list | pl.Series, drop=False):
        """
        Filter trajectories with at least one variable observation
        in a given sequence of values.

        Filtering returns the complete trajectories of particles
        where one or more observations of the given variable are found
        in the given list or Series of values.

        Parameters
        ----------
        var : str
            Name of variable contained in TrajFrame object.
        values : list | Series
            Values of variables used to filter trajectories.
        drop : boolean
            Determines if fitered trajectories should be returned as a
            new TrajFrame (False) or instead dropped from the
            existing TrajFrame (True).

        Returns
        -------
        TrajFrame object
            Complete TrajFrame, including the complete Lagrangian trajectories
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

        # ---------------------------------------
        # Applying specified filter to TrajFrame.
        # ---------------------------------------
        if self.data.schema[var] == pl.List:
            # Filter list variable elementwise using specified expression:
            if drop is True:
                trajectory_data = self.data.filter(~(pl.col(var).list.eval(pl.element().is_in(values))).list.any())
            else:
                trajectory_data = self.data.filter(pl.col(var).list.eval(pl.element().is_in(values)).list.any())

        else:
            # Filter non-list variable using specified expression:
            if drop is True:
                trajectory_data = self.data.filter(~(pl.col(var).is_in(values)))
            else:
                trajectory_data = self.data.filter(pl.col(var).is_in(values))

        # Return TrajFrame object with updated trajectory data.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

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
        self : TrajFrame object
            TrajFrame object passed from TrajFrame class method.
        cum_dist : logical
            Compute the cumulative distance travelled by each particle -
            default is False.
        unit : string
            Unit for distance travelled output - default is 'km' -
            alternative option 'm'.

        Returns
        -------
        TrajFrame object.
        Original TrajFrame object is returned with new column variable
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

        # ------------------------------------------------
        # Computing haversine distance along trajectories.
        # ------------------------------------------------
        # Defining radius of the Earth, re (m), as volumetric mean radius from NASA.
        # See: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
        re = 6371000
        # Define conversion factor meters to kilometers:
        m_to_km = 1 / 1E3

        # Explode ID and haversine distance computed from expr into
        # long format:
        df_exp = (self.data.select(
                    id = pl.col('id').repeat_by(pl.col('lon').list.len()).explode(),
                    dist = 2*re*((pl.col('lat').list.diff().explode().radians()*0.5).sin()**2
                                +pl.col('lat').explode().radians().cos()
                                *pl.col('lat').explode().radians().cos()
                                *(pl.col('lon').list.diff().explode().radians()*0.5).sin()**2
                                ).sqrt().arcsin().fill_null(value=0),
                    ))

        # Calculate accumulated distance along trajectories:
        if cum_dist:
            # Apply unit conversion:
            if unit == 'km':
                df_exp = (df_exp
                            .group_by(by=pl.col('id'), maintain_order=True)
                            .agg([
                                (pl.col('dist')*m_to_km).explode().cumsum(),
                                ])
                            )
            else:
                df_exp = (df_exp
                            .group_by(by=pl.col('id'), maintain_order=True)
                            .agg([
                                pl.col('dist').explode().cumsum(),
                                ])
                            )
        else:
            if unit == 'km':
                df_exp = (df_exp
                            .group_by(by=pl.col('id'), maintain_order=True)
                            .agg([
                                (pl.col('dist')*m_to_km).explode(),
                                ])
                            )
            else:
                df_exp = (df_exp
                            .group_by(by=pl.col('id'), maintain_order=True)
                            .agg([
                                pl.col('dist').explode(),
                                ])
                            )

        # Update TrajFrame with new distance column:
        trajectory_data = (self.data
                           .with_columns(dist=pl.lit(value=0, dtype=pl.List))
                           .update(df_exp, on='id', how='inner')
                           )

        # Return TrajFrame object with updated trajectory data.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define compute_grouped_expr() method.

    def compute_grouped_expr(self, group:str, expr:pl.Expr, alias:str):
        """
        Compute polars expression over groups. 

        Parameters
        ----------
        group : str
            Name of column variable to group according to unique values. 
            The expression will be computed for each group member.
        expr : Expression
            Compute expression to aggregate each group. 
        alias : str
            Name of output statistic from computing grouped expression.

        Returns
        -------
        TrajFrame
            Original TrajFrame is returned with the computed grouped
            expression included in the summary_data where the groups
            are given as the coordinate dimension.

        Examples
        --------
        Calculating summed volume transport of Lagrangian trajectories
        grouped by start date column variable.

        >>> expr = pl.col('vol').sum()
        >>> trajectories.compute_grouped_expr(group='start_date', expr=expr)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(group, str) is False:
            raise TypeError('invalid type - group must be specfified as a string')
        if isinstance(expr, pl.Expr) is False:
            raise TypeError('invalid type - expr must be specified as a polars expression')

        # -----------------------------
        # Calculate grouped expression:
        # -----------------------------
        grouped_data = (self.data
                        .groupby(by=group, maintain_order=True)
                        .agg(expr.alias(alias))
                        )

        # Collect result if input df is a LazyFrame:
        if isinstance(self.data, pl.LazyFrame):
            grouped_data = grouped_data.collect(streaming=True)

        # ---------------------------------------------
        # Adding grouped expression to summary DataSet.
        # ---------------------------------------------
        # Construct xarray DataArray from polars Series:
        result_array = xr.DataArray(data=grouped_data[alias].to_numpy(),
                                    dims=[group],
                                    coords={
                                        group:([group], grouped_data[group].to_numpy())
                                    },
                                    name=alias
                                    )

        # Add DataArray to summary DataSet:
        self.summary_data[alias] = result_array

        # Return TrajFrame object with updated trajectory data.
        return TrajFrame(source=self.data, summary_source=self.summary_data)

##############################################################################
# Define compute_binned_statistic_1d() method.

    def compute_binned_statistic_1d(self, var:str, values:str, statistic:str, bin_breaks:list, alias=None, group=None, summary_var=False):
        """
        Compute a 1-dimensional binned statistic using variables stored in
        a TrajFrame.

        This is a generalization of a histogram function. A histogram divides
        the chosen column varaible into bins, and returns the count of the number
        of points in each bin. This function allows the computation of the sum,
        mean, median, or other statistic of the values within each bin.

        Parameters
        ----------
        var : str
            Name of column variable whose values will binned.

        values : str
            Name of the column variable on which the statistic will be computed.

        group : str
            Name of column variable to group according to unique values using group_by()
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
        TrajFrame
            Original TrajFrame is returned with the 1-dimensional binned
            statistic included in the summary_data where the mid-points
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
        # Determine column names with List dtype:
        list_cols = [col for col in self.data.columns if self.data.schema[col] == pl.List]
        # Explode positions from condensed format to long format
        # (one observation per row) if any inputs are lists:
        if len(set([var, values, group]) - set(list_cols)) == 3:
            df_exp = self.data
        else:
            df_exp = self.data.explode(columns=list_cols)

        if group is None:
            # Calculate 1-dimensional statistic from Data/LazyFrame:
            result = binned_statistic_1d(df=df_exp,
                                         var=var,
                                         values=values,
                                         statistic=statistic,
                                         bin_breaks=bin_breaks,
                                         )
        else:
            if self.traj_mode == 'lazy':
                # Calculate 1-dimensional grouped statistic from LazyFrame:
                result = binned_lazy_group_statistic_1d(ldf=df_exp,
                                                        var=var,
                                                        values=values,
                                                        groups=group,
                                                        statistic=statistic,
                                                        bin_breaks=bin_breaks,
                                                        )
            else:
                # Calculate 1-dimensional grouped statistic from DataFrame:
                result = binned_group_statistic_1d(df=df_exp,
                                                    var=var,
                                                    values=values,
                                                    groups=group,
                                                    statistic=statistic,
                                                    bin_breaks=bin_breaks,
                                                    )

        # ----------------------------------------
        # Adding 1-D statistic to summary DataSet.
        # ----------------------------------------
        # Add result DataArray to DataSet as named variable:
        if alias is None:
            self.summary_data[result.name] = result
        else:
            self.summary_data[alias] = result

        # Return TrajFrame object with updated TrajFrame & SummaryFrame.
        return TrajFrame(source=self.data, summary_source=self.summary_data)

##############################################################################
# Define compute_binned_statistic_2d() method.


    def compute_binned_statistic_2d(self, var_x:str, var_y:str, values:str, statistic:str, bin_breaks:list, alias=None, group=None, summary_var=False):
        """
        Compute a 2-dimensional binned statistic using the variables stored
        in a TrajFrame.

        This is a generalization of a 2-D histogram function. A 2-D histogram divides
        the chosen column varaibles into bins, and returns the count of the number
        of points in each bin. This function allows the computation of the sum,
        mean, median, or other statistic of the values within each bin.

        Parameters
        ----------
        var_x : str
            Name of variable whose values to be binned along the first dimension.

        var_y : str
            Name of variable whose values to be binned along the second dimension.

        values : str
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
        # Determine column names with List dtype:
        list_cols = [col for col in self.data.columns if self.data.schema[col] == pl.List]
        # Explode positions from condensed format to long format
        # (one observation per row) if any inputs are lists:
        if len(set([var_x, var_y, values, group]) - set(list_cols)) == 4:
            df_exp = self.data
        else:
            df_exp = self.data.explode(columns=list_cols)

        if group is None:
            # Calculate 2-dimensional statistic from Data/LazyFrame:
            result = binned_statistic_2d(df=df_exp,
                                         var_x=var_x,
                                         var_y=var_y,
                                         values=values,
                                         statistic=statistic,
                                         bin_breaks=bin_breaks,
                                         )
        else:
            if self.traj_mode == 'lazy':
                # Calculate 2-dimensional grouped statistic from LazyFrame:
                result = binned_lazy_group_statistic_2d(ldf=df_exp,
                                                        var_x=var_x,
                                                        var_y=var_y,
                                                        values=values,
                                                        groups=group,
                                                        statistic=statistic,
                                                        bin_breaks=bin_breaks,
                                                        )
            else:
                # Calculate 2-dimensional grouped statistic from DataFrame:
                result = binned_group_statistic_2d(df=df_exp,
                                                   var_x=var_x,
                                                   var_y=var_y,
                                                   values=values,
                                                   groups=group,
                                                   statistic=statistic,
                                                   bin_breaks=bin_breaks,
                                                   )

        # ----------------------------------------
        # Adding 2-D statistic to summary DataSet.
        # ----------------------------------------
        # Add result DataArray to DataSet as named variable:
        if alias is None:
            self.summary_data[result.name] = result
        else:
            self.summary_data[alias] = result

        # Return TrajFrame object with updated summary_data.
        return TrajFrame(source=self.data, summary_source=self.summary_data)

##############################################################################
# Define compute_property_lof() method.

    def compute_property_lof(self, subvol:str, prop_in:str, prop_out:str, bin_breaks:list, alias=None, direction='+1', group=None):
        """
        Compute Lagrangian Overturning Function in discrete property-space.

        The net volume transport distribution in the chosen property
        space is accumulated in specified direction.

        Parameters
        ----------
        subvol : str
            Name of the variable storing water parcel volume transport.

        prop_in : str
            Name of the inflow property variable to bin volume transports.

        prop_out : str
            Name of the outflow property variable to bin volume transports.

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
        TrajFrame
            Original TrajFrame is returned with Lagrangian overturning
            functions included in the summary_data where the mid-points
            of the specified bins are given as the coordinate dimension.
        """
        # -----------------
        # Raise exceptions.
        # -----------------
        if isinstance(bin_breaks, list) is False:
            raise TypeError('invalid type - bin_breaks must be specified as a list')
        if isinstance(self.schema[subvol], pl.List):
            raise TypeError(f'invalid type - {subvol} cannot be a Polars Lists dtype')
        if isinstance(self.schema[prop_in], pl.List):
            raise TypeError(f'invalid type - {prop_in} cannot be a Polars Lists dtype')
        if isinstance(self.schema[prop_out], pl.List):
            raise TypeError(f'invalid type - {prop_out} cannot be a Polars Lists dtype')
        if alias is not None:
            if isinstance(alias, str) is False:
                raise TypeError('invalid type - alias must be specified as a string')

        if subvol not in self.columns:
            raise ValueError(f'invalid variable - {subvol} is not contained in TrajFrame')
        if prop_in not in self.columns:
            raise ValueError(f'invalid variable - {prop_in} is not contained in TrajFrame')
        if prop_out not in self.columns:
            raise ValueError(f'invalid variable - {prop_out} is not contained in TrajFrame')
        if group is not None:
            if group not in self.columns:
                raise ValueError(f'invalid variable - {group} is not contained in TrajFrame')

        # Determine common prefix of property variable:
        prop = os.path.commonprefix([prop_in, prop_out])
        prop = prop.replace("_", "")

        # ---------------------------------------------------
        # Calculating 1-D binned statistic in property-space.
        # ---------------------------------------------------
        # Binning Volume Transport according to Inflow Properties:
        if group is None:
            # Calculate 1-dimensional inflow statistic from TrajFrame:
            result_in = binned_statistic_1d(df=self.data,
                                            var=prop_in,
                                            values=subvol,
                                            statistic='sum',
                                            bin_breaks=bin_breaks,
                                            )
        else:
            if self.traj_mode == 'lazy':
                # Calculate 1-dimensional grouped statistic from LazyFrame:
                result_in = binned_lazy_group_statistic_1d(ldf=self.data,
                                                            var=prop_in,
                                                            values=subvol,
                                                            statistic='sum',
                                                            bin_breaks=bin_breaks,
                                                            groups=group,
                                                            )
            else:
                # Calculate 1-dimensional inflow grouped statistic from TrajFrame:
                result_in = binned_group_statistic_1d(df=self.data,
                                                        var=prop_in,
                                                        values=subvol,
                                                        statistic='sum',
                                                        bin_breaks=bin_breaks,
                                                        groups=group,
                                                        )

        # Binning Volume Transport according to Outflow Properties:
        if group is None:
            # Calculate 1-dimensional outflow statistic from TrajFrame:
            result_out = binned_statistic_1d(df=self.data,
                                             var=prop_out,
                                             values=subvol,
                                             statistic='sum',
                                             bin_breaks=bin_breaks,
                                             )
        else:
            if self.traj_mode == 'lazy':
                # Calculate 1-dimensional grouped statistic from LazyFrame:
                result_out = binned_lazy_group_statistic_1d(ldf=self.data,
                                                            var=prop_out,
                                                            values=subvol,
                                                            statistic='sum',
                                                            bin_breaks=bin_breaks,
                                                            groups=group,
                                                            )
            else:
                # Calculate 1-dimensional outflow grouped statistic from TrajFrame:
                result_out = binned_group_statistic_1d(df=self.data,
                                                        var=prop_out,
                                                        values=subvol,
                                                        statistic='sum',
                                                        bin_breaks=bin_breaks,
                                                        groups=group,
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
            self.summary_data['LOF_'+prop] = result_lof
        else:
            self.summary_data[alias] = result_lof

        # Return TrajFrame object with updated summary_data.
        return TrajFrame(source=self.data, summary_source=self.summary_data)

##############################################################################
# Define get_start_time() method.

    def get_start_time(self):
        """
        Returns time when water parcels are seeded (start of trajectory).

        The start time is given in the format of the {time} core variable
        and stored for each trajectory in the SummaryFrame.

        Parameters
        ----------
        self : TrajFrame
            TrajFrame passed from TrajFrame class method.

        Returns
        -------
        TrajFrame
            Original TrajFrame is returned with appended column
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
        trajectory_data = (self.data
                           .with_columns(
                               start_time=pl.col('time').list.first()
                           ))

        # Return TrajFrame object with updated DataFrame.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

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
        self : TrajFrame
            TrajFrame passed from TrajFrame class method.

        Returns
        -------
        TrajFrame
            Original TrajFrame is returned with appended column
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
        trajectory_data = (self.data
                           .with_columns(
                               lon_start=pl.col('lon').list.first(),
                               lat_start=pl.col('lat').list.first(),
                               depth_start=pl.col('depth').list.first(),
                           ))

        # Return TrajFrame object with updated DataFrame.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define get_end_time() method.

    def get_end_time(self):
        """
        Returns time when water parcels are terminated (end of trajectory).

        The end time is given in the format of the {time} core variable
        and stored for each trajectory in the SummaryFrame.

        Parameters
        ----------
        self : TrajFrame
            TrajFrame passed from TrajFrame class method.

        Returns
        -------
        TrajFrame
            Original TrajFrame is returned with appended column
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
        trajectory_data = (self.data
                           .with_columns(
                               end_time=pl.col('time').list.last()
                           ))

        # Return TrajFrame object with updated DataFrame.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

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
        self : TrajFrame
            TrajFrame object passed from TrajFrame class method.

        Returns
        -------
        TrajFrame
            Original TrajFrame is returned with appended column
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
        trajectory_data = (self.data
                           .with_columns(
                               lon_end=pl.col('lon').list.last(),
                               lat_end=pl.col('lat').list.last(),
                               depth_end=pl.col('depth').list.last(),
                           ))
        # Return TrajFrame object with updated DataFrame.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

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
        self : TrajFrame
            TrajFrame passed from TrajFrame class method.

        Returns
        -------
        TrajFrame
            Original TrajFrame is returned with appended column
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
        if isinstance(self.schema["time"].inner, pl.Datetime) is False:
            raise TypeError("times must be stored as polars Datetime dtype")

        # --------------------------------------
        # Calculate duration of each trajectory.
        # --------------------------------------
        trajectory_data = (self.data
                           .with_columns(
                               dt=(pl.col('time').list.last() - pl.col('time').list.first()),
                           ))

        # Return TrajFrame object with updated DataFrame.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define get_values() function.

    def get_values(self, var:str, date:str, alias:str=None, fmt="%Y-%m-%d"):
        """
        Returns the values of a specified column variable at a specified
        date for each trajectory.

        Parameters
        ----------
        self : TrajFrame
            TrajFrame passed from TrajFrame class method.
        var : string
            Name of the column variable in the TrajFrame.
        date : string
            date on which to get values of specified column variable.
        alias : string
            New name of the resulting column variable.
        fmt : string
            Datetime format of specified date. Default
            format is YYYY-MM-DD.

        Returns
        -------
        TrajFrame
            Original TrajFrame is returned with appended column
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
        # Determine column names with List dtype:
        list_cols = [col for col in self.columns if self.schema[col] == pl.List]
        # Explode positions from condensed format to long format
        # one observation per row:
        df_exp = self.data.explode(columns=list_cols)

        # Filtering TrajFrame for specified date and then selecting specified variable.
        df_exp = (df_exp
                    .filter(pl.col("time") == pl.lit(date).str.strptime(pl.Datetime, format=fmt))
                    .select([pl.col('id'), pl.col(var).alias(var + "_i")])
                    )

        # Update TrajFrame with new column:
        trajectory_data = (self.data
                        .with_columns(pl.lit(value=0, dtype=self.schema[var].inner).alias(var+'_i'))
                        .update(df_exp, on='id', how='inner')
                        )

        # Rename new column if alias specified:
        if alias is not None:
            trajectory_data = trajectory_data.rename({var+'_i':alias})

        # Return TrajFrame object with updated DataFrame.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define get_max() function.

    def get_max(self, var):
        """
        Returns maximum value of a specified variable for each trajectory.

        The maximum value of the variable is returned for all trajectories
        as a new column variable in the SummaryFrame.

        Parameters
        ----------
        self : TrajFrame
            TrajFrame passed from TrajFrame class method.
        var : string
            Name of the variable in the TrajFrame.

        Returns
        -------
        TrajFrame
            Original TrajFrame is returned with appended column
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
        trajectory_data = (self.data
                           .with_columns(
                               pl.col(var).list.max().alias(var+'_max'),
                           ))

        # Return TrajFrame object with updated DataFrame.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define get_min() function.

    def get_min(self, var):
        """
        Returns minimum value of a specified variable for each trajectory.

        The minimum value of the variable is returned for all trajectories
        as a new column variable in the SummaryFrame.

        Parameters
        ----------
        self : TrajFrame
            TrajFrame passed from TrajFrame class method.
        var : string
            Name of the variable in the TrajFrame.

        Returns
        -------
        TrajFrame
            Original TrajFrame is returned with appended column
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
        trajectory_data = (self.data
                           .with_columns(
                               pl.col(var).list.min().alias(var+'_min'),
                           ))

        # Return TrajFrame object with updated DataFrame.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define add_variable() method.

    def add_variable(self, name:str, values:list=None, expr:pl.Expr=None):
        """
        Adds a new variable to the existing TrajFrame object.

        The variable data can either be provided using a Polars
        expression or a list of values.

        Parameters
        ----------
        self : TrajFrame
            TrajFrame passed from TrajFrame class method.
        name : str
            New variable name to be added to TrajFrame.
        values : list
            values of new variable to be added to the TrajFrame.
        expr : pl.Expr
            Expression used to determine values of new variable.
            The expression must use only columns contained in the
            TrajFrame. 

        Returns
        -------
        TrajFrame
            Original TrajFrame is returned with the new column variable
            appended to the TrajFrame.

        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if values is not None:
            if isinstance(values, list) is False:
                raise TypeError("inavlid type - values must be specified as a list")
        if isinstance(name, str) is False:
            raise TypeError('invalid type - variable name must be specfied as a string')
        if expr is not None:
            if isinstance(expr, pl.Expr) is False:
                raise TypeError("invalid type - expr must be specified as a Polars Expression")

        # -----------------------------------------------------
        # Returning updated TrajFrame with new column variable.
        # -----------------------------------------------------
        # Define new column variable using Polars Expression:
        if expr is not None:
            trajectory_data = (self.data
                            .with_columns(
                                expr.alias(name)
                            )
                            )
        # Define new column variable using list of values:
        else:
            trajectory_data = (self.data
                            .with_columns(
                                pl.Series(name=name,
                                          values=values
                                          )
                            )
                            )

        # Return TrajFrame object with updated DataFrame.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define plot_trajectories() function.

    def plot_trajectories(self, sample_size=None, kwargs=None):
        """
        Plot Lagrangian trajectories in a plotly
        express interactive figure.

        Parameters
        ----------
        self : TrajFrame
            TrajFrame passed from TrajFrame class method.
        sample_size : integer
            Size of random sample of Lagrangian trajectories
            to plot.
        kwargs : dict
            Additional keyword arguments to be passed to plotly line_geo(),
            when creating the figure.

        Returns
        -------
        plotly.graph_objs._figure.Figure
            Interactive figure of Lagrangian particle trajectories plotted
            as a plotly geographic line plot.

        Examples
        --------
        Plot trajectories of ten randomly sampled Lagrangian particles.

        >>>  trajectories.plot_trajectories(sample_size=10).
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if sample_size is not None:
            if isinstance(sample_size, int) is False:
                raise TypeError("invalid type: sample_size must be specified as an integer")
        if kwargs is not None:
            if isinstance(kwargs, dict) is False:
                raise TypeError("invalid type: kwargs must be specified in a dictionary")
        if 'lon' not in self.columns:
            raise ValueError('invalid value: no column variable lon in TrajFrame')
        if 'lat' not in self.columns:
            raise ValueError('invalid value: no column variable lat in TrajFrame')

        # ---------------------------------------------
        # Creating Plotly Express geographic line plot.
        # ---------------------------------------------
        # Determine column names with List dtype:
        list_cols = [col for col in self.columns if self.schema[col] == pl.List]
        # Explode positions from condensed format to long format
        # one observation per row:
        if sample_size is not None:
            df_exp = self.data.sample(n=sample_size).explode(columns=list_cols)
        else:
            df_exp = self.data.explode(columns=list_cols)

        # Collect exploded result if LazyFrame:
        if isinstance(df_exp, pl.LazyFrame):
            df_exp = df_exp.collect(streaming=True)

        if kwargs is None:
            kwargs = {}
        # Update kwargs if no color variable specified:
        if 'color' not in kwargs.keys():
            kwargs['color'] = 'id'
        # Geographic line plot:
        figure = px.line_geo(data_frame=df_exp,
                             lon='lon',
                             lat='lat',
                             **(kwargs),
                             )

        # Return plotly express interactive figure.
        return figure

##############################################################################
# Define plot_timeseries() function.

    def plot_timeseries(self, var, sample_size=None, kwargs=None):
        """
        Plot timeseries of property sampled along Lagrangian trajectories
        in a plotly express interactive figure.

        Parameters
        ----------
        self : TrajFrame
            TrajFrame passed from TrajFrame class method.
        var : str
            Name of column variable to plot timeseries.
        sample_size : integer
            Size of random sample of Lagrangian trajectories
            to plot.
        kwargs : dict
            Additional keyword arguments to be passed to plotly line(),
            when creating the figure.

        Returns
        -------
        plotly.graph_objs._figure.Figure
            Interactive figure of property timeseries sampled along
            Langrangian trajectories.

        Examples
        --------
        Plot the temperature along ten randomly sampled Lagrangian
        trajectories.

        >>>  trajectories.plot_timeseries(var='temp', sample_size=10).
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if sample_size is not None:
            if isinstance(sample_size, int) is False:
                raise TypeError("invalid type: sample_size must be specified as an integer")
        if kwargs is not None:
            if isinstance(kwargs, dict) is False:
                raise TypeError("invalid type: kwargs must be specified in a dictionary")
        if var not in self.columns:
            raise ValueError(f'invalid value: no column variable {var} in TrajFrame')

        # ---------------------------------------------
        # Creating Plotly Express geographic line plot.
        # ---------------------------------------------
        # Explode positions from condensed format to long format
        # one observation per row:
        if sample_size is not None:
            df_exp = self.data.sample(n=sample_size).explode(columns=['time', var])
        else:
            df_exp = self.data.explode(columns=['time', var])

        # Collect exploded result if LazyFrame:
        if isinstance(df_exp, pl.LazyFrame):
            df_exp = df_exp.collect(streaming=True)

        if kwargs is None:
            kwargs = {}
        # Update kwargs if no color variable specified:
        if 'color' not in kwargs.keys():
            kwargs['color'] = 'id'
        # Timeseries line plot:
        figure = px.line(data_frame=df_exp,
                         x='time',
                         y=var,
                         **(kwargs),
                         )

        # Return plotly express interactive figure.
        return figure
