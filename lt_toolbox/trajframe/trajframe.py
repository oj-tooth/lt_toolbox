##############################################################################
"""
trajframe.py

Description:
Defines the TrajFrame Class from a DataFrame or LazyFrame containing atmosphere
ocean parcel trajectories (and accompanying tracers) stored in a tabular format.
"""
##############################################################################
# Import relevant packages.
import os
import numpy as np
import polars as pl
import xarray as xr
import plotly.express as px

# Import types:
from typing_extensions import Self

# Import utility functions
from .utils.list_ops import (
    EagerListOperations,
    LazyListOperations,
    )
from .utils.filter_frame_utils import filter_traj_polygon, filter_traj, filter_summary
from .utils.compute_frame_utils import (
    binned_statistic_1d, binned_statistic_2d,
    binned_group_statistic_1d,
    binned_group_statistic_2d,
    binned_lazy_group_statistic_1d,
    binned_lazy_group_statistic_2d,
    )
from .utils.interpolate_frame_utils import interpolation_1d, interpolation_2d, interpolation_3d

##############################################################################
# Define TrajFrame Class.

class TrajFrame:
    """
    Create a TrajFrame from Lagrangian trajectories stored in either a
    polars DataFrame or LazyFrame or an xarray DataSet.

    Parameters
    ----------
    source: DataFrame | LazyFrame | DataSet
        Lagrangian trajectories to be stored in TrajFrame.
        Trajectories specified in eager or lazy tabular data formats
        can be stored in long-format or condensed formats.
        Trajectories specified in an xarray DataSet will be transformed
        to a condensed DataFrame before TrajFrame creation. 
    condense: bool, default: False
        Transform DataFrame or LazyFrame from long-format to condensed
        format where data is stored in list columns.
    rename_cols : dict, default: None
        Rename columns variables using key value pairs that map from
        current to new column names.
    summary_source : DataSet, default: None
        DataSet storing summary statistics in the form of n-dimensional 
        DataArrays generated from Lagrangian trajectory data contained in
        the TrajFrame.

    Returns
    --------
    TrajFrame
        Complete trajectories, including all column variables contained
        in .data attribute. Summary statistics stored as n-dimensional
        arrays in .summary_data.

    Examples
    --------
    Creating TrajFrame, trajectories, with example_trajectories.csv file in eager mode.

    >>> filename = 'example_trajectories.csv'
    >>> data = pl.read_csv(filename)
    >>> trajectories = TrajFrame(source=data)

    Creating TrajFrame, trajectories, with multiple parquet files in lazy mode.

    >>> filenames = [ 'example_trajectories1.parquet', 'example_trajectories2.parquet']
    >>> data = pl.concat([pl.scan_csv(file) for file in filenames])
    >>> trajectories = TrajFrame(source=data)

    Creating TrajFrame, trajectories, from a .zarr file with dimensions (traj x obs).
    The water parcel IDs must be stored in a 2-dimensional array, trajectory. When
    creating a TrajFrame from a DataSet, condense is defined as True by default.

    >>> filename = 'example_trajectories.zarr'
    >>> dataset = xr.open_zarr(filename, chunks=None)
    >>> trajectories = TrajFrame(source=dataset, condense=True)
    """

    def __init__(self,
                 source:pl.DataFrame | pl.LazyFrame | xr.Dataset,
                 condense:bool=False,
                 rename_cols:dict | None=None,
                 summary_source:xr.Dataset | None=None
                 ):
        # -------------------
        # Raising exceptions:
        # -------------------
        # Determine if source is Polars DataFrame:
        is_dataframe = isinstance(source, pl.DataFrame)
        # Determine if source is Polars LazyFrame:
        is_lazyframe = isinstance(source, pl.LazyFrame)
        # Determine if source is xarray DataSet:
        is_dataset = isinstance(source, xr.Dataset)

        # Raise error if source is not a Polars DataFrame / LazyFrame:
        if (is_dataframe | is_lazyframe | is_dataset) is False:
            raise TypeError("source must be specified as an xarray DataSet or polars DataFrame or LazyFrame")

        # Raise error if condense is not a boolean:
        if isinstance(condense, bool) is False:
            raise TypeError("condense must be specified as a boolean")

        # Raise error if summary_source is not an xarray DataSet:
        if summary_source is not None:
            if isinstance(summary_source, xr.Dataset) is False:
                raise TypeError("summary_source must be specified an xarray DataSet")

        # Raise error if attrs is not a dictionary:
        if rename_cols is not None:
            if isinstance(rename_cols, dict) is False:
                raise TypeError("rename columns mapping specified as a dictionary")

        # Raise error if trajectory and obs coords not present in xarray DataSet:
        if is_dataset:
            coord_list = list(source.coords)
            if 'trajectory' not in coord_list:
                raise ValueError("\'trajectory\' must be specified as a coordinate in the xarray DataSet")
            if 'obs' not in coord_list:
                raise ValueError("\'obs\' must be specified as a coordinate in the xarray DataSet")

        # ----------------------------------------------
        # Constructing Lagrangian TrajFrame from source:
        # ----------------------------------------------
        if is_dataset:
            # Define condense as True by default:
            condense = True
            # Transforming xarray DataSet to Dask DataFrame:
            df_dask = (source
                       .to_dask_dataframe(dim_order=['trajectory', 'obs'], set_index=False)
                       .dropna()
                       .drop(columns='obs')
                       .rename(columns={'trajectory':'id'})
                       )
            # Defining polars Dataframe from xarray DataSet:
            source = (pl.from_pandas(df_dask.compute()))

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
        if is_dataframe | is_dataset:
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
            raise ValueError("core variable missing from TrajFrame: \'id\'")
        if 'time' not in self.columns:
            raise ValueError("core variable missing from TrajFrame: \'time\'")

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

    def __str__(self) -> str:
        # Return summary of TrajFrame object.

        # Construct summary string for TrajFrame:
        if self.traj_mode == 'eager':
            traj_str = f"<TrajFrame object>\n\n----- Trajectory DataFrame -----\nTrajectories: {self.data.shape[0]}\nVariables: {self.columns}\n{self.data.glimpse}\n"
        elif self.traj_mode == 'lazy':
            traj_str = f"<TrajFrame object>\n\n----- Trajectory LazyFrame -----\nSchema: {self.data.schema}\nOptimised Query Plan:\n{self.traj_query_plan}\n"

        return traj_str  

##############################################################################
# Define repr() method.

    def __repr__(self) -> str:
        # Return summary of TrajFrame object.

        # Construct summary string for TrajFrame:
        if self.traj_mode == 'eager':
            traj_str = f"<TrajFrame object>\n\n----- Trajectory DataFrame -----\nTrajectories: {self.data.shape[0]}\nVariables: {self.columns}\n{self.data.glimpse}\n"
        elif self.traj_mode == 'lazy':
            traj_str = f"<TrajFrame object>\n\n----- Trajectory LazyFrame -----\nSchema: {self.data.schema}\nOptimised Query Plan:\n{self.traj_query_plan}\n"

        return traj_str

##############################################################################
# Define len() method.

    def __len__(self) -> int:
        # Return the total number of trajectories in TrajFrame:
        if self.traj_mode == 'eager':
            len_str = self.data.shape[0]
        elif self.traj_mode == 'lazy':
            len_str = self.data.select(pl.count()).collect(streaming=True)

        return len_str

##############################################################################
# Define collect() method.

    def collect(self,
                streaming:bool=False,
                **kwargs
                ) -> Self:
        """
        Execute LazyFrame optimised query and collect DataFrame.

        Parameters
        ----------
        streaming : bool, default: False
            Run parts of the query concurrently with streaming.
        **kwargs (optional)
            Additional keyword arguments to be passed to Polars collect() function.

        Returns
        --------
        TrajFrame
            TrajFrame is returned as an eager DataFrame following query execution.

        Examples
        --------
        Execute and collect the Lagrangian trajectories resulting from a simple
        filter without implementing streaming:

        >>> trajectories.filter('id < 100').collect(streaming=False)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(self.data, pl.LazyFrame) is False:
            raise TypeError("Lagrangian trajectories are not stored in a LazyFrame")
        if isinstance(streaming, bool) is False:
            raise TypeError("streaming must be specified as a boolean")

        # ------------------------------------------------------------
        # Collect LazyFrame(s) as DataFrame following query execution.
        # ------------------------------------------------------------
        # Execute optimised query plan on trajectory frame.
        trajectory_data = self.data.collect(streaming=streaming, **kwargs)

        # Return TrajFrame object including eager DataFrame:
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define use_datetime.

    def use_datetime(self,
                     start_date:str,
                     unit:str='s',
                     fmt:str="%Y-%m-%d"
                     ) -> Self:
        """
        Convert time column variable to Datetime format.

        Parameters
        ----------
        start_date : str
            Starting date to use when converting time column variable to Datetime.
        unit : str, default: 's'
            Unit time variable is stored as (e.g., 's', 'd', 'w' etc.).
        fmt : str, default: "%Y-%m-%d"
            Datetime format of specified start data. Default format is YYYY-MM-DD.

        Returns
        --------
        TrajFrame
            TrajFrame is returned with transformed time column variable
            containing datetimes.

        Examples
        --------
        Convert time in seconds in TrajFrame to datetime with start date '2000-01-01'
        using default Datetime format.

        >>> trajectories.use_datetime(start_date='2000-01-01')
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(start_date, str) is False:
            raise TypeError("start_date must be specified as a string")
        if isinstance(unit, str) is False:
            raise TypeError("unit must be specified as a string")
        if unit not in ['w', 'd', 'h', 'm', 's']:
            raise ValueError("unit must be specified as one of \'s\', \'m\', \'h\', \'D\', \'W\'")
        if (self.data.schema['time'] == pl.Datetime) | (self.data.schema['time'] == pl.List(pl.Datetime)) is True:
            raise TypeError("time already exists with dtype = \'Datetime\'")

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

    def transform_trajectory_coords(self,
                                    lon:np.ndarray,
                                    lat:np.ndarray,
                                    depth:np.ndarray
                                    ):
        """
        Transform trajectories from model grid coordinates {i,j,k}
        to geographical coordinates {lon, lat, depth}.

        Lagrangian trajectory positions are (bi-)linearly interpolated from
        the specified ocean general circulation model grid.

        Both z and terrain following sigma vertical coordinate systems
        are supported.

        Parameters
        ----------
        lon : ndarray
            Longitudes associated with the center of each model grid cell. 
            This must be specified as a 2-D array with dimensions {j, i}.
        lat : ndarray
            Latitudes associated with the center of each model grid cell.
            This must be specified as a 2-D array with dimensions {j, i}.
        depth : ndarray
            Depths associated with model vertical grid levels.
            This must be as specified as either a 1-D array with dimensions {k}
            or a 3-D array with dimensions {k, j, i}.

        Returns
        -------
        TrajFrame
            TrajFrame containing Lagrangian trajectories in geographical
            coords {lon, lat, depth}.

        Examples
        --------
        Transforming Lagrangian trajectories with positions referenced to model
        coordinate system {x, y, z} to geographical coordinates {lon, lat, depth}
        using the ocean general circulation horizontal and vertical model grids.
        Here, we show a simple example for the Nucleus for European Modelling
        of the Ocean ORCA C-grid with a z-level vertical coordinate system:

        >>> lon_mdl = ds_grid.nav_lon.values
        >>> lat_mdl = ds_grid.nav_lat.values
        >>> depth_mdl = ds_grid.nav_lev.values
        >>> trajectories.transform_trajectory_coords(lon=lon_mdl, lat=lat_mdl, depth=depth_mdl)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(lon, np.ndarray) is False:
            raise TypeError("longitude must be specified as an ndarray")
        if isinstance(lat, np.ndarray) is False:
            raise TypeError("latitude must be specified as an ndarray")
        if isinstance(depth, np.ndarray) is False:
            raise TypeError("depth must be specified as an ndarray")
    
        if (depth.ndim != 1) & (depth.ndim != 3):
            raise ValueError("depth must be specified as either a 1-D or 3-D array")

        # ---------------------------------------------------------
        # Transforming Lagrangian Trajectories stored in TrajFrame.
        # ---------------------------------------------------------
        # Determine number of dimensions for depth array:
        ndim_depth = depth.ndim
        # Determine column names with List dtype:
        list_cols = [col for col in self.columns if self.schema[col] == pl.List]
        # Explode positions from condensed format to long format
        # one observation per row:
        df_exp = self.data.explode(columns=list_cols)

        if self.traj_mode == 'eager':
            if ndim_depth == 1:
                # Apply coordinate transformation with 1D depth array:
                df_exp = (df_exp
                            .pipe(interpolation_2d, fields=[lon, lat], dims=['x', 'y'], aliases=['x', 'y'])
                            .pipe(interpolation_1d, field=depth, dim='z', alias='z')
                            )
            else:
                # Apply coordinate transformation with 3D depth array:
                df_exp = (df_exp
                            .pipe(interpolation_2d, fields=[lon, lat], dims=['x', 'y'], aliases=['x', 'y'])
                            .pipe(interpolation_3d, fields=[depth], dims=['z'], aliases=['z'])
                            )

        elif self.traj_mode == 'lazy':
            if ndim_depth == 1:
                # Apply coordinate transformation with 1D depth array:
                df_exp = (df_exp
                    .map_batches(lambda df : interpolation_2d(df, fields=[lon, lat], dims=['x', 'y'], aliases=['x', 'y']),
                                streamable=True
                                )
                    .map_batches(lambda df : interpolation_1d(df, field=depth, dim='z', alias='z'),
                                streamable=True
                                )
                    )
            else:
                # Apply coordinate transformation with 3D depth array:
                df_exp = (df_exp
                    .map_batches(lambda df : interpolation_2d(df, fields=[lon, lat], dims=['x', 'y'], aliases=['x', 'y']),
                                streamable=True
                                )
                    .map_batches(lambda df : interpolation_3d(df, fields=[depth], dims=['z'], aliases=['z']),
                                streamable=True
                                )
                    )

        # Return output from exploded to Polars list dtypes:
        df_exp = (df_exp
                    .group_by(pl.col('id'), maintain_order=True)
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

    def filter(self,
               expr:str | pl.Expr,
               drop:bool=False
               ) -> Self:
        """
        Filter trajectories using conditional on a single column variable
        specified with a string or Polars expression.

        Filtering returns a reduced TrajFrame where only the complete
        trajectories meeting the specified condition are retained.
        The exception is when users specify drop=True, in which case
        trajectories meeting the specified condition are dropped from the
        TrajFrame.

        When the expression variable is specified as 'time' only the
        observations (obs) meeting the specified date conditions
        are retained for all trajectories.

        Parameters
        ----------
        expr : str | Expression
            String expression of the form "{variable} {operator} {value}",
            where {variable} represents the column variable contained in
            the TrajFrame used to filter trajectories, {operator}
            represents one of the six standard comparison operators and
            {value} represents the value with which to compare the {variable}
            to. Users can alternatively specify a polars expression.
        drop : bool, default: False
            Indcates if fitered trajectories should be retained in the
            new TrajFrame (False) or instead dropped from the
            existing TrajFrame (True).

        Returns
        -------
        TrajFrame
            TrajFrame including the complete Lagrangian trajectories
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
                raise TypeError("variable must be specified as a string")

            if isinstance(drop, bool) is False:
                raise TypeError("drop must be specified as a boolean")

            if operator not in operator_list:
                raise ValueError("unknown comparison operator specified: \'" + operator + "\'. Use one of the standard Python comparison operators: ==, !=, <, >, <=, >=")
        else:
            if isinstance(expr, pl.Expr) is False:
                raise TypeError('expr must be specified as either a string query or polars expression')

        # ---------------------------------------
        # Applying specified filter to TrajFrame.
        # ---------------------------------------
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

    def filter_polygon(self,
                       xy_vars:list,
                       x_poly:list,
                       y_poly:list,
                       drop:bool=False
                       ) -> Self:
        """
        Filter trajectories which intersect a specified polygon.

        Filtering returns the complete trajectories of particles
        which have been inside the boundary of a given polygon at
        any point in their lifetime.

        Parameters
        ----------
        xy_vars : list(str)
            List of x and y coordinate column variable names.
        x_poly : list
            List of x-coordinates representing the boundary of the polygon.
        y_poly : list
            List of y-coordinates representing the boundary of the polygon.
        drop : bool, default: False
            Determines if fitered trajectories should be returned as a
            new TrajFrame (False) or instead dropped from the
            existing TrajFrame (True).

        Returns
        -------
        TrajFrame
            TrajFrame including the complete Lagrangian trajectories
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

    def filter_isin(self,
                    var:str,
                    values:list | pl.Series,
                    drop:bool=False
                    ) -> Self:
        """
        Filter trajectories with at least one variable observation
        in a given sequence of values.

        Filtering returns the complete trajectories of particles
        where one or more observations of the given variable are found
        in the given list or Series of values.

        Parameters
        ----------
        var : str
            Name of variable contained in TrajFrame.
        values : list | Series
            Values of variables used to filter trajectories.
        drop : bool, default: False
            Determines if fitered trajectories should be returned as a
            new TrajFrame (False) or instead dropped from the
            existing TrajFrame (True).

        Returns
        -------
        TrajFrame
            TrajFrame including the complete Lagrangian trajectories
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

    def compute_distance(self,
                         cum_dist:bool=False,
                         unit:str='km'
                         ) -> Self:
        """
        Compute distance travelled by particles along their
        of trajectories.

        Either the distance travelled between particle positions
        or the cumulative distance travelled is computed
        and returned for all trajectories a new variable.
    
        The final element in each list is Null since the distance
        magnitude and direction is undefined at the final position.

        Parameters
        ----------
        cum_dist : bool, default: False
            Compute the cumulative distance travelled by each particle -
            default is False.
        unit : str, default: 'km'
            Unit for distance travelled output. The default is kilometers, 'km',
            alternatively meters can be specified, 'm'.

        Returns
        -------
        TrajFrame.
            TrajFrame is returned with new column variable containing the distance
            travelled by particles along their trajectories.

        Examples
        --------
        Computing distance travelled by particles for all trajectories,
        specifying cumulative distance as False and unit as default 'km'.

        >>> trajectories.compute_distance(cum_dist=False, unit='km')
        """
        # ------------------
        # Raise exceptions.
        # ------------------
        # Define np.array with available options for
        # distance output units.
        unit_options = ['m', 'km']

        # Raising exception when unavailable unit is specified.
        if unit not in unit_options:
            raise ValueError("unit must be specified as either \'m\' or \'km\'")

        # Assigning boolean value to use_km:
        use_km = True if unit == 'km' else False

        # Raising exception if longitude or latitude variables are
        # not included in TrajFrame:
        if ('lon' not in self.columns) | ('lat' not in self.columns):
            raise ValueError("required column variable missing from TrajFrame: \'lon\', \'lat\'")

        # ------------------------------------------------
        # Computing haversine distance along trajectories.
        # ------------------------------------------------
        if self.traj_mode == 'eager':
            trajectory_data = (self.data
                               .eager_list_ops.haversine_dist(cum_dist=cum_dist, use_km=use_km)
                               )
        elif self.traj_mode == 'lazy':
            trajectory_data = (self.data
                               .lazy_list_ops.haversine_dist(cum_dist=cum_dist, use_km=use_km)
                               )

        # Return TrajFrame object with updated trajectory data.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define compute_velocity() method.

    def compute_velocity(self) -> Self:
        """
        Compute magnitude and direction of the velocity vectors describing
        each Lagrangian trajectory.

        Two column variables are returned to the TrajFrame object:
        speed (m/s) and direction (degrees) each with List dtype.

        The final element in each list is Null since the velocity
        magnitude and direction is undefined at the final position.

        Returns
        -------
        TrajFrame.
            TrajFrame is returned with two new column variables: magnitude (speed)
            and the direction (bearing) of the velocity vectors describing each
            Lagrangian trajectory.

        Examples
        --------
        Computing velocity of particles along their Lagrangian trajectories,
        by default the speed is returned with units of the {dist} / {time}
        column variables (e.g., m/s).

        >>> trajectories.compute_velocity()
        """
        # ------------------
        # Raise exceptions.
        # ------------------
        # Raising exception if longitude, latitude, distance variables
        # are not included in TrajFrame:
        if ('lon' not in self.columns) | ('lat' not in self.columns):
            raise ValueError("required variable missing from TrajFrame: \'lon\', \'lat\'")
        if ('dist' not in self.columns):
            raise ValueError("required variable missing from TrajFrame: \'dist\'")

        # -------------------------------------------------
        # Computing velocity along Lagrangian Trajectories.
        # -------------------------------------------------
        # Compute direction (bearing) of velocity vectors:
        if self.traj_mode == 'eager':
            trajectory_data = (self.data
                               .eager_list_ops.velocity_direction()
                               )
        elif self.traj_mode == 'lazy':
            trajectory_data = (self.data
                               .lazy_list_ops.velocity_direction()
                               )

        # Compute magnitude (speed) of velocity vectors:
        if self.traj_mode == 'eager':
            trajectory_data = (trajectory_data.
                               eager_list_ops.velocity_magnitude()
                               )
        elif self.traj_mode == 'lazy':
            trajectory_data = (trajectory_data.
                               lazy_list_ops.velocity_magnitude()
                               )

        # Return TrajFrame object with updated trajectory data.
        return TrajFrame(source=trajectory_data, summary_source=self.summary_data)

##############################################################################
# Define compute_grouped_expr() method.

    def compute_grouped_expr(self,
                             group:str,
                             expr:pl.Expr,
                             alias:str,
                             append:bool=False
                             ) -> Self:
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
        append : bool, default: False
            If set to True, the grouped expression will be appended
            to the existing summary_data, otherwise summary_data will
            be replaced with a new DataSet.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with the computed grouped
            expression included in the summary_data where the groups
            are given as the coordinate dimension.

        Examples
        --------
        Calculating summed volume transport of Lagrangian trajectories
        grouped by the start_date column variable.

        >>> expr = pl.col('vol').sum()
        >>> trajectories.compute_grouped_expr(group='start_date', expr=expr)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(group, str) is False:
            raise TypeError('group must be specfified as a string')
        if isinstance(expr, pl.Expr) is False:
            raise TypeError('expr must be specified as a polars expression')
        if isinstance(alias, str) is False:
            raise TypeError('alias must be specified as a string')
        if isinstance(append, bool) is False:
            raise TypeError('append must be specified as a boolean')

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
        # Append to or replace summary_data with empty DataSet:
        if append is False:
            self.summary_data = xr.Dataset()

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

    def compute_binned_statistic_1d(self,
                                    var:str,
                                    values:str,
                                    statistic:str,
                                    bin_breaks:list,
                                    alias:str | None=None,
                                    group:str | None=None,
                                    append:bool=False
                                    ) -> Self:
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
        statistic: str
            The statistic to compute; empty bins will be assigned a Null value.
            The following statistics are available:
            * 'mean' : compute the mean of values for points within each bin.
            * 'std' : compute the standard deviation within each bin.
            * 'median' : compute the median of values for points within each
            bin.
            * 'count' : compute the count of points within each bin. This is
            identical to an unweighted histogram.
            * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
            * 'min' : compute the minimum of values for points within each bin.
            * 'max' : compute the maximum of values for point within each bin.
        bin_breaks: list
            List of bin edges used in the binning of var variable.
        alias : str, default: None
            New name of output statistics.
        group : str, default: None
            Name of column variable to group according to unique values using group_by()
            method. A 1-dimensional binned statistic will be computed for each
            group member.
        append : bool, default: False
            If set to True, the 1-dimensional binned statistic will be appended
            to the existing summary_data, otherwise summary_data will be replaced
            with a new DataSet.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with the 1-dimensional binned
            statistic included in the summary_data where the mid-points
            of the specified bins are given as the coordinate dimension.

        Examples
        --------
        Calculating the total volume transport of the Lagrangian trajectories
        according to the longitude of their starting positions.

        >>> bin_lon = np.arange(-40, 10, 1).tolist()
        >>> trajectories.compute_binned_statistic_1d(var='start_lon', values='vol', statistic='sum', bin_breaks=bin_lon, alias='vol_transport')
        """
        # -----------------
        # Raise exceptions.
        # -----------------
        if isinstance(bin_breaks, list) is False:
            raise TypeError('bin_breaks must be specified as a list')
        if alias is not None:
            if isinstance(alias, str) is False:
                raise TypeError('alias must be specified as a list')

        if var not in self.columns:
            raise ValueError(f'variable {var} is not contained in TrajFrame')
        if values not in self.columns:
            raise ValueError(f'variable {values} is not contained in TrajFrame')
        if group is not None:
            if group not in self.columns:
                raise ValueError(f'variable {group} is not contained in TrajFrame')
        if isinstance(append, bool) is False:
            raise TypeError('append must be specified as a boolean')

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
        # Append to or replace summary_data with empty DataSet:
        if append is False:
            self.summary_data = xr.Dataset()
        # Add result DataArray to DataSet as named variable:
        if alias is None:
            self.summary_data[result.name] = result
        else:
            self.summary_data[alias] = result

        # Return TrajFrame object with updated TrajFrame & SummaryFrame.
        return TrajFrame(source=self.data, summary_source=self.summary_data)

##############################################################################
# Define compute_binned_statistic_2d() method.

    def compute_binned_statistic_2d(self,
                                    var_x:str,
                                    var_y:str,
                                    values:str,
                                    statistic:str,
                                    bin_breaks:list,
                                    alias:str | None=None,
                                    group:str | None=None,
                                    append:bool=False
                                    ) -> Self:
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
        statistic: str
            The statistic to compute; empty bins will be assigned a Null value.
            The following statistics are available:
            * 'mean' : compute the mean of values for points within each bin.
            * 'std' : compute the standard deviation within each bin.
            * 'median' : compute the median of values for points within each
            bin.
            * 'count' : compute the count of points within each bin. This is
            identical to an unweighted histogram.
            * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
            * 'min' : compute the minimum of values for points within each bin.
            * 'max' : compute the maximum of values for point within each bin.
        bin_breaks: list
            List of lists including bin edges used in the binning of var_x
            and var_y variables.
        alias : str, default: None
            New name of output statistics.
        group : str, default: None
            Name of variable to group according to unique values using group_by()
            method. A 1-dimensional binned statistic will be computed for each
            group member.
        append : bool, default: False
            If set to True, the 2-dimensional binned statistic will be appended
            to the existing summary_data, otherwise summary_data will be replaced
            with a new DataSet.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with the 2-dimensional binned
            statistic included in the summary_data where the mid-points
            of the specified bins are given as the coordinate dimensions.

        Examples
        --------
        Calculating the mean potential temperature recorded along Lagrangian
        trajectories in discrete longitude-latitude bins.

        >>> bin_lon = np.arange(-40, 10, 1).tolist()
        >>> bin_lat = np.arange(30, 60, 1).tolist()
        >>> trajectories.compute_binned_statistic_2d(var_x='lon', var_y='lat', values='potemp', statistic='mean', bin_breaks=[bin_lon, bin_lat], alias='potemp_mean')
        """
        # -----------------
        # Raise exceptions.
        # -----------------
        if isinstance(bin_breaks, list) is False:
            raise TypeError('bin_breaks must be specified as a list')
        if alias is not None:
            if isinstance(alias, str) is False:
                raise TypeError('alias must be specified as a string')

        if var_x not in self.columns:
            raise ValueError(f'variable {var_x} is not contained in TrajFrame')
        if var_y not in self.columns:
            raise ValueError(f'variable {var_y} is not contained in TrajFrame')
        if values not in self.columns:
            raise ValueError(f'variable {values} is not contained in TrajFrame')
        if group is not None:
            if group not in self.columns:
                raise ValueError(f'variable {group} is not contained in TrajFrame')
        if isinstance(append, bool) is False:
            raise TypeError('append must be specified as a boolean')

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
        # Append to or replace summary_data with empty DataSet:
        if append is False:
            self.summary_data = xr.Dataset()
        # Add result DataArray to DataSet as named variable:
        if alias is None:
            self.summary_data[result.name] = result
        else:
            self.summary_data[alias] = result

        # Return TrajFrame object with updated summary_data.
        return TrajFrame(source=self.data, summary_source=self.summary_data)

##############################################################################
# Define compute_property_lof() method.

    def compute_property_lof(self,
                             subvol:str,
                             prop_in:str,
                             prop_out:str,
                             bin_breaks:list,
                             alias:str | None=None,
                             direction:str='+1',
                             group:str | None=None,
                             append:bool=False
                             ) -> Self:
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
        alias: str, default: None
            New name for Lagrangian overturning funtion in property-coordinates.
        direction : str, default: '+1'
            Direction of integration. Two options are available: to integrate from
            the smallest to largest bin use '+1' or to integrate from the largest
            to smallest bin use '-1'.
        group : str, default: None
            Name of variable to group according to unique values using group_by()
            method. A Lagrangian overturning function will be computed for each
            group member.
        append : bool, default: False
            If set to True, the Lagrangian overturning function will be appended
            to the existing summary_data, otherwise summary_data will be replaced
            with a new DataSet.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with Lagrangian overturning
            functions included in the summary_data where the mid-points
            of the specified bins are given as the coordinate dimension.

        Examples
        --------
        Calculating the Lagrangian overturning function in potential density
        space for all trajectories grouped by the {start_date} column variable.

        See Tooth et al. (2023) for further details on the definition of the
        Lagrangian overturning function in property-space and its application
        to quantifying along-stream water mass transformation.

        >>> bin_sigma0 = np.arange(25, 28, 0.01).tolist()
        >>> trajectories.compute_property_lof(subvol='vol', prop_in='sigma0_in', prop_out='sigma0_out', bin_breaks=bin_sigma0, alias='LOF_sigma0', direction='+1', group='start_date')
        """
        # -----------------
        # Raise exceptions.
        # -----------------
        if isinstance(bin_breaks, list) is False:
            raise TypeError('bin_breaks must be specified as a list')
        if isinstance(self.schema[subvol], pl.List):
            raise TypeError(f'variable {subvol} cannot be a polars Lists dtype')
        if isinstance(self.schema[prop_in], pl.List):
            raise TypeError(f'variable {prop_in} cannot be a polars Lists dtype')
        if isinstance(self.schema[prop_out], pl.List):
            raise TypeError(f'variable {prop_out} cannot be a polars Lists dtype')
        if alias is not None:
            if isinstance(alias, str) is False:
                raise TypeError('alias must be specified as a string')

        if subvol not in self.columns:
            raise ValueError(f'variable {subvol} is not contained in TrajFrame')
        if prop_in not in self.columns:
            raise ValueError(f'variable {prop_in} is not contained in TrajFrame')
        if prop_out not in self.columns:
            raise ValueError(f'variable {prop_out} is not contained in TrajFrame')
        if group is not None:
            if group not in self.columns:
                raise ValueError(f'variable {group} is not contained in TrajFrame')
        if isinstance(append, bool) is False:
            raise TypeError('append must be specified as a boolean')

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

        # Calculate accumulative sum of net volume transport:
        # Case 1. Accumulate with increasing property values:
        if direction == '+1':
            # Accumlate along property dimension:
            result_lof = result_net['subvol_in'].cumsum(dim=prop) - result_net['subvol_out'].cumsum(dim=prop)
        # Case 2. Accumulate with decreasing property values:
        elif direction == '-1':
            # Reverse DataArray along property dimension
            # and accumlate along property dimension:
            result_lof = (result_net['subvol_in'].reindex({prop:list(reversed(result_net[prop]))}).cumsum(dim=prop)
                          - result_net['subvol_out'].reindex({prop:list(reversed(result_net[prop]))}).cumsum(dim=prop))

        # ----------------------------------------
        # Adding LOF statistic to summary DataSet.
        # ----------------------------------------
        # Append to or replace summary_data with empty DataSet:
        if append is False:
            self.summary_data = xr.Dataset()
        # Add result_lof DataArray to DataSet as named variable:
        if alias is None:
            self.summary_data['LOF_'+prop] = result_lof
        else:
            self.summary_data[alias] = result_lof

        # Return TrajFrame object with updated summary_data.
        return TrajFrame(source=self.data, summary_source=self.summary_data)

##############################################################################
# Define compute_probability() method.

    def compute_probability(self,
                            bin_res:float,
                            prob_type:str='pos',
                            group:str | None=None,
                            append:bool=False,
                            ) -> Self:
        """
        Compute Lagrangian probability in discrete geographical (longitude,
        latitude) space.
        
        Lagrangian probability is the likelihood that a Lagrangian trajectory
        will enter a given geographical bin at least once (prob_type='traj')
        or the likelihood that a Lagrangian trajectory position will be found
        in a given geographical bin (prob_type='pos').

        Parameters
        ----------
        bin_res : float
            Geographical bin resolution to compute Lagrangian probability.
            The bin resolution should be specified in degrees.
        prob_type : str, default: 'pos'
            Type of Lagrangian probability to compute. Options are 'pos' and
            'traj'. The default is 'pos' which returns the probability that
            a Lagrangian trajectory position is found in any given
            geographical bin. The 'traj' option returns the probability that
            a Lagrangian trajectory will enter any given geographical bin.
        group : str, default: None
            Name of column variable to group according to unique values using
            group_by() before computing Lagrangian probability.
        append : bool, default: False
            If set to True, the probability statistic  will be appended to the
            existing summary_data, otherwise summary_data will be replaced with
            a new DataSet.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with the 2-dimensional Lagrangian probability
            statistic included in the summary_data where the mid-points of the
            latitude-longitude bins are given as the dimensions.

        Examples
        --------
        Calculating the Lagrangian probability of trajectories in discrete (1 x 1)
        degree longitude-latitude bins. The probability is computed using the 'traj'
        option which returns the probability that a Lagrangian trajectory will
        enter any given geographical bin.

        >>> trajectories.compute_probability(bin_res=1, prob_type='traj')
        """
        # -----------------
        # Raise exceptions.
        # -----------------
        if isinstance(bin_res, float) is False:
            raise TypeError('bin_res must be specified as a float')
        if isinstance(prob_type, str) is False:
            raise TypeError('probability type must be specified as a string')
        if group is not None:
            if isinstance(group, str) is False:
                raise TypeError('group must be specified as a string')
            if group not in self.columns:
                raise ValueError(f'variable {group} is not contained in TrajFrame')
        if isinstance(append, bool) is False:
            raise TypeError('append must be specified as a boolean')

        # ---------------------------------
        # Calculating 2-D binned statistic.
        # ---------------------------------
        # Define x, y variables:
        var_x = 'lat'
        var_y = 'lon'
        values = 'lon'
        # Define bin breaks:
        bin_x = np.arange(-90, 90+bin_res, bin_res).tolist()
        bin_y = np.arange(-180, 180+bin_res, bin_res).tolist()
        bin_breaks = [bin_x, bin_y]

        # Define drop duplicates according to probability type:
        if prob_type == 'pos':
            drop_duplicates = False
        elif prob_type == 'traj':
            drop_duplicates = True

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
                                         statistic='count',
                                         bin_breaks=bin_breaks,
                                         drop_duplicates=drop_duplicates,
                                         )
        else:
            if self.traj_mode == 'lazy':
                # Calculate 2-dimensional grouped statistic from LazyFrame:
                result = binned_lazy_group_statistic_2d(ldf=df_exp,
                                                        var_x=var_x,
                                                        var_y=var_y,
                                                        values=values,
                                                        groups=group,
                                                        statistic='count',
                                                        bin_breaks=bin_breaks,
                                                        drop_duplicates=drop_duplicates,
                                                        )
            else:
                # Calculate 2-dimensional grouped statistic from DataFrame:
                result = binned_group_statistic_2d(df=df_exp,
                                                   var_x=var_x,
                                                   var_y=var_y,
                                                   values=values,
                                                   groups=group,
                                                   statistic='count',
                                                   bin_breaks=bin_breaks,
                                                   drop_duplicates=drop_duplicates,
                                                   )

        # ----------------------------------------
        # Adding 2-D statistic to summary DataSet.
        # ----------------------------------------
        # Append to or replace summary_data with empty DataSet:
        if append is False:
            self.summary_data = xr.Dataset()

        if prob_type == 'pos':
            # Calculate Lagrangian probability of positions:
            self.summary_data['probability'] = result / result.sum()

        elif prob_type == 'traj':
            # Calculate number of trajectories:
            if self.traj_mode == 'eager':
                n_traj = self.data['id'].unique().len()
            elif self.traj_mode == 'lazy':
                n_traj = self.data['id'].unique().len().collect(streaming=True)
            # Calculate Lagrangian probability of trajectories:
            self.summary_data['probability'] = result / n_traj

        # Return TrajFrame object with updated summary_data.
        return TrajFrame(source=self.data, summary_source=self.summary_data)

##############################################################################
# Define get_start_time() method.

    def get_start_time(self) -> Self:
        """
        Get time when particles are seeded (start of trajectory).

        The start time is returned for each trajectory in the format of
        the {time} column variable in a new column variable {start_time}.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with appended column variable {start_time}
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

    def get_start_loc(self) -> Self:
        """
        Get locations where particles are seeded (start of trajectory).

        The start locations of each trajectory are added in the form of three
        new column variables: {lon_start}, {lat_start} and {depth_start}.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with appended column variables {lon_start},
            {lat_start}, {depth_start}.

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

    def get_end_time(self) -> Self:
        """
        Get time when particles are terminated (end of trajectory).

        The end time is returned for each trajectory in the format of
        the {time} column variable in a new column variable {end_time}.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with appended column variable {end_time}
            containing the times each trajectory was terminated.

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

    def get_end_loc(self) -> Self:
        """
        Get locations where particles are terminated (end of trajectory).

        The end locations of each trajectory are added in the form of three
        new column variables: {lon_end}, {lat_end} and {depth_end}.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with appended column variables {lon_end},
            {lat_end}, {depth_end}.

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

    def get_duration(self) -> Self:
        """
        Get time taken for particles to be terminated following seeding
        (duration of trajectory).

        The duration of each trajectory is stored in the polars Duration
        dtype format.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with appended column variable {total_time}.

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
            raise TypeError("time must be specified as polars Datetime dtype")

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

    def get_values(self,
                   var:str,
                   date:str,
                   alias:str | None=None,
                   fmt:str="%Y-%m-%d"
                   ) -> Self:
        """
        Get the values of a specified column variable at a given
        date along each trajectory.

        Parameters
        ----------
        var : str
            Name of the column variable in the TrajFrame.
        date : str
            date on which to get values of specified column variable.
        alias : str, default: None
            New name of the resulting column variable.
        fmt : str, default: "%Y-%m-%d"
            Datetime format of specified date. Default
            format is YYYY-MM-DD.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with appended column variable {var}_i
            recording the values of spcified column variable at the given
            date.

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
            raise ValueError(f"variable {var} not contained in TrajFrame")

        if isinstance(date, str) is False:
            raise TypeError("date must be specified as a string in format given to fmt")

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

    def get_max(self,
                var:str
                ) -> Self:
        """
        Get maximum value of a specified variable for each trajectory.

        The maximum value of the variable is returned for all trajectories
        as a new column variable.

        Parameters
        ----------
        var : str
            Name of column variable to find maximum value along each 
            trajectory.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with appended column variable {var}_max
            containing the maximum values along each trajectory.

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
            raise ValueError(f"variable {var} not found in main DataFrame")

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

    def get_min(self,
                var:str
                ) -> Self:
        """
        Get minimum value of a specified variable for each trajectory.

        The minimum value of the variable is returned for all trajectories
        as a new column variable.

        Parameters
        ----------
        var : str
            Name of column variable to find minimum value along each 
            trajectory.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with appended column variable {variable}_min
            containing the minmum values along each trajectory.

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

    def add_variable(self,
                     name:str,
                     values:list | None=None,
                     expr:pl.Expr | None=None,
                     list_expr:bool=False
                     ) -> Self:
        """
        Add a new column variable to the TrajFrame.

        The variable can either be defined using a polars
        expression or a list of values.

        Parameters
        ----------
        name : str
            New column variable name to be added to TrajFrame.
        values : list, default: None
            values of new variable to be added to the TrajFrame.
        expr : Expression, default: None
            Expression used to determine values of new variable.
            The expression must use only columns contained in the
            TrajFrame.
        list_expr : bool, default: False
            Use list expression to determine values of new variable.
            The default value is False.

        Returns
        -------
        TrajFrame
            TrajFrame is returned with the new column variable
            appended to the TrajFrame.

        Examples
        --------
        Add a new column variable {sigma0} to the TrajFrame containing
        the potential density values recorded along particle trajectories.

        Method 1: Using a list of values with an eager TrajFrame, trajectories.
        Here, we first import the Gibbs Seawater Toolbox package, gsw.

        >>> import gsw
        >>> sigma0 = gsw.density.sigma0(SA=trajectories.data['sal'], CT=trajectories.data['temp']).tolist()
        >>> trajectories.add_variable(name='sigma0', values=sigma0).

        Method 2: Using a polars expression with a lazy TrajFrame, trajectories.
        Importantly, this method is much more performant approach to create a new variable
        than using a list of values since the expression is only evaluated when the
        TrajFrame is collected by leveraging the eager/lazy_list_ops extensions of the
        polars API.

        >>> trajectories.add_variable(name='sigma0', expr=gsw.density.sigma0(SA=pl.col('sal'), CT=pl.col('temp')), list_expr=True)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if values is not None:
            if isinstance(values, list) is False:
                raise TypeError("values must be specified as a list")
        if isinstance(name, str) is False:
            raise TypeError('variable name must be specfied as a string')
        if expr is not None:
            if isinstance(expr, pl.Expr) is False:
                raise TypeError("expr must be specified as a Polars Expression")

        # -----------------------------------------------------
        # Returning updated TrajFrame with new column variable.
        # -----------------------------------------------------
        # Define new column variable using polars Expression:
        if expr is not None:
            # Use list expression to determine values of new variable:
            if list_expr:
                if self.traj_mode == 'eager':
                    trajectory_data = (self.data
                                    .eager_list_ops.apply_expr(expr=expr, alias=name)
                                    )
                elif self.traj_mode == 'lazy':
                    trajectory_data = (self.data
                                    .lazy_list_ops.apply_expr(expr=expr, alias=name)
                                    )
            else:
                trajectory_data = (self.data
                                   .with_columns(
                                       expr.alias(name)
                                   ))
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

    def plot_trajectories(self,
                          sample_size:int | None=None,
                          **kwargs
                          ):
        """
        Plot Lagrangian trajectories in a plotly
        express interactive figure.

        Parameters
        ----------
        sample_size : int, default: None
            Size of random sample of Lagrangian trajectories
            to plot.
        **kwargs (optional)
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

        if sample_size is not None:
            if self.traj_mode == 'eager':
                # Sample IDs from TrajFrame and explode positions from condensed
                # format to long format one observation per row:
                df_exp = self.data.sample(n=sample_size).explode(columns=list_cols)
            elif self.traj_mode == 'lazy':
                # Create random sample of IDs from LazyFrame:
                id_series = (self.data
                             .select(
                                 pl.col('id').min().alias('min'),
                                 pl.col('id').max().alias('max'),
                                 )
                            ).collect(streaming=True)
                id_samples = np.random.randint(low=id_series['min'],
                                               high=id_series['max'],
                                               size=sample_size)
                # Sample IDs from TrajFrame and explode columns:
                df_exp = (self.data
                          .filter(pl.col('id').is_in(id_samples))
                          .explode(columns=list_cols)
                          )
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

    def plot_timeseries(self,
                        var:str,
                        sample_size:int | None=None,
                        **kwargs
                        ):
        """
        Plot timeseries of property sampled along Lagrangian trajectories
        in a plotly express interactive figure.

        Parameters
        ----------
        var : str
            Name of column variable to plot timeseries.
        sample_size : int, default: None
            Size of random sample of Lagrangian trajectories
            to plot.
        **kwargs (optional)
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
            if self.traj_mode == 'eager':
                # Sample IDs from TrajFrame and explode variables from condensed
                # format to long format one observation per row:
                df_exp = self.data.sample(n=sample_size).explode(columns=['time', var])
            elif self.traj_mode == 'lazy':
                # Create random sample of IDs from LazyFrame:
                id_series = (self.data
                             .select(
                                 pl.col('id').min().alias('min'),
                                 pl.col('id').max().alias('max'),
                                 )
                            ).collect(streaming=True)
                id_samples = np.random.randint(low=id_series['min'],
                                               high=id_series['max'],
                                               size=sample_size)
                # Sample IDs from TrajFrame and explode columns:
                df_exp = (self.data
                          .filter(pl.col('id').is_in(id_samples))
                          .explode(columns=['time', var])
                          )
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
