##############################################################################
# list_ops.py
#
# Description:
# Defines the list operations polars DataFrame & LazyFrame registered
# namespaces.
#
# Date Created:
# 2023/12/28
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.
import numpy as np
import polars as pl

##############################################################################
# Define DataFrames ListOperations Class.
@pl.api.register_dataframe_namespace("eager_list_ops")
class EagerListOperations:
    """
    List Operations API Extention for polars DataFrames to apply
    custom expressions on List dtype columns.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing positions and propeties of Lagrangian
        trajectories.

    Class Methods
    -------------
    apply_expr(expr:pl.Expr, alias:str) -> pl.DataFrame
        Applies a custom expression on List dtype columns of a DataFrame.
    """
    def __init__(self, df: pl.DataFrame):
        self._df = df

##############################################################################
# Define apply_expr() method.

    def apply_expr(self, expr:pl.Expr, alias:str) -> pl.DataFrame:
        """
        Apply a custom polars expression on List dtype columns of a
        polars DataFrame.

        Parameters
        ----------
        expr : Expr
            Custom polars expression to apply on List dtype columns.
        alias : str
            Name of column variable to store expression results.

        Returns
        -------
        df : DataFrame
            DataFrame containing Lagrangian trajectories and new column
            variable storing the result of the custom expression.
        """
        return (self._df
                .explode(
                    columns=[col for col in self._df.columns if self._df.schema[col] == pl.List]
                    )
                .with_columns(
                    expr.alias(alias)
                    )
                .group_by(by='id', maintain_order=True)
                .agg(pl.all())
                )

##############################################################################
# Define haversine_dist() method.

    def haversine_dist(self, cum_dist:bool=False, use_km:bool=True) -> pl.DataFrame:
        """
        Compute the Haversine distance between consecutive positions
        along Lagrangian trajectories.
        
        Parameters
        ----------
        cum_dist : bool, optional
            Whether to compute the cumulative distance along the
            trajectory. The default is False.
            
        use_km : bool, optional
            Whether to return distance in kilometers.
            The default is True.
        
        Returns
        -------
        df : DataFrame
            DataFrame containing Lagrangian trajectories and new column
            variable storing the Haversine distance between consecutive
            positions along the trajectory.
        """
        # Defining radius of the Earth, re (m), as volumetric mean radius from NASA.
        # See: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
        re = 6371000
        # Define conversion factor meters to kilometers:
        m_to_km = 1 / 1E3

        # Selecting longitude and latitude values (n -> n+1):
        df_exp = (self._df
                .select(
                        id = pl.col('id'),
                        lon1 = pl.col('lon').list.slice(offset=0, length=pl.col('lon').list.len()-1),
                        lat1 = pl.col('lat').list.slice(offset=0, length=pl.col('lat').list.len()-1),
                        lon2 = pl.col('lon').list.slice(offset=1, length=pl.col('lon').list.len()),
                        lat2 = pl.col('lat').list.slice(offset=1, length=pl.col('lat').list.len())
                    ))

        # Exploding longitude and latitude values from condensed lists to long-format:
        # Adding new column variable for change in longitude (n -> n+1):
        df_exp = (df_exp
                .explode(['lon1', 'lat1', 'lon2', 'lat2'])
                .with_columns(
                    dlon = pl.col('lon2') - pl.col('lon1'),
                    dlat = pl.col('lat2') - pl.col('lat1'),
                    ))

        # Calculating Haversine distance in metres:
        df_exp = (df_exp
                .with_columns(
                    dist = 2*re*np.arcsin(np.sqrt(
                        np.sin(pl.col('dlat').radians()/2)**2 +
                        (np.cos(pl.col('lat1').radians()) *
                         np.cos(pl.col('lat2').radians()) *
                         np.sin(pl.col('dlon').radians()/2)**2)
                        ))
                    )
                )

        # Transforming distance from metres to kilometers:
        if use_km:
            df_exp = df_exp.with_columns(dist=pl.col('dist')*m_to_km)

        # Applying cumulative sum & returning DataFrame to condensed list format:
        if cum_dist:
            df_exp = (df_exp
                    .group_by(by='id', maintain_order=True)
                    .agg(pl.col('dist').cumsum())
                    )
        else:
            df_exp = (df_exp
                    .group_by(by='id', maintain_order=True)
                    .agg(pl.col('dist'))
                    )

        # Appending distance column variable to original DataFrame:
        df = (self._df.with_columns(dist=pl.lit(value=0.0, dtype=pl.List(pl.Float64)))
            .update(df_exp, on='id', how='inner')
            )

        return df

##############################################################################
# Define velocity_direction() method.

    def velocity_direction(self) -> pl.DataFrame:
        """
        Compute the velocity direction of Lagrangian trajectories
        in degrees.
        
        Returns
        -------
        df : DataFrame
            DataFrame containing Lagrangian trajectories and new column
            variable storing the direction of the trajectory in degrees.
        """
        # Selecting longitude and latitude values (n -> n+1):
        df_exp = (self._df
                  .select(
                        id = pl.col('id'),
                        lon1 = pl.col('lon').list.slice(offset=0, length=pl.col('lon').list.len()-1),
                        lat1 = pl.col('lat').list.slice(offset=0, length=pl.col('lat').list.len()-1),
                        lon2 = pl.col('lon').list.slice(offset=1, length=pl.col('lon').list.len()),
                        lat2 = pl.col('lat').list.slice(offset=1, length=pl.col('lat').list.len())
                        ))

        # Exploding longitude and latitude values from condensed lists to long-format:
        # Adding new column variable for change in longitude (n -> n+1):
        df_exp = (df_exp
                .explode(['lon1', 'lat1', 'lon2', 'lat2'])
                .with_columns(
                    dlon = pl.col('lon2') - pl.col('lon1'),
                    ))

        # Calculating velocity direction (bearing) in degrees:
        df_exp = (df_exp
                .with_columns(
                    direction = np.degrees(np.arctan2(
                        pl.col('lat2').radians().cos() * pl.col('dlon').radians().sin(),
                        (pl.col('lat1').radians().cos() * pl.col('lat2').radians().sin())
                        - (pl.col('lat1').radians().sin() * pl.col('lat2').radians().cos()*
                           pl.col('dlon').radians().cos())
                        )))
                )

        # Returning DataFrame to condensed list format:
        df_exp = (df_exp
                .group_by(by='id', maintain_order=True)
                .agg(pl.col('direction'))
                )

        # Appending direction column variable to original DataFrame:
        df = (self._df
              .with_columns(direction=pl.lit(value=0.0, dtype=pl.List(pl.Float64)))
              .update(df_exp, on='id', how='inner')
              )

        return df

##############################################################################
# Define velocity_magnitude() method.

    def velocity_magnitude(self) -> pl.DataFrame:
        """
        Compute the velocity magnitude of Lagrangian trajectories
        in units {dist} / {time}.
        
        Returns
        -------
        df : DataFrame
            DataFrame containing Lagrangian trajectories and new column
            variable storing the magnitude of the trajectory velocity.
        """
        # Defining polars expression to compute absolute difference between
        # consecutive time values and remove initial null value:
        expr_dt = (pl.col('time').list.diff()
                   .list.eval(pl.element().abs())
                   .list.slice(offset=1, length=pl.col('time').list.len()))

        # Selecting columns for velocity magnitude calculation:
        df_exp = (self._df
                  .select(
                        id = pl.col('id'),
                        dist = pl.col('dist'),
                        dt = expr_dt
                        ))

        # Exploding List column variables from condensed to long-format:
        # Calculating velocity magnitude:
        df_exp = (df_exp
                .explode(['dist', 'dt'])
                .with_columns(
                    speed = pl.col('dist') / pl.col('dt'),
                    ))

        # Returning DataFrame to condensed list format:
        df_exp = (df_exp
                .group_by(by='id', maintain_order=True)
                .agg(pl.col('speed'))
                )

        # Appending speed column variable to original DataFrame:
        df = (self._df
              .with_columns(speed=pl.lit(value=0.0, dtype=pl.List(pl.Float64)))
              .update(df_exp, on='id', how='inner')
              )

        return df

##############################################################################
# Define LazyFrames ListOperations Class.
@pl.api.register_lazyframe_namespace("lazy_list_ops")
class LazyListOperations:
    """
    List Operations API Extention for polars LazyFrames to apply
    custom expressions on List dtype columns.

    Parameters
    ----------
    ldf : LazyFrame
        LazyFrame containing positions and properties of Lagrangian
        trajectories.

    Class Methods
    -------------
    apply_expr(expr:pl.Expr, alias:str) -> pl.LazyFrame
        Applies a custom expression on List dtype columns of a LazyFrame.
    """
    def __init__(self, df: pl.LazyFrame):
        self._df = df

##############################################################################
# Define apply_expr() method.

    def apply_expr(self, expr:pl.Expr, alias: str) -> pl.LazyFrame:
        """
        Apply a custom polars expression on List dtype columns of a
        polars LazyFrame.

        Parameters
        ----------
        expr : Expr
            Custom polars expression to apply on List dtype columns.
        alias : str
            Name of column variable to store expression results.

        Returns
        -------
        ldf : LazyFrame
            LazyFrame containing Lagrangian trajectories and new column
            variable storing the result of the custom expression.
        """
        return (self._df
                .explode(
                    columns=[col for col in self._df.columns if self._df.schema[col] == pl.List]
                    )
                .with_columns(
                    expr.alias(alias)
                    )
                .group_by(by='id', maintain_order=True)
                .agg(pl.all())
                )

##############################################################################
# Define haversine_dist() method.

    def haversine_dist(self, cum_dist:bool=False, use_km:bool=True) -> pl.LazyFrame:
        """
        Compute the Haversine distance between consecutive positions
        along Lagrangian trajectories.
        
        Parameters
        ----------
        cum_dist : bool, optional
            Whether to compute the cumulative distance along the
            trajectory. The default is False.
            
        use_km : bool, optional
            Whether to return distance in kilometers.
            The default is True.
        
        Returns
        -------
        ldf : LazyFrame
            LazyFrame containing Lagrangian trajectories and new column
            variable storing the Haversine distance between consecutive
            positions along the trajectory.
        """
        # Defining radius of the Earth, re (m), as volumetric mean radius from NASA.
        # See: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
        re = 6371000
        # Define conversion factor meters to kilometers:
        m_to_km = 1 / 1E3

        # Selecting longitude and latitude values (n -> n+1):
        ldf_exp = (self._df
                   .select(
                       id = pl.col('id'),
                       lon1 = pl.col('lon').list.slice(offset=0, length=pl.col('lon').list.len()-1),
                       lat1 = pl.col('lat').list.slice(offset=0, length=pl.col('lat').list.len()-1),
                       lon2 = pl.col('lon').list.slice(offset=1, length=pl.col('lon').list.len()),
                       lat2 = pl.col('lat').list.slice(offset=1, length=pl.col('lat').list.len())
                    ))

        # Exploding longitude and latitude values from condensed lists to long-format:
        # Adding new column variable for change in longitude (n -> n+1):
        ldf_exp = (ldf_exp
                   .explode(['lon1', 'lat1', 'lon2', 'lat2'])
                   .with_columns(
                       dlon = pl.col('lon2') - pl.col('lon1'),
                       dlat = pl.col('lat2') - pl.col('lat1'),
                       ))

        # Calculating Haversine distance in metres:
        # An intermediate collection is required to
        # update the LazyFrame schema.
        ldf_exp = (ldf_exp
                   .with_columns(
                       dist = (2*re*np.arcsin(np.sqrt(
                           np.sin(pl.col('dlat').radians()/2)**2 +
                           (np.cos(pl.col('lat1').radians()) *
                            np.cos(pl.col('lat2').radians()) *
                            np.sin(pl.col('dlon').radians()/2)**2)
                            ))).cast(pl.Float64)
                            )
                    )

        # Transforming distance from metres to kilometers:
        if use_km:
            ldf_exp = ldf_exp.with_columns(dist=pl.col('dist')*m_to_km)

        # Applying cumulative sum & returning LazyFrame to condensed list format:
        if cum_dist:
            ldf_exp = (ldf_exp
                       .group_by(by='id', maintain_order=True)
                       .agg(pl.col('dist').cumsum())
                       )
        else:
            ldf_exp = (ldf_exp
                       .group_by(by='id', maintain_order=True)
                       .agg(pl.col('dist'))
                       )

        # Appending distance column variable to original LazyFrame:
        ldf = (self._df
               .with_columns(dist=pl.lit(value=0.0, dtype=pl.List(pl.Float64)))
               .update(ldf_exp.select(['id', 'dist']), on='id', how='inner')
               )

        return ldf

##############################################################################
# Define velocity_direction() method.

    def velocity_direction(self) -> pl.LazyFrame:
        """
        Compute the velocity direction of Lagrangian trajectories
        in degrees.
        
        Returns
        -------
        ldf : LazyFrame
            LazyFrame containing Lagrangian trajectories and new column
            variable storing the direction of the trajectory in degrees.
        """
        # Selecting longitude and latitude values (n -> n+1):
        ldf_exp = (self._df
                   .select(
                       id = pl.col('id'),
                       lon1 = pl.col('lon').list.slice(offset=0, length=pl.col('lon').list.len()-1),
                       lat1 = pl.col('lat').list.slice(offset=0, length=pl.col('lat').list.len()-1),
                       lon2 = pl.col('lon').list.slice(offset=1, length=pl.col('lon').list.len()),
                       lat2 = pl.col('lat').list.slice(offset=1, length=pl.col('lat').list.len())
                       ))

        # Exploding longitude and latitude values from condensed lists to long-format:
        # Adding new column variable for change in longitude (n -> n+1):
        ldf_exp = (ldf_exp
                   .explode(['lon1', 'lat1', 'lon2', 'lat2'])
                   .with_columns(
                       dlon = pl.col('lon2') - pl.col('lon1'),
                       ))

        # Calculating velocity direction (bearing) in degrees:
        # An intermediate collection is required to
        # update the LazyFrame schema.
        ldf_exp = (ldf_exp
                   .with_columns(
                       direction = (np.degrees(np.arctan2(
                           pl.col('lat2').radians().cos() * pl.col('dlon').radians().sin(),
                           (pl.col('lat1').radians().cos() * pl.col('lat2').radians().sin())
                           - (pl.col('lat1').radians().sin() * pl.col('lat2').radians().cos()*
                              pl.col('dlon').radians().cos())
                              ))).cast(pl.Float64)
                                )
                    )

        # Returning LazyFrame to condensed list format:
        ldf_exp = (ldf_exp
                   .group_by(by='id', maintain_order=True)
                   .agg(pl.col('direction'))
                   )

        # Appending direction column variable to original LazyFrame:
        ldf = (self._df
               .with_columns(direction=pl.lit(value=0.0, dtype=pl.List(pl.Float64)))
               .update(ldf_exp.select(['id', 'direction']), on='id', how='inner')
               )

        return ldf

##############################################################################
# Define velocity_magnitude() method.

    def velocity_magnitude(self) -> pl.LazyFrame:
        """
        Compute the velocity magnitude of Lagrangian trajectories
        in units {dist} / {time}.
        
        Returns
        -------
        ldf : LazyFrame
            LazyFrame containing Lagrangian trajectories and new column
            variable storing the magnitude of the trajectory velocity.
        """
        # Defining polars expression to compute absolute difference between
        # consecutive time values and remove initial null value:
        expr_dt = (pl.col('time').list.diff()
                   .list.eval(pl.element().abs())
                   .list.slice(offset=1, length=pl.col('time').list.len()))

        # Selecting columns for velocity magnitude calculation:
        ldf_exp = (self._df
                   .select(
                       id = pl.col('id'),
                       dist = pl.col('dist'),
                       dt = expr_dt
                       ))

        # Calculating velocity magnitude:
        # An intermediate collection is required to update the
        # LazyFrame schema.
        ldf_exp = (ldf_exp
                   .explode(['dist', 'dt'])
                   .with_columns(
                       speed = (pl.col('dist') / pl.col('dt')).cast(pl.Float64),
                       )
                       )

        # Returning LazyFrame to condensed list format:
        ldf_exp = (ldf_exp
                   .group_by(by='id', maintain_order=True)
                   .agg(pl.col('speed'))
                   )

        # Appending speed column variable to original LazyFrame:
        ldf = (self._df
               .with_columns(speed=pl.lit(value=0.0, dtype=pl.List(pl.Float64)))
               .update(ldf_exp.select(['id', 'speed']), on='id', how='inner')
               )

        return ldf
