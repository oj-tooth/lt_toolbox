##############################################################################
"""
# filter_frame_utils.py
#
# Description:
# Defines functions for filtering trajectories stored in TrajFrames.
"""
##############################################################################
# Importing relevant packages.

import polars as pl
import numpy as np
from matplotlib.path import Path

##############################################################################
# Define filter_traj() function.

def filter_traj(df:pl.DataFrame,
                variable:str,
                operator:str,
                value:str,
                value_dtype:type,
                drop:bool
                ) -> pl.DataFrame:
    """
    Filter trajectories using conditional on a single column variable
    specified with a string expression.

    Filtering returns a reduced DataFrame where only the
    trajectories meeting the specified condition are retained.
    The exception is when users specify drop=True, in which case
    trajectories meeting the specified condition are dropped from the
    DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing variable to filter.
    variable : str
        Name of the variable to filter.
    operator : str
        Logical operator used to filter variable.
    value : str
        Value used to filter variable.
    value_dtype : type
        Polars type of the value used to filter variable.
    drop : bool
        Indcates if fitered trajectories should be retained in the
        new DataFrame (False) or dropped from the DataFrame (True).

    Returns
    -------
    df_reduced : DataFrame
        Reduced DataFrame, including the Lagrangian trajectories
        which meet (do not meet) the specified filter condition.

    """
    # ---------------------------------------
    # Applying specified filter to DataFrame.
    # ---------------------------------------
    # Apply filter according to specfied comparison operator:
    # Case 1. Equal
    if operator == '==':
        # Filter DataFrame according to drop argument:
        if drop is True:
            df_reduced = df.filter((~pl.col(variable).list.eval(pl.element() == pl.lit(value).cast(value_dtype))).list.any())
        else:
            df_reduced = df.filter((pl.col(variable).list.eval(pl.element() == pl.lit(value).cast(value_dtype))).list.any())

    # Case 2. Not Equal
    elif operator == '!=':
        # Filter DataFrame according to drop argument:
        if drop is True:
            df_reduced = df.filter((~pl.col(variable).list.eval(pl.element() != pl.lit(value).cast(value_dtype))).list.any())
        else:
            df_reduced = df.filter((pl.col(variable).list.eval(pl.element() != pl.lit(value).cast(value_dtype))).list.any())

    # Case 3. Less Than
    elif operator == '<':
        # Filter DataFrame according to drop argument:
        if drop is True:
            df_reduced = df.filter((~pl.col(variable).list.eval(pl.element() < pl.lit(value).cast(value_dtype))).list.any())
        else:
            df_reduced = df.filter((pl.col(variable).list.eval(pl.element() < pl.lit(value).cast(value_dtype))).list.any())

    # Case 4. Greater Than
    elif operator == '>':
        # Filter DataFrame according to drop argument:
        if drop is True:
            df_reduced = df.filter((~pl.col(variable).list.eval(pl.element() > pl.lit(value).cast(value_dtype))).list.any())
        else:
            df_reduced = df.filter((pl.col(variable).list.eval(pl.element() > pl.lit(value).cast(value_dtype))).list.any())

    # Case 5. Less Than or Equal
    elif operator == '<=':
        # Filter DataFrame according to drop argument:
        if drop is True:
            df_reduced = df.filter((~pl.col(variable).list.eval(pl.element() <= pl.lit(value).cast(value_dtype))).list.any())
        else:
            df_reduced = df.filter((pl.col(variable).list.eval(pl.element() <= pl.lit(value).cast(value_dtype))).list.any())

    # Case 6. Greater Than or Equal
    elif operator == '>=':
        # Filter DataFrame according to drop argument:
        if drop is True:
            df_reduced = df.filter((~pl.col(variable).list.eval(pl.element() >= pl.lit(value).cast(value_dtype))).list.any())
        else:
            df_reduced = df.filter((pl.col(variable).list.eval(pl.element() >= pl.lit(value).cast(value_dtype))).list.any())

    # Return filtered DataFrame:
    return df_reduced

##############################################################################
# Define filter_summary() function.

def filter_summary(df:pl.DataFrame,
                   variable:str,
                   operator:str,
                   value:str,
                   value_dtype:type,
                   drop:bool
                   ) -> pl.DataFrame:
    """
    Filter trajectories using conditional on a single column variable
    specified with a string expression.

    Filtering returns a reduced SummaryFrame where only the
    trajectory rows meeting the specified condition are retained.
    The exception is when users specify drop=True, in which case
    trajectory rows meeting the specified condition are dropped from
    the SummaryFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing variable to filter.
    variable : str
        Name of the variable to filter.
    operator : str
        Logical operator used to filter variable.
    value : str
        Value used to filter variable.
    value_dtype : type
        Polars type of the value used to filter variable.
    drop : bool
        Indcates if fitered trajectories should be retained in the
        new DataFrame (False) or dropped from the DataFrame (True).

    Returns
    -------
    df_reduced : DataFrame
        Reduced DataFrame, including the Lagrangian trajectories
        which meet (do not meet) the specified filter condition.

    """
    # ---------------------------------------
    # Applying specified filter to DataFrame.
    # ---------------------------------------
    # Apply filter according to specfied comparison operator:
    # Case 1. Equal
    if operator == '==':
        # Filter DataFrame according to drop argument:
        if drop is True:
            df_reduced = df.filter(~(pl.col(variable) == pl.lit(value).cast(value_dtype)))
        else:
            df_reduced = df.filter(pl.col(variable) == pl.lit(value).cast(value_dtype))

    # Case 2. Not Equal
    elif operator == '!=':
        # Filter DataFrame according to drop argument:
        if drop is True:
            df_reduced = df.filter(~(pl.col(variable) != pl.lit(value).cast(value_dtype)))
        else:
            df_reduced = df.filter(pl.col(variable) != pl.lit(value).cast(value_dtype))

    # Case 3. Less Than
    elif operator == '<':
        # Filter DataFrame according to drop argument:
        if drop is True:
            df_reduced = df.filter(~(pl.col(variable) < pl.lit(value).cast(value_dtype)))
        else:
            df_reduced = df.filter(pl.col(variable) < pl.lit(value).cast(value_dtype))

    # Case 4. Greater Than
    elif operator == '>':
        # Filter DataFrame according to drop argument:
        if drop is True:
            df_reduced = df.filter(~(pl.col(variable) > pl.lit(value).cast(value_dtype)))
        else:
            df_reduced = df.filter(pl.col(variable) > pl.lit(value).cast(value_dtype))

    # Case 5. Less Than or Equal
    elif operator == '<=':
        # Filter DataFrame according to drop argument:
        if drop is True:
            df_reduced = df.filter(~(pl.col(variable) <= pl.lit(value).cast(value_dtype)))
        else:
            df_reduced = df.filter(pl.col(variable) <= pl.lit(value).cast(value_dtype))

    # Case 6. Greater Than or Equal
    elif operator == '>=':
        # Filter DataFrame according to drop argument:
        if drop is True:
            df_reduced = df.filter(~(pl.col(variable) >= pl.lit(value).cast(value_dtype)))
        else:
            df_reduced = df.filter(pl.col(variable) >= pl.lit(value).cast(value_dtype))

    # Return filtered DataFrame:
    return df_reduced

##############################################################################
# Define filter_traj_polygon() function.

def filter_traj_polygon(df:pl.DataFrame,
                        xy_vars:list,
                        x_poly:list,
                        y_poly:list,
                        drop:bool
                        ) -> pl.DataFrame:
    """
    Filter trajectories which intersect a specified polygon.

    Filtering returns the complete trajectories of particles
    which have been inside the boundary of a given polygon at
    any point in their lifetime.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing x and y coordinate variables.
    xy_vars : list(str)
        List of x and y coordinate variable names.
    x_poly : list
        List of x-coordinates representing the boundary of the polygon.
    y_poly : list
        List of y-coordinates representing the boundary of the polygon.
    drop : boolean
        Determines if fitered trajectories should be returned as a
        new TrajArray object (False) or instead dropped from the
        existing TrajArray object (True).

    Returns
    -------
    df_reduced : DataFrame
        Reduced DataFrame, including the Lagrangian trajectories
        which meet (do not meet) the specified filter condition.

    """
    # ----------------
    # Define Polygon:
    # ----------------
    # Initialise Path object using coordinate tuples:
    poly_coords = list(zip(x_poly, y_poly))
    polygon = Path(poly_coords)

    # -------------------------------
    # Define Trajectory Coordinates:
    # -------------------------------
    # Store Lagrangian trajectory points as coordinate tuples:
    traj_points = np.array(list(zip(df[xy_vars[0]].to_numpy(), df[xy_vars[1]].to_numpy())))

    # -----------------------------------
    # Identify Intersecting Trajectories:
    # -----------------------------------
    # Determine Lagrangian trajectory coordinates which intersect polygon boundary:
    poly_bool = pl.Series(name='poly_bool', values=polygon.contains_points(traj_points))
    # Define unique Lagrangian trajectory IDs:
    poly_ids = df['id'].filter(poly_bool).unique()

    # Filter DataFrame to exclude / include IDs which intersect polygon.
    if drop is True:
        df_reduced = df.filter(~pl.col('id').is_in(poly_ids))
    else:
        df_reduced = df.filter(pl.col('id').is_in(poly_ids))

    # Return filtered Lagrangian trajectories:
    return df_reduced
