##############################################################################
# find_utils.py
#
# Description:
# Defines functions for finding indices of attribute variables in trajectories
# objects.
#
# Last Edited:
# 2020/12/29
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.

import numpy as np

##############################################################################
# Define find_traj() function.


def find_traj(self, find_type, variable, val='NaN', min_val='NaN', max_val='NaN'):
    """
    Finding indices within trajectories using attribute variable.

    Find returns the indices of trajectory points where the specified
    attribute variable takes a value between a specified min and max
    (including these values) or value equal to val.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    find_type : string
        Name of find method to be used - 'between' or 'equal'.
    variable : string
        Name of the variable in the trajectories object.
    val : numeric
        Value variable should equal (using 'equal' method).
    min_val : numeric
        Minimum value variable should equal or be greater than
        (using 'between' method).
    max_val : numeric
        Maximum value variable should equal or be less than
        (using 'between' method).

    Returns
    -------
    DataSet.
        Complete trajectories, including all attribute variables,
        which meet the filter specification.
    """

    # -----------------------------
    # Routine for find_between().
    # -----------------------------
    if find_type == 'between':

        # ------------------------------------------------------------
        # Finding indices for attribute variable between min and max.
        # ------------------------------------------------------------
        # Defining indices where trajectory points meeting conditions
        # will be stored, use numpy vectorisation, np.where().
        indices = np.where((self.data[variable].values <= max_val) & (self.data[variable].values >= min_val))

        # Returning the tuple of arrays containing indices for satisfactory
        # trajectory points.
        return indices

    # ----------------------------
    # Routine for find_equal().
    # ----------------------------
    else:

        # -----------------------------------------------------
        # Finding indices for attribute variable equal to val.
        # -----------------------------------------------------

        # Defining indices where trajectory points meeting conditions
        # will be stored, use numpy vectorisation, np.where().
        indices = np.where(self.data[variable].values == val)

        # Returning the tuple of arrays containing indices for satisfactory
        # trajectory points.
        return indices
