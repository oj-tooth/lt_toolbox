##############################################################################
# test_find.py
#
# Description:
# Defines testing classes for methods conatined in find_utils.py.
#
# Last Edited:
# 2021/01/11
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.

import pytest
import numpy as np
import numpy.testing as npt
import xarray as xr
from lt_toolbox.trajectory_array import TrajArray

##############################################################################
# Defining test DataSet and trajectories object.

# --------------------------------------------
# Defining ndarrays containing test variables.
# --------------------------------------------

# Below we define a ndarrays containing 4 observations (obs) for
# 4 trajectories (traj) released over 2 seeding levels.

# Trajectory variable.
test_trajectory = np.array([[1, 1, 1, 1],
                            [2, 2, 2, np.nan],
                            [np.nan, 3, 3, 3],
                            [np.nan, 4, 4, 4]],
                           dtype="float64"
                           )

# Latitude variable.
test_lat = np.array([[2, 4, 6, 8],
                    [12, 14, 16, np.nan],
                    [np.nan, 24, 26, 28],
                    [np.nan, 34, 36, 38]],
                    dtype="float64"
                    )

# Longitude variable.
test_lon = np.array([[-40, -40, -40, -40],
                    [-40, -40, -40, np.nan],
                    [np.nan, -40, -40, -40],
                    [np.nan, -40, -40, -40]],
                    dtype="float64"
                    )

poly = [[[-45, 30], [-45, 35], [-35, 35], [-35, 30], [-45, 30]]]

# Depth variable.
test_z = np.array([[-0.5, -1, -1.5, -2],
                  [-0.5, -1, -1.5, np.nan],
                  [np.nan, -2, -2.5, -3],
                  [np.nan, -2, -2.5, -3]],
                  dtype="float64"
                  )

# Time variables - datetime64 and timedelta64.
test_datetime = np.array([['2000-01-01', '2000-01-06', '2000-01-11', '2000-01-16'],
                          ['2000-01-01', '2000-01-06', '2000-01-11', 'nat'],
                          ['nat', '2000-01-06', '2000-01-11', '2000-01-16'],
                          ['nat', '2000-01-06', '2000-01-11', '2000-01-16']],
                         dtype='datetime64'
                         )

test_timedelta = np.array([[np.timedelta64(0, 'D'), np.timedelta64(5, 'D'), np.timedelta64(10, 'D'), np.timedelta64(15, 'D')],
                           [np.timedelta64(0, 'D'), np.timedelta64(5, 'D'), np.timedelta64(10, 'D'), np.timedelta64('nAt')],
                           [np.timedelta64('nAt'), np.timedelta64(5, 'D'), np.timedelta64(10, 'D'), np.timedelta64(15, 'D')],
                           [np.timedelta64('nAt'), np.timedelta64(5, 'D'), np.timedelta64(10, 'D'), np.timedelta64(15, 'D')]]
                          )

# Temperature variable.
test_temp = np.array([[2, 5, 7, 9],
                     [12, 18, 16, np.nan],
                     [np.nan, 14, 17, 25],
                     [np.nan, 12, 18, 16]],
                     dtype='float64'
                     )

# ------------------------------------------------------------
# Defining testing Dataset, test_dataset, from ndarrays above.
# ------------------------------------------------------------
test_dataset_datetime = xr.Dataset(data_vars=dict(
    trajectory=(["traj", "obs"], test_trajectory),
    time=(["traj", "obs"], test_datetime),
    lat=(["traj", "obs"], test_lat),
    lon=(["traj", "obs"], test_lon),
    z=(["traj", "obs"], test_z),
    temp=(["traj", "obs"], test_temp)
))

test_dataset_timedelta = xr.Dataset(data_vars=dict(
    trajectory=(["traj", "obs"], test_trajectory),
    time=(["traj", "obs"], test_timedelta),
    lat=(["traj", "obs"], test_lat),
    lon=(["traj", "obs"], test_lon),
    z=(["traj", "obs"], test_z),
    temp=(["traj", "obs"], test_temp)
))


# Setting the variable attributes for temperature tracer.
test_dataset_datetime.temp.attrs = {
                        'long_name': "in-situ temperature",
                        'standard_name': "temp",
                        'units': "C"
                        }

test_dataset_timedelta.temp.attrs = {
                        'long_name': "in-situ temperature",
                        'standard_name': "temp",
                        'units': "C"
                        }
# ------------------------------------------------------
# Defining test trajectories objects using test_dataset.
# ------------------------------------------------------
# Trajectories object with datetime64 time variable.
traj_datetime = trajectories(test_dataset_datetime)

# Trajectories object with timedelta64 time variable.
traj_timedelta = trajectories(test_dataset_timedelta)

##############################################################################
# Defining TestFind class.

# -----------------------------------------------------------
# Finding trajectory points using attribute variable/polygon.
# -----------------------------------------------------------
# Finding indices within trajectories object using temp. tracer for testing.
test_between = traj_datetime.find_between(variable='temp', min_val=18, max_val=25)
test_equal_single = traj_datetime.find_equal(variable='temp', val=25)
poly = [[[-45, 30], [-45, 35], [-35, 35], [-35, 30], [-45, 30]]]
test_polygon_single = traj_datetime.find_polygon(polygon=poly)
test_polygon_multi = traj_datetime.find_polygon(polygon=[poly[0], poly[0], poly[0], poly[0]])

# Storing expected ndarrays of temp to be returned.
expected_temp_between = np.array([[1, 2, 3], [1, 3, 2]])

expected_temp_equal_single = np.array([[2], [3]])

expected_temp_polygon = np.array([[3], [1]])


class TestFind(object):

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_between, expected_temp_between),
            (test_polygon_single, expected_temp_polygon),
            (test_polygon_multi, expected_temp_polygon),
            (test_equal_single, expected_temp_equal_single)
        ])
    def test_find_values(self, test, expected):
        # Test if find indices are are returned correctly.
        npt.assert_array_almost_equal(test, expected, decimal=6)

    @pytest.mark.parametrize(
        "test, expected",
        [
            (type(test_between[0][0]), np.int64),
            (type(test_polygon_single[0][0]), np.int64),
            (type(test_polygon_multi[0][0]), np.int64),
            (type(test_equal_single[0][0]), np.int64)
        ])
    def test_find_type(self, test, expected):
        # Test if find indices are float64 type.
        assert np.issubdtype(test, expected)

    @pytest.mark.parametrize(
        "test, expected",
        [
            (np.ndim(test_between), 2),
            (np.ndim(test_polygon_single), 2),
            (np.ndim(test_polygon_multi), 2),
            (np.ndim(test_equal_single), 2)
        ])
    def test_find_dims(self, test, expected):
        # Test if find indices have correct no. dimensions.
        assert test == expected

    @pytest.mark.parametrize(
        "test, expected",
        [
            (np.shape(test_between)[0], 2),
            (np.shape(test_polygon_single)[0], 2),
            (np.shape(test_polygon_multi)[0], 2),
            (np.shape(test_equal_single)[0], 2)
        ])
    def test_find_size(self, test, expected):
        # Test if find indices are sized correctly.
        assert test == expected
