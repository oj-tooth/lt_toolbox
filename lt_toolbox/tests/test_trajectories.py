##############################################################################
# test_trajectories.py
#
# Description:
# Defines testing classes for methods conatined in the trajectories class.
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
from lt_toolbox.trajectories import trajectories

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
# Defining TestAddSeed class.

# Add seed_level to trajectories objects for testing.
test_dt = traj_datetime.add_seed()
test_td = traj_timedelta.add_seed()

# Storing expected values of seed_level in ndarray, expected_seed_level.
expected_seed_level = np.array([1, 1, 2, 2])


class TestAddSeed(object):

    def test_add_seed_values_datetime64(self):
        # Test if seed_level values are returned correctly.
        npt.assert_array_equal(test_dt.seed_level.values, expected_seed_level)

    def test_add_seed_values_timedelta64(self):
        # Test if seed_level values are returned correctly.
        npt.assert_array_equal(test_td.seed_level.values, expected_seed_level)

    def test_add_seed_dims(self):
        # Test if seed_level array has correct no. dimensions.
        assert np.ndim(test_dt.seed_level.values) == 1

    def test_add_seed_size(self):
        # Test if seed_level array is sized correctly.
        assert np.shape(test_dt.seed_level.values)[0] == 4

    def test_add_seed_type(self):
        # Test if seed_levels are int64 type.
        assert np.issubdtype(test_dt.seed_level.values.dtype, np.int64)


##############################################################################
# Defining TestAddId class.

# Add unique id to trajectories object for testing.
test_dt = traj_datetime.add_id()

# Storing expected values of id in ndarray, expected_id.
expected_id = np.array([1, 2, 3, 4])


class TestAddId(object):

    def test_add_id_values(self):
        # Test if id values are returned correctly.
        npt.assert_array_equal(test_dt.id.values, expected_id)

    def test_add_id_dims(self):
        # Test if id array has correct no. dimensions.
        assert np.ndim(test_dt.id.values) == 1

    def test_add_id_size(self):
        # Test if id array is sized correctly.
        assert np.shape(test_dt.seed_level.values)[0] == 4

    def test_add_id_type(self):
        # Test if id values are int64 type.
        assert np.issubdtype(test_dt.id.values.dtype, np.int64)


##############################################################################
# Defining TestAddVariable class.

# -------------------------------------------------------
# Defining new variable to append to trajectories object.
# -------------------------------------------------------
# Define temperature in Kelvin, temp_K, as new variable for trajectories
# object.
temp_K = traj_datetime.temp.values + 273.15

# Defining attributes dictionary, attrs, for temp_K variable.
attrs = {'long_name': 'in-situ temperature in Kelvin',
         'standard_name': 'temp_K',
         'units': 'Kelvin'
         }

# Add temp_K to trajectories object for testing.
test_dt = traj_datetime.add_variable(data=temp_K, attributes=attrs)

# Storing expected values of temp_K in ndarray, expected_temp_K.
expected_temp_K = test_temp + 273.15


class TestAddVariable(object):

    def test_add_variable_values(self):
        # Test if temp_K values are returned correctly.
        npt.assert_array_equal(test_dt.temp_K.values, expected_temp_K)

    def test_add_variable_type(self):
        # Test if temp_K values are float64 type.
        assert np.issubdtype(test_dt.temp_K.values.dtype, np.float64)

    def test_add_variable_dims(self):
        # Test if added temp_K variable has correct no. dimensions.
        assert np.ndim(test_dt.temp_K.values) == 2

    def test_add_variable_data(self):
        # Test TypeError when data is not provided as ndarray.
        with pytest.raises(TypeError) as exception_info:
            # Add temp_K to trajectories object for testing with
            # incorrect data.
            traj_datetime.add_variable(data=list(temp_K), attributes=attrs)

        # Testing if the correct error message is returned to the user.
        assert exception_info.match("data must be provided as an ndarray")

    def test_add_variable_attrs(self):
        # Test TypeError when attributes are not provided as a dictionary.
        with pytest.raises(TypeError) as exception_info:
            # Add temp_K to trajectories object for testing with incorrect
            # attributes.
            traj_datetime.add_variable(data=temp_K, attributes=list(attrs))

        # Testing if the correct error message is returned to the user.
        assert exception_info.match("variable attributes must be provided as a dictionary")


##############################################################################
# Defining TestGetValue class.

# ------------------------------------------------------------
# Get values of temp at '2000-01-06' from trajectories object.
# ------------------------------------------------------------
# Add temp_i to trajectories object for testing.
test_dt = traj_datetime.get_value('temp', '2000-01-16')

# Storing expected values of temp in ndarray, expected_temp_i.
expected_temp_i = np.array([9, np.nan, 25, 16])


class TestGetValue(object):

    def test_get_value_values(self):
        # Test if temp_i values are returned correctly.
        npt.assert_array_equal(test_dt.temp_i.values, expected_temp_i)

    def test_get_value_type(self):
        # Test if temp_i values are float64 type.
        assert np.issubdtype(test_dt.temp_i.values.dtype, np.float64)

    def test_get_value_dims(self):
        # Test if added temp_i variable has correct no. dimensions.
        assert np.ndim(test_dt.temp_i.values) == 1

    def test_get_value_size(self):
        # Test if added temp_i variable is sized correctly.
        assert np.shape(test_dt.temp_i.values)[0] == 4

    def test_get_value_variable_type(self):
        # Test TypeError when variable is not provided as string.
        with pytest.raises(TypeError) as exception_info:
            # Get temp from trajectories object for testing with
            # incorrect variable.
            traj_datetime.get_value(variable=1, time_level='2000-01-16')

        # Testing if the correct error message is returned to the user.
        assert exception_info.match("variable must be specified as a string")

    def test_get_value_variable_member(self):
        # Test ValueError when variable is not found in DataSet.
        with pytest.raises(ValueError) as exception_info:
            # Get 'temperature' from trajectories object for testing with
            # incorrect variable.
            traj_datetime.get_value(variable='temperature', time_level='2000-01-16')

        # Testing if the correct error message is returned to the user.
        assert exception_info.match("variable: \'temperature\' not found in Dataset")

    def test_get_value_time_level_type(self):
        # Test TypeError when attributes are not provided as a string.
        with pytest.raises(TypeError) as exception_info:
            # Add temp_i to trajectories object for testing with incorrect
            # time_level.
            traj_datetime.get_value(variable='temp', time_level=1)

        # Testing if the correct error message is returned to the user.
        assert exception_info.match("time_level must be specified as a string in the format YYYY-MM-DD")


##############################################################################
# Defining TestGetMax class.

# ------------------------------------------------
# Get max values of temp from trajectories object.
# ------------------------------------------------
# Add temp_max to trajectories object for testing.
test_dt = traj_datetime.get_max('temp')

# Storing expected values of temp_max in ndarray, expected_temp_max.
expected_temp_max = np.array([9, 18, 25, 18])


class TestGetMax(object):

    def test_get_max_values(self):
        # Test if temp_max values are returned correctly.
        npt.assert_array_equal(test_dt.temp_max.values, expected_temp_max)

    def test_get_max_type(self):
        # Test if temp_max values are float64 type.
        assert np.issubdtype(test_dt.temp_max.values.dtype, np.float64)

    def test_get_max_dims(self):
        # Test if added temp_max variable has correct no. dimensions.
        assert np.ndim(test_dt.temp_max.values) == 1

    def test_get_max_size(self):
        # Test if added temp_max variable is sized correctly.
        assert np.shape(test_dt.temp_max.values)[0] == 4

    def test_get_max_variable_type(self):
        # Test TypeError when variable is not provided as string.
        with pytest.raises(TypeError) as exception_info:
            # Get temp_max from trajectories object for testing with
            # incorrect variable.
            traj_datetime.get_max(variable=1)

        # Testing if the correct error message is returned to the user.
        assert exception_info.match("variable must be specified as a string")

    def test_get_max_variable_member(self):
        # Test ValueError when variable is not found in DataSet.
        with pytest.raises(ValueError) as exception_info:
            # Get 'temperature' from trajectories object for testing with
            # incorrect variable.
            traj_datetime.get_max(variable='temperature')

        # Testing if the correct error message is returned to the user.
        assert exception_info.match("variable: \'temperature\' not found in Dataset")


##############################################################################
# Defining TestGetMin class.

# ------------------------------------------------
# Get min values of temp from trajectories object.
# ------------------------------------------------
# Add temp_min to trajectories object for testing.
test_dt = traj_datetime.get_min('temp')

# Storing expected values of temp_min in ndarray, expected_temp_min.
expected_temp_min = np.array([2, 12, 14, 12])


class TestGetMin(object):

    def test_get_min_values(self):
        # Test if temp_min values are returned correctly.
        npt.assert_array_equal(test_dt.temp_min.values, expected_temp_min)

    def test_get_min_type(self):
        # Test if temp_min values are float64 type.
        assert np.issubdtype(test_dt.temp_min.values.dtype, np.float64)

    def test_get_min_dims(self):
        # Test if added temp_min variable has correct no. dimensions.
        assert np.ndim(test_dt.temp_min.values) == 1

    def test_get_main_size(self):
        # Test if added temp_min variable is sized correctly.
        assert np.shape(test_dt.temp_min.values)[0] == 4

    def test_get_min_variable_type(self):
        # Test TypeError when variable is not provided as string.
        with pytest.raises(TypeError) as exception_info:
            # Get temp_min from trajectories object for testing with
            # incorrect variable.
            traj_datetime.get_min(variable=1)

        # Testing if the correct error message is returned to the user.
        assert exception_info.match("variable must be specified as a string")

    def test_get_min_variable_member(self):
        # Test ValueError when variable is not found in DataSet.
        with pytest.raises(ValueError) as exception_info:
            # Get 'temperature' from trajectories object for testing with
            # incorrect variable.
            traj_datetime.get_min(variable='temperature')

        # Testing if the correct error message is returned to the user.
        assert exception_info.match("variable: \'temperature\' not found in Dataset")


##############################################################################
# Defining TestStartTime class.

# ---------------------------------------------------------
# Get start times of trajectories from trajectories object.
# ---------------------------------------------------------
# Add t_start (datetime64) to trajectories object for testing.
test_dt = traj_datetime.get_start_time()
# Add t_start (timedelta64) to trajectories object for testing.
test_td = traj_timedelta.get_start_time()

# Storing expected values of t_start in dattime64 in ndarray,
# expected_t_start_dt.
expected_t_start_dt = np.array(['2000-01-01',
                                '2000-01-01',
                                '2000-01-06',
                                '2000-01-06'],
                               dtype="datetime64"
                               )

# Storing expected values of t_start in timedelta64 in ndarray,
# expected_t_start_td.
expected_t_start_td = np.array([np.timedelta64(0, 'D'),
                                np.timedelta64(0, 'D'),
                                np.timedelta64(5, 'D'),
                                np.timedelta64(5, 'D')],
                               )


class TestGetStartTime(object):

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_dt.t_start.values, expected_t_start_dt),
            (test_td.t_start.values, expected_t_start_td)
        ])
    def test_get_start_time_values(self, test, expected):
        # Test if t_start values are returned correctly.
        npt.assert_array_equal(test, expected)

    def test_get_start_time_type_datetime64(self):
        # Test if t_start values are datetime64 type.
        assert np.issubdtype(test_dt.t_start.values.dtype, np.datetime64)

    def test_get_start_time_type_timedelta64(self):
        # Test if t_start values are timedelta64 type.
        assert np.issubdtype(test_td.t_start.values.dtype, np.timedelta64)

    def test_get_start_time_dims(self):
        # Test if added t_start variable has correct no. dimensions.
        assert np.ndim(test_dt.t_start.values) == 1

    def test_get_start_time_size(self):
        # Test if added t_start variable is sized correctly.
        assert np.shape(test_dt.t_start.values)[0] == 4


##############################################################################
# Defining TestEndTime class.

# ---------------------------------------------------------
# Get exit times of trajectories from trajectories object.
# ---------------------------------------------------------
# Add t_end (datetime64) to trajectories object for testing.
test_dt = traj_datetime.get_end_time()
# Add t_end (timedelta64) to trajectories object for testing.
test_td = traj_timedelta.get_end_time()

# Storing expected values of t_end in datetime64 in ndarray,
# expected_t_end_dt.
expected_t_end_dt = np.array(['2000-01-16',
                              '2000-01-11',
                              '2000-01-16',
                              '2000-01-16'],
                             dtype="datetime64"
                             )

# Storing expected values of t_end in timedelta64 in ndarray,
# expected_t_end_td.
expected_t_end_td = np.array([np.timedelta64(15, 'D'),
                              np.timedelta64(10, 'D'),
                              np.timedelta64(15, 'D'),
                              np.timedelta64(15, 'D')]
                             )


class TestGetEndTime(object):

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_dt.t_end.values, expected_t_end_dt),
            (test_td.t_end.values, expected_t_end_td)
        ])
    def test_get_end_time_values(self, test, expected):
        # Test if t_end values are returned correctly.
        npt.assert_array_equal(test, expected)

    def test_get_end_time_type_datetime64(self):
        # Test if t_end values are datetime64 type.
        assert np.issubdtype(test_dt.t_end.values.dtype, np.datetime64)

    def test_get_end_time_type_timedelta64(self):
        # Test if t_end values are timedelta64 type.
        assert np.issubdtype(test_td.t_end.values.dtype, np.timedelta64)

    def test_get_end_time_dims(self):
        # Test if added t_end variable has correct no. dimensions.
        assert np.ndim(test_dt.t_end.values) == 1

    def test_get_end_time_size(self):
        # Test if added t_end variable is sized correctly.
        assert np.shape(test_dt.t_end.values)[0] == 4


##############################################################################
# Defining TestDuration class.

# ------------------------------------------------------------
# Get duration times of trajectories from trajectories object.
# ------------------------------------------------------------
# Add t_total to trajectories object for testing.
test_dt = traj_datetime.get_duration()
# Add t_total to trajectories object for testing.
test_td = traj_timedelta.get_duration()

# Storing expected values of t_total in timedelta64 in ndarray,
# expected_t_total_dt, for datetime64 input.
expected_t_total = np.array([np.timedelta64(15, 'D'),
                             np.timedelta64(10, 'D'),
                             np.timedelta64(10, 'D'),
                             np.timedelta64(10, 'D')]
                            )


class TestDuration(object):

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_dt.t_total.values, expected_t_total),
            (test_td.t_total.values, expected_t_total)
        ])
    def test_get_duration_values(self, test, expected):
        # Test if t_total values are returned correctly.
        npt.assert_array_equal(test, expected)

    def test_get_duration_type_datetime64(self):
        # Test if t_total values are timedelta64 type.
        assert np.issubdtype(test_dt.t_total.values.dtype, np.timedelta64)

    def test_get_duration_type_timedelta64(self):
        # Test if t_total values are timedelta64 type.
        assert np.issubdtype(test_td.t_total.values.dtype, np.timedelta64)

    def test_get_duration_dims(self):
        # Test if added t_total variable has correct no. dimensions.
        assert np.ndim(test_dt.t_total.values) == 1

    def test_get_duration_size(self):
        # Test if added t_total variable is sized correctly.
        assert np.shape(test_dt.t_total.values)[0] == 4


##############################################################################
# Defining TestStartLoc class.

# -------------------------------------------------------------
# Get start locations of trajectories from trajectories object.
# -------------------------------------------------------------
# Add lat/lon/z_start to trajectories object for testing.
test_dt = traj_datetime.get_start_loc()

# Storing expected values of lat/lon/z_start in ndarrays,
# expected_lat_start, expected_lon_start and expected_z_start.
expected_lat_start = np.array([2, 12, 24, 34], dtype="float64")
expected_lon_start = np.array([-40, -40, -40, -40], dtype="float64")
expected_z_start = np.array([-0.5, -0.5, -2, -2], dtype="float64")


class TestGetStartLoc(object):

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_dt.lat_start.values, expected_lat_start),
            (test_dt.lon_start.values, expected_lon_start),
            (test_dt.z_start.values, expected_z_start)
        ])
    def test_get_start_loc_values(self, test, expected):
        # Test if lat/lon/z_start values are returned correctly.
        npt.assert_array_equal(test, expected)

    def test_get_start_loc_type_lat(self):
        # Test if lat_start values are float64 type.
        assert np.issubdtype(test_dt.lat_start.values.dtype, np.float64)

    def test_get_start_loc_type_lon(self):
        # Test if lon_start values are float64 type.
        assert np.issubdtype(test_dt.lon_start.values.dtype, np.float64)

    def test_get_start_loc_type_z(self):
        # Test if z_start values are float64 type.
        assert np.issubdtype(test_dt.z_start.values.dtype, np.float64)

    def test_get_start_loc_dims(self):
        # Test if added lat_start variable has correct no. dimensions.
        assert np.ndim(test_dt.lat_start.values) == 1

    def test_get_start_loc_size(self):
        # Test if added lat_start variable is sized correctly.
        assert np.shape(test_dt.lat_start.values)[0] == 4


##############################################################################
# Defining TestEndLoc class.

# -----------------------------------------------------------
# Get end locations of trajectories from trajectories object.
# -----------------------------------------------------------
# Add lat/lon/z_end to trajectories object for testing.
test_dt = traj_datetime.get_end_loc()

# Storing expected values of lat/lon/z_end in ndarrays,
# expected_lat_end, expected_lon_end and expected_z_end.
expected_lat_end = np.array([8, 16, 28, 38], dtype="float64")
expected_lon_end = np.array([-40, -40, -40, -40], dtype="float64")
expected_z_end = np.array([-2, -1.5, -3, -3], dtype="float64")


class TestGetEndLoc(object):

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_dt.lat_end.values, expected_lat_end),
            (test_dt.lon_end.values, expected_lon_end),
            (test_dt.z_end.values, expected_z_end)
        ])
    def test_get_end_loc_values(self, test, expected):
        # Test if lat/lon/z_end values are returned correctly.
        npt.assert_array_equal(test, expected)

    def test_get_end_loc_type_lat(self):
        # Test if lat_end values are float64 type.
        assert np.issubdtype(test_dt.lat_end.values.dtype, np.float64)

    def test_get_end_loc_type_lon(self):
        # Test if lon_end values are float64 type.
        assert np.issubdtype(test_dt.lon_end.values.dtype, np.float64)

    def test_get_end_loc_type_z(self):
        # Test if z_end values are float64 type.
        assert np.issubdtype(test_dt.z_end.values.dtype, np.float64)

    def test_get_end_loc_dims(self):
        # Test if added lat_end variable has correct no. dimensions.
        assert np.ndim(test_dt.lat_end.values) == 1

    def test_get_end_loc_size(self):
        # Test if added lat_end variable is sized correctly.
        assert np.shape(test_dt.lat_end.values)[0] == 4


##############################################################################
# Defining TestComputeDistance class.

# ------------------------------------------------------
# Get distance travelled along trajectories of particles.
# ------------------------------------------------------
# Compute dist and cumdist from trajectories object for testing.
test_dist_km = traj_datetime.compute_distance(cumsum_dist=False, unit='km')
test_cumdist_km = traj_datetime.compute_distance(cumsum_dist=True, unit='km')
test_dist_m = traj_datetime.compute_distance(cumsum_dist=False, unit='m')
test_cumdist_m = traj_datetime.compute_distance(cumsum_dist=True, unit='m')

# Storing expected values of dist and cumdist in ndarrays for both distance
# and cumulative distance in both meters and kilometers.
expected_dist_m = np.array([[np.nan, 222389.85328968, 222389.85328968, 222389.85328968],
                            [np.nan, 222389.85328968, 222389.85328968, np.nan],
                            [np.nan, np.nan, 222389.85328968, 222389.85328968],
                            [np.nan, np.nan, 222389.85328968, 222389.85328968]]
                           )

expected_cumdist_m = np.array([[0., 222389.85328968, 444779.70657936, 667169.55986904],
                               [0., 222389.85328968, 444779.70657936, 444779.70657936],
                               [0., 0., 222389.85328968, 444779.70657936],
                               [0., 0., 222389.85328968, 444779.70657936]]
                              )

expected_dist_km = expected_dist_m / 1000
expected_cumdist_km = expected_cumdist_m / 1000


class TestComputeDistance(object):

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_dist_km.dist.values, expected_dist_km),
            (test_cumdist_km.cumdist.values, expected_cumdist_km),
            (test_dist_m.dist.values, expected_dist_m),
            (test_cumdist_m.cumdist.values, expected_cumdist_m),
        ])
    def test_compute_distance_values(self, test, expected):
        # Test if dist and cumdist values are returned correctly.
        npt.assert_array_almost_equal(test, expected, decimal=6)

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_dist_km.dist.values.dtype, np.float64),
            (test_cumdist_km.cumdist.values.dtype, np.float64),
        ])
    def test_compute_distance_type(self, test, expected):
        # Test if dist and cumdist values are float64 type.
        assert np.issubdtype(test, expected)

    @pytest.mark.parametrize(
        "test, expected",
        [
            (np.ndim(test_dist_km.dist.values), 2),
            (np.ndim(test_cumdist_km.cumdist.values), 2),
        ])
    def test_compute_distance_dims(self, test, expected):
        # Test if added dist and cumdist variables have correct no. dimensions.
        assert test == expected

    @pytest.mark.parametrize(
        "test, expected",
        [
            (np.shape(test_dist_km.dist.values)[0], 4),
            (np.shape(test_cumdist_km.cumdist.values)[0], 4),
        ])
    def test_compute_distance_size(self, test, expected):
        # Test if added dist and cumdist variables are sized correctly.
        assert test == expected


##############################################################################
# Defining TestComputeDisplacement class.

# ---------------------------------------------------------
# Get zonal/meridional/vertical displacements of particles.
# ---------------------------------------------------------
# Compute dx/dy/dz from trajectories object for testing.
test_dx_km = traj_datetime.compute_dx(unit='km')
test_dy_km = traj_datetime.compute_dy(unit='km')
test_dz_km = traj_datetime.compute_dz(unit='km')
test_dx_m = traj_datetime.compute_dx(unit='m')
test_dy_m = traj_datetime.compute_dy(unit='m')
test_dz_m = traj_datetime.compute_dz(unit='m')

# Storing expected values of dx/dy/dz in ndarrays in both
# meters and kilometers.
expected_dx_m = np.array([[np.nan, 0., 0., 0.],
                          [np.nan, 0., 0., np.nan],
                          [np.nan, np.nan, 0., 0.],
                          [np.nan, np.nan, 0., 0.]]
                         )

expected_dy_m = np.array([[np.nan, 222389.85328912, 222389.85328912, 222389.85328912],
                          [np.nan, 222389.85328912, 222389.85328912, np.nan],
                          [np.nan, np.nan, 222389.85328912, 222389.85328912],
                          [np.nan, np.nan, 222389.85328912, 222389.85328912]]
                         )

expected_dz_m = np.array([[np.nan, -0.5, -0.5, -0.5],
                          [np.nan, -0.5, -0.5, np.nan],
                          [np.nan, np.nan, -0.5, -0.5],
                          [np.nan, np.nan, -0.5, -0.5]]
                         )

expected_dx_km = expected_dx_m / 1000
expected_dy_km = expected_dy_m / 1000
expected_dz_km = expected_dz_m / 1000


class TestComputeDisplacement(object):

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_dx_m.dx.values, expected_dx_m),
            (test_dy_m.dy.values, expected_dy_m),
            (test_dz_m.dz.values, expected_dz_m),
            (test_dx_km.dx.values, expected_dx_km),
            (test_dy_km.dy.values, expected_dy_km),
            (test_dz_km.dz.values, expected_dz_km)
        ])
    def test_compute_displacement_values(self, test, expected):
        # Test if dx/dy/dz values are returned correctly.
        npt.assert_array_almost_equal(test, expected, decimal=6)

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_dx_m.dx.values.dtype, np.float64),
            (test_dy_m.dy.values.dtype, np.float64),
            (test_dz_m.dz.values.dtype, np.float64)
        ])
    def test_compute_displacement_type(self, test, expected):
        # Test if dx/dy/dz values are float64 type.
        assert np.issubdtype(test, expected)

    @pytest.mark.parametrize(
        "test, expected",
        [
            (np.ndim(test_dx_m.dx.values), 2),
            (np.ndim(test_dy_m.dy.values), 2),
            (np.ndim(test_dz_m.dz.values), 2)
        ])
    def test_compute_displacement_dims(self, test, expected):
        # Test if added dx/dy/dz variable has correct no. dimensions.
        assert test == expected

    @pytest.mark.parametrize(
        "test, expected",
        [
            (np.shape(test_dx_m.dx.values)[0], 4),
            (np.shape(test_dy_m.dy.values)[0], 4),
            (np.shape(test_dz_m.dz.values)[0], 4),
        ])
    def test_compute_displacement_size(self, test, expected):
        # Test if added dx/dy/dz variable is sized correctly.
        assert test == expected


##############################################################################
# Defining TestComputeVelocity class.

# ------------------------------------------------------
# Get zonal/meridional/vertical velocities of particles.
# ------------------------------------------------------
# Compute u/v/w from trajectories object for testing.
test_dt = traj_datetime.compute_u(unit='m/s')
test_dt = traj_datetime.compute_v(unit='m/s')
test_dt = traj_datetime.compute_w(unit='m/s')

test_td = traj_timedelta.compute_u(unit='m/s')
test_td = traj_timedelta.compute_v(unit='m/s')
test_td = traj_timedelta.compute_w(unit='m/s')

# Computing time-step, dt, as 5 days in seconds.
dt = 5*(3600*24)

# Storing expected values of u/v/w in ndarrays in m/s.
expected_u = expected_dx_m / dt

expected_v = expected_dy_m / dt

expected_w = expected_dz_m / dt


class TestComputeVelocity(object):

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_dt.u.values, expected_u),
            (test_dt.v.values, expected_v),
            (test_dt.w.values, expected_w),
            (test_td.u.values, expected_u),
            (test_td.v.values, expected_v),
            (test_td.w.values, expected_w)
        ])
    def test_compute_velocity_values(self, test, expected):
        # Test if u/v/w values are returned correctly.
        npt.assert_array_almost_equal(test, expected, decimal=6)

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_dt.u.values.dtype, np.float64),
            (test_dt.v.values.dtype, np.float64),
            (test_dt.w.values.dtype, np.float64),
            (test_td.u.values.dtype, np.float64),
            (test_td.v.values.dtype, np.float64),
            (test_td.w.values.dtype, np.float64)
        ])
    def test_compute_velocity_type(self, test, expected):
        # Test if u/v/w values are float64 type.
        assert np.issubdtype(test, expected)

    @pytest.mark.parametrize(
        "test, expected",
        [
            (np.ndim(test_dt.u.values), 2),
            (np.ndim(test_dt.v.values), 2),
            (np.ndim(test_dt.w.values), 2),
            (np.ndim(test_td.u.values), 2),
            (np.ndim(test_td.v.values), 2),
            (np.ndim(test_td.w.values), 2)
        ])
    def test_compute_velocity_dims(self, test, expected):
        # Test if added u/v/w variable has correct no. dimensions.
        assert test == expected

    @pytest.mark.parametrize(
        "test, expected",
        [
            (np.shape(test_dt.u.values)[0], 4),
            (np.shape(test_dt.v.values)[0], 4),
            (np.shape(test_dt.w.values)[0], 4),
            (np.shape(test_td.u.values)[0], 4),
            (np.shape(test_td.v.values)[0], 4),
            (np.shape(test_td.w.values)[0], 4)
        ])
    def test_compute_velocity_size(self, test, expected):
        # Test if added u/v/w variable is sized correctly.
        assert test == expected


##############################################################################
# Defining TestFilters class.

# --------------------------------------------------------
# Filtering trajectories using attribute variable/polygon.
# --------------------------------------------------------
# Filtering trajectories object using temp. tracer for testing.
test_between = traj_datetime.filter_between(variable='temp', min_val=18, max_val=25, drop=False)
test_equal_single = traj_datetime.filter_equal(variable='temp', val=25, drop=False)
test_equal_multi = traj_datetime.filter_equal(variable='temp', val=[2, 25], drop=False)

# Filtering trajectories object using time for testing.
test_between_time = traj_datetime.filter_between(variable='time', min_val=np.datetime64('2000-01-01'), max_val=np.datetime64('2000-01-06'), drop=False)
test_equal_single_time = traj_datetime.filter_equal(variable='time', val=np.datetime64('2000-01-01'), drop=False)
test_equal_multi_time = traj_datetime.filter_equal(variable='time', val=[np.datetime64('2000-01-01'), np.datetime64('2000-01-06')], drop=False)

# Filtering trajectories object using polygon, poly, for testing.
poly = [[[-45, 30], [-45, 35], [-35, 35], [-35, 30], [-45, 30]]]
test_polygon_single_pos = traj_datetime.filter_polygon(polygon=poly, method='pos', drop=False)
test_polygon_multi_pos = traj_datetime.filter_polygon(polygon=[poly[0], poly[0], poly[0], poly[0]], method='pos', drop=False)

# Storing expected ndarrays of temp to be returned.
expected_temp_between = np.array([[12, 18, 16, np.nan],
                                  [np.nan, 14, 17, 25],
                                  [np.nan, 12, 18, 16]],
                                 dtype='float64'
                                 )

expected_temp_equal_single = np.array([[np.nan, 14, 17, 25]], dtype='float64')

expected_temp_equal_multi = np.array([[2, 5, 7, 9],
                                      [np.nan, 14, 17, 25]],
                                     dtype='float64'
                                     )

expected_time_between = np.array([[2, 5],
                                  [12, 18],
                                  [np.nan, 14],
                                  [np.nan, 12]],
                                 dtype='float64'
                                 )

expected_time_equal_single = np.array([[2],
                                       [12],
                                       [np.nan],
                                       [np.nan]],
                                      dtype='float64')

expected_time_equal_multi = np.array([[2, 5],
                                      [12, 18],
                                      [np.nan, 14],
                                      [np.nan, 12]],
                                     dtype='float64')

expected_temp_polygon = np.array([[np.nan, 12, 18, 16]], dtype='float64')


class TestFilters(object):

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_between.temp.values, expected_temp_between),
            (test_polygon_single_pos.temp.values, expected_temp_polygon),
            (test_polygon_multi_pos.temp.values, expected_temp_polygon),
            (test_equal_single.temp.values, expected_temp_equal_single),
            (test_equal_multi.temp.values, expected_temp_equal_multi),
            (test_between_time.temp.values, expected_time_between),
            (test_equal_single_time.temp.values, expected_time_equal_single),
            (test_equal_multi_time.temp.values, expected_time_equal_multi)
        ])
    def test_filter_values(self, test, expected):
        # Test if filtered temp values are returned correctly.
        npt.assert_array_almost_equal(test, expected, decimal=6)

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_between.temp.values.dtype, np.float64),
            (test_polygon_single_pos.temp.values.dtype, np.float64),
            (test_polygon_multi_pos.temp.values.dtype, np.float64),
            (test_equal_single.temp.values.dtype, np.float64),
            (test_equal_multi.temp.values.dtype, np.float64),
            (test_between_time.temp.values.dtype, np.float64),
            (test_equal_single_time.temp.values.dtype, np.float64),
            (test_equal_multi_time.temp.values.dtype, np.float64)
        ])
    def test_filter_type(self, test, expected):
        # Test if filtered temp values are float64 type.
        assert np.issubdtype(test, expected)

##############################################################################
# Defining TestTimes class.

# ------------------------------------------
# Computing residence time and transit time.
# ------------------------------------------
# Computing residence / transit times for testing.
poly = [[[-45, 30], [-45, 35], [-35, 35], [-35, 30], [-45, 30]]]

test_residence_time_dt = traj_datetime.compute_residence_time(polygon=poly)
test_residence_time_td = traj_timedelta.compute_residence_time(polygon=poly)
test_transit_time_dt = traj_datetime.filter_polygon(polygon=poly, method='pos').compute_transit_time(polygon=poly)
test_transit_time_td = traj_timedelta.filter_polygon(polygon=poly, method='pos').compute_transit_time(polygon=poly)

# Storing expected ndarrays of temp to be returned.
expected_residence_time = np.array([0, 0, 0, 5])
expected_transit_time = np.array([0])


class TestTimes(object):

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_residence_time_dt.residence_time.values, expected_residence_time),
            (test_residence_time_td.residence_time.values, expected_residence_time),
            (test_transit_time_dt.transit_time.values, expected_transit_time),
            (test_transit_time_td.transit_time.values, expected_transit_time),
        ])
    def test_time_values(self, test, expected):
        # Test if time array elements are returned correctly.
        npt.assert_array_almost_equal(test, expected, decimal=6)

    @pytest.mark.parametrize(
        "test, expected",
        [
            (test_residence_time_dt.residence_time.values.dtype, np.float64),
            (test_residence_time_td.residence_time.values.dtype, np.float64),
            (test_transit_time_dt.transit_time.values.dtype, np.float64),
            (test_transit_time_td.transit_time.values.dtype, np.float64),
        ])
    def test_time_type(self, test, expected):
        # Test if returned time array elements are float64 type.
        assert np.issubdtype(test, expected)
