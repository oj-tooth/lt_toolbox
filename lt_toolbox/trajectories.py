##############################################################################
# trajectories.py
#
# Description:
# Defines the trajectories Class from a .nc file containing atmosphere
# ocean parcel trajectories (and accompanying tracers), stored following
# CF-conventions implemented with the NCEI trajectory template.
#
# Last Edited:
# 2020/12/15
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.

import numpy as np
import xarray as xr
import pandas as pd

##############################################################################
# Define trajectories Class.


class trajectories:
    def __init__(self, ds):
        """
        Create a trajectories object from .nc file.

        Parameters
        ----------
        filename: netCDF File
                  Trajectories stored following CF-conventions implemented
                  with the NCEI trajectory template.

        For NCEI trajectory template see
        https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl

        Returns
        --------

        Examples
        --------
        """
        # Defining data, an xarray DataSet.
        self.data = ds

        # For improved useability, extract variables from data,
        # storing them as variables in the class.

        # Position attribute variables.
        self.lat = self.data.lat
        self.lon = self.data.lon
        self.z = self.data.z

        # Time attribute variable.
        self.time = self.data.time

        # Tracer attribute variables.
        self.temp = self.data.temp
        self.sal = self.data.sal
        self.sigma0 = self.data.sigma0

    def filter_between(self, variable, min, max):
        """
        Filter trajectories between two values of an attribute variable.

        Filtering returns the complete trajectories where the specified
        attribute variable takes a value between a specified min and max
        (including these values).

        Parameters
        ----------
        variable : string
            Name of the variable in the trajectories object.
        min : numeric
            Minimum value variable should equal or be greater than.
        max : numeric
            Maximum value variable should equal or be less than.

        Returns
        -------
        trajectories object
            Complete trajectories, including all attribute variables,
            which meet the filter specification.

        Examples
        --------
        >>> filtered_traj = trajectories.filter_between('lat', 0, 20)
        """
        if variable == 'time':
            # Routine for filtering with time.
            # --------------------------------

            # Finding the minimum and maximum observations from specified min and max.
            obs_min = np.where(self.data[variable].values[0, :] == pd.to_timedelta(min))[0]
            obs_max = np.where(self.data[variable].values[0, :] == pd.to_timedelta(max))[0]

            # Returning the subseteed xarray DataSet as a trajectories object -
            # this enables multiple filtering to take place.
            return trajectories(self.data.isel(obs=xr.DataArray(np.arange(obs_min, obs_max + 1), dims=["obs"])))

        else:
            # Routine for filtering with attribute variables != time.
            # -------------------------------------------------------

            # Defining empty list to store rows where trajectories
            # meeting conditions will be stored
            rows = []

            # Loop over each row and determine if any var values are
            # between min and max (inclusive).
            for i in range(0, len(self.data[variable].values)):
                ind = np.any((self.data[variable].values[i, :] <= max) & (self.data[variable].values[i, :] >= min))

                # For trajectories meeting condition, store the row no. in rows.
                if ind == 1:
                    rows.append(i)
                else:
                    pass

            # Returning the subsetted xarray DataSet as a trajectories object -
            # this enables multiple filtering to take place.
            return trajectories(self.data.isel(traj=xr.DataArray(rows, dims=["traj"])))


# Testing with ORCA01 Preliminary Data.
traj = trajectories(xr.open_dataset('ORCA1-N406_TRACMASS_output_run.nc'))
test = traj.filter_between('time', 2592000000000000, 10*2592000000000000)
print(test.data)
