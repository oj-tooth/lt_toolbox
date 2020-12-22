##############################################################################
# trajectories.py
#
# Description:
# Defines the trajectories Class from a .nc file containing atmosphere
# ocean parcel trajectories (and accompanying tracers), stored following
# CF-conventions implemented with the NCEI trajectory template.
#
# Last Edited:
# 2020/12/22
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.

import xarray as xr
from filter_utils import filter_traj
from compute_utils import compute_displacement
# import matplotlib.pyplot as plt

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
        # ----------------------------------------
        # Storing input Dataset as data attribute.
        # ----------------------------------------
        # Defining data as input xarray DataSet.
        self.data = ds

        # --------------------------------------------
        # Define trajectories obj attribute variables.
        # --------------------------------------------
        # For improved useability, extract variables from data,
        # storing them as variables in the class.

        # Defining list of variables contained in data.
        variables = list(self.data.variables)

        # Set all variables in DataSet to attribute
        # variables of trajectories object.
        for var in variables:
            setattr(self, var, getattr(self.data, var))

##############################################################################
# Define filter_between() method.

    def filter_between(self, variable, min_val, max_val):
        """
        Filter trajectories between two values of an attribute variable.

        Filtering returns the complete trajectories where the specified
        attribute variable takes a value between a specified min and max
        (including these values).

        Parameters
        ----------
        variable : string
            Name of the variable in the trajectories object.
        min_val : numeric
            Minimum value variable should equal or be greater than.
        max_val : numeric
            Maximum value variable should equal or be less than.

        Returns
        -------
        trajectories object
            Complete trajectories, including all attribute variables,
            which meet the filter specification.

        Examples
        --------
        Filtering all trajectories where Latitude is between 0 N - 20 N.
        >>> filtered_traj = trajectories.filter_between('lat', 0, 20)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if (isinstance(min_val, int) or isinstance(min_val, float)) is False:
            raise TypeError("min must be specified as integer or float")

        if (isinstance(max_val, int) or isinstance(max_val, float)) is False:
            raise TypeError("max must be specified as integer or float")

        # Define ds, the filtered DataSet.
        ds = filter_traj(self, filt_type='between', variable=variable, min_val=min_val, max_val=max_val)

        # Returning this filtered DataSet as a trajectories object -
        # this enables multiple filtering to take place.
        return trajectories(ds)

##############################################################################
# Define filter_equal() method.

    def filter_equal(self, variable, val):
        """
        Filter trajectories with attribute variable equal to value.

        Filtering returns the complete trajectories where the specified
        attribute variable takes the value specified by val.

        Parameters
        ----------
        variable : string
            Name of the variable in the trajectories object.
        val : numeric
            Value variable should equal.

        Returns
        -------
        trajectories object
            Complete trajectories, including all attribute variables,
            which meet the filter specification.

        Examples
        --------
        Filtering all trajectories where Latitude equals 0 N.
        >>> filtered_traj = trajectories.filter_equal('lat', 0)
        """
        # -------------------
        # Raising exceptions.
        # -------------------
        if isinstance(variable, str) is False:
            raise TypeError("variable must be specified as a string")

        if (isinstance(variable, int) or isinstance(variable, float)) is False:
            raise TypeError("val must be specified as integer or float")

        # ----------------------------------
        # Defining ds, the filtered DataSet.
        # ----------------------------------
        ds = filter_traj(self, filt_type='equal', variable=variable, val=val)

        # -----------------------------------------------------------------
        # Returning the subsetted xarray DataSet as a trajectories object -
        # this enables multiple filtering to take place.
        # -----------------------------------------------------------------
        return trajectories(ds)

##############################################################################
# Define compute_dx() method.

    def compute_dx(self):
        """
        Compute particle zonal displacements from trajectories.

        Zonal (x) displacements between particle positions for
        all trajectories are returned as a new DataArray, dx,
        within the trajectories object.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.

        Returns
        -------
        trajectories object.
        Original trajectories object is returned with appended attribute
        variable DataArray containing particle zonal displacements
        with dimensions (traj x obs).

        The first observation (obs) for all trajectories
        (traj) is NaN since the zonal distance from the origin
        of a particle at the origin is not defined.

        Examples
        --------
        Computing zonal displacements for all trajectories.
        >>> traj_dx = trajectories.compute_dx()
        """
        # -----------------------------------------
        # Computing dx with compute_displacement().
        # -----------------------------------------
        dx = compute_displacement(self, 'x')

        # ---------------------
        # Adding dx to DataSet.
        # ---------------------
        # Append zonal displacement DataArray to original DataSet.
        self.data['dx'] = xr.DataArray(dx, dims=["traj", "obs"])
        # Adding attributes to zonal displacement DataArray.
        self.data.dx.attrs = {
                             'long_name': "zonal displacement",
                             'standard_name': "dx",
                             'units': "km",
                             'positive': "eastward"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define compute_dy() method.

    def compute_dy(self):
        """
        Compute particle meridional displacements from trajectories.

        Meridional (y) displacements between particle positions for
        all trajectories are returned as a new DataArray, dy,
        within the trajectories object.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.

        Returns
        -------
        trajectories object.
        Original trajectories object is returned with appended attribute
        variable DataArray containing particle meridional displacements
        with dimensions (traj x obs).

        The first observation (obs) for all trajectories
        (traj) is NaN since the meridional distance from the origin
        of a particle at the origin is not defined.

        Examples
        --------
        Computing meridional displacements for all trajectories.
        >>> traj_dy = trajectories.compute_dy()
        """
        # -----------------------------------------
        # Computing dy with compute_displacement().
        # -----------------------------------------
        dy = compute_displacement(self, 'y')

        # ---------------------
        # Adding dy to DataSet.
        # ---------------------
        # Append meridional displacement DataArray to original DataSet.
        self.data['dy'] = xr.DataArray(dy, dims=["traj", "obs"])
        # Adding attributes to meridional displacement DataArray.
        self.data.dy.attrs = {
                                'long_name': "meridional displacement",
                                'standard_name': "dy",
                                'units': "km",
                                'positive': "northward"
                                }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)

##############################################################################
# Define compute_dz() method.

    def compute_dz(self):
        """
        Compute particle vertical displacements from trajectories.

        Vertical (z) displacements between particle positions for
        all trajectories are returned as a new DataArray, dz,
        within the trajectories object.

        Parameters
        ----------
        self : trajectories object
            Trajectories object passed from trajectories class method.

        Returns
        -------
        trajectories object.
        Original trajectories object is returned with appended attribute
        variable DataArray containing particle vertical displacements
        with dimensions (traj x obs).

        The first observation (obs) for all trajectories
        (traj) is NaN since the vertical distance from the origin
        of a particle at the origin is not defined.

        Examples
        --------
        Computing vertical displacements for all trajectories.
        >>> traj_dz = trajectories.compute_dz()
        """
        # -----------------------------------------
        # Computing dz with compute_displacement().
        # -----------------------------------------
        dz = compute_displacement(self, 'z')

        # ---------------------
        # Adding dz to DataSet.
        # ---------------------
        # Append vertical displacement DataArray to original DataSet.
        self.data['dz'] = xr.DataArray(dz, dims=["traj", "obs"])
        # Adding attributes to vertical displacement DataArray.
        self.data.dz.attrs = {
                             'long_name': "vertical displacement",
                             'standard_name': "dz",
                             'units': "km",
                             'positive': "downward"
                             }

        # Return trajectories object with updated DataSet.
        return trajectories(self.data)


##############################################################################
# Testing with ORCA01 Preliminary Data.
traj = trajectories(xr.open_dataset('ORCA1-N406_TRACMASS_output_run.nc'))
traj_test = traj.compute_dz()
print(traj_test.data)
# plt.plot(traj_test.dz[0, :])
# plt.show()
