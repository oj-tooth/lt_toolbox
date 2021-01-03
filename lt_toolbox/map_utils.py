##############################################################################
# map_utils.py
#
# Description:
# Defines functions for mapping attribute variables for trajectories objects.
#
# Last Edited:
# 2021/01/02
#
# Created By:
# Ollie Tooth
#
##############################################################################
# Importing relevant packages.

import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

##############################################################################
# Define normalise() function.


def normalise(data):
    """
    Normalises values in 1 or 2-dimensional array to between [0, 1]
    by using the minimum and maximum values across the flattened array.

    Parameters
    ----------
    data : ndarray
        1 or 2-dimensional array containing values to be normalised.

    Returns
    -------
    out : ndarray
        1 or 2-dimensional array containing normalised values.
    """
    # -------------------------------------------------
    # Defining minimum and maximum values of the array.
    # -------------------------------------------------
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)

    # -----------------------------------------------
    # Normalising elements of data using min and max.
    # -----------------------------------------------
    # Computations require final dimension to be equal, so transpose.
    data_norm = np.divide((data - data_min), (data_max - data_min))

    # Returning normalised data in ndarray, data_norm.
    return data_norm

##############################################################################
# Define map_trajectories() function.


def map_trajectories(self, col_variable):
    """
    Map surface trajectories (latitudes and longitudes) of
    particles on an orthographic projection of Earth's surface.

    Latitudes and longitudes of particle positions are connected
    to visualise surface trajectories. Trajectories can also be
    optionally coloured according to a specified scalar variable
    given by col_variable.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    col_variable : string
        Name of variable in the trajectories object to colour
        mapped trajectories.

    Returns
    -------
    """
    # ----------------------------------------
    # Configuiring figure dimensions and axes.
    # ----------------------------------------
    # Initialising figure.
    plt.figure()

    # Find the mean latitude and longitude from our
    # plot data in order to centre our orthographic
    # projection.
    lat_mean = np.nanmean(self.data.lat.values)
    lon_mean = np.nanmean(self.data.lon.values)

    # Configuire axes with an orthographic projection,
    # centred on position (lon_mean, lat_mean).
    ax = plt.axes(projection=ccrs.Orthographic(lon_mean, lat_mean))

    # Add land and coastlines to trajectory plot.
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')
    # Add grid lines of constant latitude and longitude.
    ax.gridlines()

    # -----------------------------------------
    # Defining variables from trajectories obj.
    # -----------------------------------------
    # Defining lat and lon for all trajectories.
    lat = self.data.lat.values
    lon = self.data.lon.values

    # -------------------------------------------
    # Subroutine for no specified color variable.
    # -------------------------------------------
    if col_variable is None:

        # -------------------------------------------------
        # Plotting trajectories with default random colors.
        # -------------------------------------------------
        # Determine number of trajectories to plot.
        traj = np.shape(lon)[0]

        # Plot all trajectories, traj.
        for n in np.arange(0, traj-1):
            # Plot trajectories with default colors specififed by
            # matplotlib.
            ax.plot(lon[n, :], lat[n, :], transform=ccrs.PlateCarree())

    # --------------------------------------------
    # Subroutine for 1-dimensional color variable.
    # --------------------------------------------
    elif np.ndim(self.data[col_variable].values) == 1:

        # Defining col as attribute variable col_variable.
        col = self.data[col_variable].values

        # -------------------------------------------------
        # Normalise colour variable to colour trajectories.
        # -------------------------------------------------
        # Normalise col for colours using normalise().
        col_norm = normalise(col)

        # ----------------------------------------
        # Plotting trajectories as coloured lines.
        # ----------------------------------------
        # Determine number of trajectories to plot.
        traj = np.shape(lon)[0]

        # Defining color from col_norm with diverging coolwarm
        # colour map.
        color = cm.coolwarm(col_norm)

        # Plot all trajectories, traj.
        for n in np.arange(0, traj-1):
            # Plot trajectories with single colour defined by color.
            ax.plot(lon[n, :], lat[n, :], c=color[n], transform=ccrs.PlateCarree())

        # ------------------------------
        # Adding colorbar to the figure.
        # ------------------------------
        # Linearly normalise data between 0.0 and 1.0.
        norm = colors.Normalize(vmin=np.nanmin(col), vmax=np.nanmax(col))

        # Defining Scalar Mappable object for colorbar.
        sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)

        # Defining colorbar, cbar, to be 0.9 of the fig height.
        cbar = plt.colorbar(sm, shrink=0.9)
        cbar.ax.get_yaxis().labelpad = 15

        # Creating colorbar label with col_variable attributes.
        col_label = self.data[col_variable].attrs['standard_name'] + " (" + self.data[col_variable].attrs['units'] + ")"
        # Add colobar label with SI units.
        cbar.ax.set_ylabel(col_label, rotation=270)

    # --------------------------------------------
    # Subroutine for 2-dimensional color variable.
    # --------------------------------------------
    # NOTE: Owing to the dependence on multiple loops, the use
    # of a 2-dimensional color variable should only be used with
    # a small number of trajectories (10-100) to avoid long
    # execution times due to computation inefficiency.
    elif np.ndim(self.data[col_variable].values) == 2:

        # Defining col as attribute variable col_var.
        col = self.data[col_variable].values

        # ---------------------------------------------------
        # Normalise colour variable to colour line segments.
        # ---------------------------------------------------
        # Normalise col for colours using normalise().
        col_norm = normalise(col)

        # ------------------------------------------------
        # Plotting trajectories as coloured line segments.
        # ------------------------------------------------
        # Determine number of trajectories to plot.
        traj = np.shape(lon)[0]
        obs = np.shape(lon)[1]

        # Plot all trajectories, traj.
        for n in np.arange(0, traj-1):
            # Within each trajectory iteration define color from col_norm with
            # diverging cool warm colour map.
            color = cm.coolwarm(col_norm[n, :])

            # Plot all line segments between observations, obs.
            for i in np.arange(0, obs-1):
                # Plot trajectories with col_norm of previous particle position
                # as colour of line segment.
                ax.plot([lon[n, i], lon[n, i+1]], [lat[n, i], lat[n, i+1]], c=color[i], transform=ccrs.PlateCarree())

        # ------------------------------
        # Adding colorbar to the figure.
        # ------------------------------
        # Linearly normalise data between 0.0 and 1.0.
        norm = colors.Normalize(vmin=np.nanmin(col), vmax=np.nanmax(col))

        # Defining Scalar Mappable object for colorbar.
        sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=norm)

        # Defining colorbar, cbar, to be 0.9 of the fig height.
        cbar = plt.colorbar(sm, shrink=0.9)
        cbar.ax.get_yaxis().labelpad = 15

        # Creating colorbar label with col_variable attributes.
        col_label = self.data[col_variable].attrs['standard_name'] + " (" + self.data[col_variable].attrs['units'] + ")"
        # Add colobar label with SI units.
        cbar.ax.set_ylabel(col_label, rotation=270)

    # Show plot.
    plt.show()

    # Return figure.
    return
