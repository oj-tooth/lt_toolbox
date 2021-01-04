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
import scipy.stats as stats
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
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black', facecolor='white')
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

##############################################################################
# Define map_probability() function.


def map_probability(self, bin_res, prob_type, cmap):
    """
    Map binned probability distribution of particle positions
    or particle pathways on an orthographic projection of
    Earth's surface.

    Particle positions are binned into a 2-dimensional
    (x-y) histogram and normalised by the total number
    of particle positions ('pos') or the total number
    of particles ('traj').

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    bin_res : numeric
        The resolution (degrees) of the grid on to which particle
        positions will be binned.
    prob_type : string
        The type of probability to be computed. 'pos' - all particle
        positions are binned and then normalised by the total number
        of particle positions. 'traj' - for each particle positions
        are counted once per bin and then normalised by the total
        number of particles.
    cmap : string
        A colormap instance or registered colormap name.

    Returns
    -------

    Note
    ----
    Aliasing is an important consideration when selecting a bin
    resolution. If the selected grid resolution is too fine,
    particles may be advected through a bin within one time step
    without adding to the bin count. This is especially relevant
    for simulations with long output time steps (> 5 days).
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
    ax.add_feature(cartopy.feature.LAND, zorder=4, edgecolor='black', facecolor='white')
    # Add grid lines of constant latitude and longitude.
    ax.gridlines()

    # -----------------------------------------
    # Defining variables from trajectories obj.
    # -----------------------------------------
    # Defining lat and lon for all trajectories.
    lat = self.data.lat.values
    lon = self.data.lon.values

    # ---------------------------------------------------------
    # Defining grid on which particle positions will be binned.
    # ---------------------------------------------------------
    # Finding the maximum and minimum values of lat and lon
    # to the nearest degree E/N.
    lat_max = np.ceil(np.nanmax(lat))
    lat_min = np.floor(np.nanmin(lat))

    lon_max = np.ceil(np.nanmax(lon))
    lon_min = np.floor(np.nanmin(lon))

    # Defining bin size with specified bin_res.
    dx = dy = bin_res

    # Defining bin edges in 1-dimensional arrays.
    bin_x = np.arange(lon_min, lon_max + dx, dx)
    bin_y = np.arange(lat_min, lat_max + dy, dy)

    # -------------------------------------------------------
    # Subroutine for probability with all particle positions.
    # -------------------------------------------------------
    if prob_type == 'pos':

        # -----------------------------------
        # Computing particle density on grid.
        # -----------------------------------
        # Using scipy to count the number of particle positions per bin
        stat = stats.binned_statistic_2d(x=lon.flatten(), y=lat.flatten(), values=None, statistic='count', bins=[bin_x, bin_y])
        # For empty bin set count value to NaN.
        stat.statistic[stat.statistic == 0] = np.nan

        # -----------------------------
        # Computing probability on grid.
        #  ------------------------------
        # Defining number of particle positions, npos.
        npos = len(lon.flatten())  # lon/lat could be used here.
        # Compute probability as a percentage, prob.
        prob = (stat.statistic / npos) * 100

    # ----------------------------------------------
    # Subroutine for probability with all particles.
    # ----------------------------------------------
    elif prob_type == 'traj':

        # Defining array to store particle density.
        density = np.zeros([len(bin_x) - 1, len(bin_y) - 1])

        # Defining no. particles, ntpart.
        npart = np.shape(lat)[0]  # lat/lon could be used here.

        # Iterate over all particles (trajectories).
        for i in np.arange(0, npart):
            # Using scipy to count the number of particles per bin.
            stat = stats.binned_statistic_2d(lon[i, :], lat[i, :], None, 'count', bins=[bin_x, bin_y])
            # Where a particle is counted more than once in bin set = 1.
            stat.statistic[stat.statistic > 1] = 1
            # Update density with counts from particle.
            density = density + stat.statistic

        # For empty bin set density value to NaN.
        density[density == 0] = np.nan

        # -----------------------------
        # Computing probability on grid.
        #  ------------------------------
        # Compute probability as a percentage, prob.
        prob = (density / npart) * 100

    # ---------------------------
    # Defining grid for plotting.
    # ---------------------------
    # Defining grid from 1-d bin edge arrays.
    y, x = np.meshgrid(bin_y, bin_x)

    # -------------------------------------
    # Plotting with matplotlib and Cartopy.
    # -------------------------------------
    plt.pcolormesh(x, y, prob, transform=ccrs.PlateCarree(), cmap=cm.get_cmap(cmap), zorder=0)
    # Adding colorbar to plot, 0.9 of the fig height
    cbar = plt.colorbar(shrink=0.9)
    cbar.ax.get_yaxis().labelpad = 15
    # Creating colorbar label with col_variable attributes.
    col_label = 'Probability (%)'
    # Add colobar label with SI units.
    cbar.ax.set_ylabel(col_label, rotation=270)

    # Show plot.
    plt.show()

    # Return figure.
    return

##############################################################################
# Define map_property() function.


def map_property(self, bin_res, variable, stat, cmap):
    """
    Map binned property of particles on an orthographic
    projection of Earth's surface.

    The particle property is binned onto a 2-dimensional
    (x-y) grid before a specified statistic is computed
    with the values in each bin.

    Bidimensional binned statistic is computed with
    scipy.stats.binned_statistic_2d().

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    bin_res : numeric
        The resolution (degrees) of the grid on to which particle
        positions will be binned.
    statistic : string
        The statistic to be computed with binned values - options
        are 'mean', 'std', 'median', 'count', 'sum', 'min' or 'max'.
    cmap : string
        A colormap instance or registered colormap name.

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
    ax.add_feature(cartopy.feature.LAND, zorder=4, edgecolor='black', facecolor='white')
    # Add grid lines of constant latitude and longitude.
    ax.gridlines()

    # -----------------------------------------
    # Defining variables from trajectories obj.
    # -----------------------------------------
    # Defining lat and lon for all trajectories.
    lat = self.data.lat.values
    lon = self.data.lon.values
    # Defining variable array from specified {variable}.
    var = self.data[variable].values

    # ---------------------------------------------------------
    # Defining grid on which particle positions will be binned.
    # ---------------------------------------------------------
    # Finding the maximum and minimum values of lat and lon
    # to the nearest degree E/N.
    lat_max = np.ceil(np.nanmax(lat))
    lat_min = np.floor(np.nanmin(lat))

    lon_max = np.ceil(np.nanmax(lon))
    lon_min = np.floor(np.nanmin(lon))

    # Defining bin size with specified bin_res.
    dx = dy = bin_res

    # Defining bin edges in 1-dimensional arrays.
    bin_x = np.arange(lon_min, lon_max + dx, dx)
    bin_y = np.arange(lat_min, lat_max + dy, dy)

    # -------------------------------------
    # Computing statistic on binned values.
    # -------------------------------------
    # Using scipy to count the number of particle positions per bin
    stat = stats.binned_statistic_2d(x=lon.flatten(), y=lat.flatten(), values=var.flatten(), statistic=stat, bins=[bin_x, bin_y])

    # ---------------------------
    # Defining grid for plotting.
    # ---------------------------
    # Defining grid from 1-d bin edge arrays.
    y, x = np.meshgrid(bin_y, bin_x)

    # -------------------------------------
    # Plotting with matplotlib and Cartopy.
    # -------------------------------------
    plt.pcolormesh(x, y, stat.statistic, transform=ccrs.PlateCarree(), cmap=cm.get_cmap(cmap), zorder=0)
    # Adding colorbar to plot, 0.9 of the fig height
    cbar = plt.colorbar(shrink=0.9)
    cbar.ax.get_yaxis().labelpad = 15

    # Creating colorbar label with variable attributes.
    col_label = self.data[variable].attrs['standard_name'] + " (" + self.data[variable].attrs['units'] + ")"

    # Add colobar label with SI units.
    cbar.ax.set_ylabel(col_label, rotation=270)

    # Show plot.
    plt.show()

    # Return figure.
    return
