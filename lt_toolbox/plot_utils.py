##############################################################################
# plot_utils.py
#
# Description:
# Defines functions for plotting attribute variables for
# trajectories objects.
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
from add_utils import add_id
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
# Define plot_trajectories() function.


def plot_trajectories(self, col_variable):
    """
    Plots surface trajectories (latitudes and longitudes) of
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
        plotted trajectories.

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

##############################################################################
# Define plot_timeseries() function.


def plot_timeseries(self, variable, col_variable):
    """
    Plots time series of specified attribute variable as it
    evolves along each particle's trajectory.

    Time series can also be optionally coloured according to
    a specified (1-dimensional) scalar variable given by
    col_variable.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    variable : string
        Name of the variable in the trajectories object.
    col_variable : string
        Name of variable in the trajectories object to colour
        plotted trajectories - must be 1-dimensional.

    Returns
    -------
    """
    # ----------------------------------------
    # Configuiring figure dimensions and axes.
    # ----------------------------------------
    # Initialising figure.
    plt.figure()
    ax = plt.axes()

    # -----------------------------------------
    # Defining variables from trajectories obj.
    # -----------------------------------------
    var = self.data[variable].values
    y_label = self.data[variable].attrs['standard_name'] + ' (' + self.data[variable].attrs['units'] + ')'

    time = self.data['time'].values
    x_label = self.data.time.attrs['standard_name']

    # -------------------------------------------
    # Subroutine for no specified color variable.
    # -------------------------------------------
    if col_variable is None:

        # Determine number of trajectories to plot.
        traj = np.shape(var)[0]

        # Defining list of variables contained in data.
        variables = list(self.data.variables)
        # Add trajectory id if not found in variables.
        if 'id' not in variables:
            self = self.add_id()

        # ------------------------------------------------------
        # Plotting time series of variable for all trajectories.
        # ------------------------------------------------------
        # Plot time series for all trajectories, traj.
        for n in np.arange(0, traj-1):
            # Defining trajectory id, i.
            i = self.data.id.values[n]
            # Plot time series of var with default colours specififed
            # by matplotlib.
            ax.plot(time[n, :], var[n, :], linewidth=2, label='$%i$' % i)

        # Return current axes positions.
        box = ax.get_position()
        # Reduce axes to 0.9 of original size.
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Add legend to the right of the axes, indicating traj no.
        ax.legend(title=r'$\bf{traj}$', loc='center left', bbox_to_anchor=(1, 0.5))
        # Adding axis labels.
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    # --------------------------------------------
    # Subroutine for 1-dimensional color variable.
    # --------------------------------------------
    else:

        # Defining col as attribute variable col_variable.
        col = self.data[col_variable].values

        # ---------------------------------------------
        # Normalise col_variable to colour trajectories.
        # ---------------------------------------------
        # Normalise col for colours using normalise().
        col_norm = normalise(col)

        # ------------------------------------------------------
        # Plotting time series of variable for all trajectories.
        # ------------------------------------------------------
        # Determine number of trajectories to plot.
        traj = np.shape(var)[0]

        # Defining color from col_norm with diverging coolwarm
        # colour map.
        color = cm.coolwarm(col_norm)

        # Plot time series for all trajectories, traj.
        for n in np.arange(0, traj-1):
            # Plot trajectories with single colour defined by color.
            ax.plot(time[n, :], var[n, :], c=color[n], linewidth=2)

        # Adding axis labels.
        plt.xlabel(x_label)
        plt.ylabel(y_label)

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
# Define plot_ts_diagram() function.


def plot_ts_diagram(self, col_variable):
    """
    Plots temperature-salinity diagram as a scatter plot of
    temp (y) and salinity (x) for every point along each
    particle's trajectory.

    Plotted points can be optionally coloured according to
    a specified (1-dimensional) scalar variable given by
    col_variable.

    When col_variable is not specified, points are coloured
    according to their trajectory id with an accompanying legend.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    col_variable : string
        Name of variable in the trajectories object to colour
        scatter points - must be 1-dimensional.

    Returns
    -------
    """
    # ----------------------------------------
    # Configuiring figure dimensions and axes.
    # ----------------------------------------
    # Initialising figure.
    plt.figure()
    ax = plt.axes()
    # Adding grid lines.
    plt.grid(zorder=0)

    # ------------------------------------------------------
    # Defining temp and sal variables from trajectories obj.
    # ------------------------------------------------------
    sal = self.data.sal.values
    x_label = self.data.sal.attrs['standard_name'] + ' (' + self.data.sal.attrs['units'] + ')'

    temp = self.data.temp.values
    y_label = self.data.temp.attrs['standard_name'] + ' (' + self.data.temp.attrs['units'] + ')'

    # -------------------------------------------
    # Subroutine for no specified color variable.
    # -------------------------------------------
    if col_variable is None:

        # Determine number of trajectories to plot.
        traj = np.shape(temp)[0]

        # Defining list of variables contained in data.
        variables = list(self.data.variables)
        # Add trajectory id if not found in variables.
        if 'id' not in variables:
            traj_id = add_id(self)

        # ------------------------------------------------------------
        # Plotting scatter plot of temp and sal for all trajectories.
        # ------------------------------------------------------------
        # Scatter plot for all trajectories, traj.
        for n in np.arange(0, traj-1):
            # Defining trajectory id, i.
            i = traj_id[n]
            # Plot scatter of sal (x) and temp (y) with default colours
            # specififed by matplotlib.
            ax.scatter(sal[n, :], temp[n, :], edgecolor='black', zorder=3, label='$%i$' % i)

        # Return current axes positions.
        box = ax.get_position()
        # Reduce axes to 0.9 of original size.
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Add legend to the right of the axes, indicating traj no.
        ax.legend(title=r'$\bf{traj}$', loc='center left', bbox_to_anchor=(1, 0.5))
        # Adding axis labels.
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    # --------------------------------------------
    # Subroutine for 1-dimensional color variable.
    # --------------------------------------------
    else:

        # Defining col as attribute variable col_variable.
        col = self.data[col_variable].values

        # ---------------------------------------------
        # Normalise col_variable to colour trajectories.
        # ---------------------------------------------
        # Normalise col for colours using normalise().
        col_norm = normalise(col)

        # ------------------------------------------------------
        # Plotting time series of variable for all trajectories.
        # ------------------------------------------------------
        # Determine number of trajectories to plot.
        traj = np.shape(temp)[0]

        # Defining color from col_norm with diverging coolwarm
        # colour map.
        color = cm.coolwarm(col_norm)

        # Plot scatter plot for all trajectories, traj.
        for n in np.arange(0, traj-1):
            # Plot scatter with single colour defined by color.
            ax.scatter(sal[n, :], temp[n, :], c=color[n], edgecolor='black', zorder=3)

        # Adding axis labels.
        plt.xlabel(x_label)
        plt.ylabel(y_label)

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