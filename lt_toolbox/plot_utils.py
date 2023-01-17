##############################################################################
# plot_utils.py
#
# Description:
# Defines functions for plotting attribute variables for trajectories objects.
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
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from .add_utils import add_id

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
# Define plot_timeseries() method.


def plot_timeseries(self, variable, col_variable=None):
    """
    Plots time series of specified attribute variable as it
    evolves along each particle's trajectory.

    Time series can also be optionally coloured according to
    a specified (1-dimensional) scalar variable given by
    col_variable.

    When col_variable is not specified, the trajectory id of
    each time series is included in a legend.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    variable : string
        Name of the variable in the trajectories object.
    col_variable : string
        Name of variable in the trajectories object to colour
        plotted trajectories - must be 1-dimensional - default
        is None.

    Returns
    -------
    """
    # -------------------
    # Raising exceptions.
    # -------------------
    if isinstance(variable, str) is False:
        raise TypeError("variable must be specified as a string")

    if (isinstance(col_variable, str) or col_variable is None) is False:
        raise TypeError("col_variable must be specified as a string")

    # ----------------------------------------
    # Configuiring figure dimensions and axes.
    # ----------------------------------------
    # Initialising figure.
    plt.figure(figsize=(8, 4))
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
        # Plot time series for a single trajectory.
        if traj == 1:
            # Defining trajectory id, i.
            i = self.data.id.values[0]
            # Plot time series of var with default colours specififed
            # by matplotlib.
            ax.plot(time[0, :], var[0, :], linewidth=2, label='$%i$' % i)

        # Plot time series for all trajectories, traj.
        else:
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

    return

##############################################################################
# Define plot_ts_diagram() method.

def plot_ts_diagram(self, col_variable=None):
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
        scatter points - must be 1-dimensional - default
        is None.

    Returns
    -------
    """
    # -------------------
    # Raising exceptions.
    # -------------------
    if (isinstance(col_variable, str) or col_variable is None) is False:
        raise TypeError("col_variable must be specified as a string")

    # ----------------------------------------
    # Configuiring figure dimensions and axes.
    # ----------------------------------------
    # Initialising figure.
    plt.figure(figsize=(10, 10))
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

        # Defining colorbar, cbar, to be 0.6 of the fig height.
        cbar = plt.colorbar(sm, shrink=0.6)
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
# Define plot_variable() method.


def plot_variable(self, variable, plane, seed_level, time_level, cmap='coolwarm'):
    """
    2-dimensional Cartesian contour plot of a specified variable
    at a specific time along particle trajectories.

    Follows the specification of the trajectory map of Betten et
    al. (2017); values of the variable are displayed on particle
    initial grid locations at the time of seeding.

    When cmap is not specified, the default colour map is 'coolwarm'
    - a diverging colormap.

    Parameters
    ----------
    self : trajectories object
        Trajectories object passed from trajectories class method.
    variable : string
        Name of the variable in the trajectories object.
    plane : string
        Seeding plane from which particles are released - options
        are 'xz' zonal-vertical and 'yz' meridional-vertical.
    seed_level : integer
        Seeding level when particles are released.
    time_level : string
        Time level along trajectories to plot variable.
    cmap : string
        A colormap instance or registered colormap name.

    Returns
    -------
    """
    # -------------------
    # Raising exceptions.
    # -------------------
    if isinstance(variable, str) is False:
        raise TypeError("variable must be specified as a string")
    if isinstance(plane, str) is False:
        raise TypeError("plan must be specified as a string - options are \'xz\' or \'yz\'")
    if isinstance(seed_level, int) is False:
        raise TypeError("seed_level must be specified as an integer")
    if isinstance(time_level, str) is False:
        raise TypeError("time_level must be specified as a string in the format \'YYYY-MM-DD\'")
    if isinstance(cmap, str) is False:
        raise TypeError("cmap must be specified as a string")

    # ----------------------------------------
    # Configuiring figure dimensions and axes.
    # ----------------------------------------
    # Initialising figure.
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    # Adding grid lines.
    plt.grid(zorder=0)

    # ---------------------------------------------------
    # Defining DataSet from which to create contour plot.
    # ---------------------------------------------------
    # Add seed_levels to DataSet, filter on specified seed level.
    # Return starting locations lat, lon and z for contour plot.
    dataset = self.add_seed().filter_equal('seed_level', seed_level).get_start_loc()
    # Return specified variable at specified time_level as {variable}_i.
    dataset = dataset.get_value(variable, time_level)

    # -------------------------------------------------
    # Defining x- and y- variables for specified plane.
    # -------------------------------------------------
    # Zonal-Vertical.
    if plane == 'xz':
        x = dataset.lon_start.values
        x_label = dataset.lon_start.attrs['standard_name'] + ' (' + dataset.lon_start.attrs['units'] + ')'

        y = dataset.z_start.values
        y_label = dataset.z_start.attrs['standard_name'] + ' (' + dataset.z_start.attrs['units'] + ')'

    # Meridional-Vertical.
    else:
        x = dataset.lat_start.values
        x_label = dataset.lat_start.attrs['standard_name'] + ' (' + dataset.lat_start.attrs['units'] + ')'

        y = dataset.z_start.values
        y_label = dataset.z_start.attrs['standard_name'] + ' (' + dataset.z_start.attrs['units'] + ')'

    # ----------------------------------------
    # Defining z variable for specified plane.
    # ----------------------------------------
    # Defining standard_name of the {variable].
    var_name = variable + '_i'
    # Defining z as the values of {variable} at the specified time-level.
    z = dataset.data[var_name].values
    z_label = dataset.data[variable].attrs['standard_name'] + ' (' + dataset.data[variable].attrs['units'] + ')'

    # ---------------------------------------------
    # Defining 2D grid of x and y for contour plot.
    # ---------------------------------------------
    # Neglecting repeated values with np.unique.
    X, Y = np.meshgrid(np.unique(x), np.unique(y))

    # Defining no. rows and cols for both 2D grids.
    rows = np.shape(X)[0]
    cols = np.shape(X)[1]

    # Defining no. trajectories, ntraj.
    ntraj = len(z)

    # Defining arrays to store indices of z on X-Y grid.
    X_ind = np.zeros(ntraj).astype(int)
    Y_ind = np.zeros(ntraj).astype(int)

    # -------------------------------------
    # Filling 2D grid Z with values from z.
    # -------------------------------------
    # Defining Z with dimensions (rows x cols) - same as X-Y grid.
    Z = np.empty([rows, cols])
    # Fill Z with NaN values.
    Z[:, :] = np.nan

    # Iterate over no. trajectories - find X and Y indices of
    # start locations and allocate the corresponding value of z.
    for i in range(0, ntraj):
        X_ind[i] = np.where(X[0, :] == x[i])[0]
        Y_ind[i] = np.where(Y[:, 0] == y[i])[0]
        Z[Y_ind[i], X_ind[i]] = z[i]

    # ---------------------------------------------------
    # Plotting contourplot of {variable} on seeding grid.
    # ---------------------------------------------------
    # Contour plot with colour map specified as cmap (str).
    cplot = ax.contourf(X, Y, Z, cmap=cm.get_cmap(cmap), zorder=4)

    # Adding axis labels.
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # ------------------------------
    # Adding colorbar to the figure.
    # ------------------------------
    # Defining colorbar, cbar, to be 0.6 of the fig height.
    cbar = fig.colorbar(cplot, shrink=0.6)
    cbar.ax.get_yaxis().labelpad = 15

    # Add colobar label with SI units.
    cbar.ax.set_ylabel(z_label, rotation=270)

    # Show plot.
    plt.show()

    # Return figure.
    return
