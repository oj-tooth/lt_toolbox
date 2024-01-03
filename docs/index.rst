Lagrangian Trajectories Toolbox
===============================

The Lagrangian Trajectories Toolbox is a Python library dedicated to the post-processing, visualisation and analysis of Lagrangian particle trajectories. 

The toolbox offers users two data structures to work with Lagrangian particle trajectories: Trajectory Arrays (TrajArrays) and Trajectory Frames (TrajFrames).
TrajArrays utilise `xarray <http://xarray.pydata.org/en/stable/#>`_` multidimensional data arrays to store attribute variables (e.g. lat, lon, temperature etc.) evaluated along trajectories. Meanwhile, TrajFrames make use of the blazingly fast `polars <https://pola-rs.github.io/polars-book/user-guide/index.html>`_ DataFrame library to store attribute variables in a tabular format. The choice of whether to use a TrajArray of TrajFrame in your analysis is often determined by the Lagrangian trajectory code used to evaluate your particle trajectories in the first place. For example, TrajArrays are best suited for working with .zarr files output from `OceanParcels <https://oceanparcels.org>`_, whereas TrajFrames are well suited for working with large .csv files generated from `TRACMASS <https://www.tracmass.org>`_.

Beyond simply storing Lagrangian data, TrajArrays and TrajFrames contain a wide range of built-in methods to enable the efficient post-processing and visualisation of particle trajectories with `plotly <https://plotly.com/python-api-reference/index.html>`_ `matplotlib <https://matplotlib.org>`_` and `Cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_.

Current Features of the LT Toolbox:
-----------------------------------

* **Store** the attribute variables of Lagrangian trajectories in a TrajArray or TrajFrame object.

* **Add** new variables, such as transit times and seeding levels, to your TrajArray or TrajFrame.

* **Filter** trajectories using any attribute variable contained in your TrajArray or TrajFrame.

* **Get** existing features, including trajectory start/end times, start/end locations and durations.

* **Compute** metrics, such as distance travelled, binned-statistics and Lagrangian probabilities from trajectories.

* **Plot** trajectories, properties and probability distributions in the form of maps, time series, temperature-salinity diagrams and more.

.. toctree::
   :maxdepth: 2
   :hidden:

   Home <self>
   Installation <installation>
   Tutorials <tutorials>
   API Reference <reference>
   Contributing <contributing>
