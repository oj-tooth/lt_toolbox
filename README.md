

<br />
<p align="center">
    <img src="docs/images/README_LT_Toolbox_Logo.png" alt="Logo" width="400" height="200">
  </a>

  <h3 align="center">LT Toolbox</h3>

  <p align="center">
    The Lagrangian Trajectories Toolbox
    <br />
    <a href="https://lt_toolbox.readthedocs.io"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href=https://github.com/oj-tooth/lt_toolbox/#example>View Demo</a>
    ·
    <a href="https://github.com/oj-tooth/lt-toolbox/issues">Report Issue</a>
    ·
  </p>
</p>

<!-- Table of Contents -->
## Table of Contents

* [About the LT Toolbox](#about-the-lt-toolbox)
  * [Background](#background)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
  * [Example](#example)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

<!-- About the LT Toolbox -->
## About The Lagrangian Trajectories Toolbox

The Lagrangian Trajectories Toolbox is a Python library dedicated to the post-processing, visualisation and analysis of Lagrangian water parcel trajectories. 

The toolbox offers users two data structures to work with Lagrangian water parcel trajectories: Trajectory Arrays (TrajArrays) and Trajectory Frames (TrajFrames).
TrajArrays utilise [xarray](http://xarray.pydata.org/en/stable/#) multidimensional data arrays to store attribute variables (e.g. lat, lon, temperature etc.) evaluated along trajectories. Meanwhile, TrajFrames make use of the blazingly fast [polars](https://pola-rs.github.io/polars-book/user-guide/index.html) DataFrame library to store attribute variables in a tabular format. The choice of whether to use a TrajArray of TrajFrame in your analysis is often determined by the Lagrangian trajectory code used to evaluate your water parcel trajectories in the first place. For example, TrajArrays are perfect for working with .zarr files output from [OceanParcels](https://oceanparcels.org), whereas TrajFrames are best suited for working with large .csv files generated from [TRACMASS](https://www.tracmass.org)

Beyond simply storing Lagrangian data, TrajArrays and TrajFrames contains a wide range of built-in methods to enable the efficient post-processing and visualisation of thousands of trajectories with both [matplotlib](https://matplotlib.org) and [Cartopy](https://scitools.org.uk/cartopy/docs/latest/).

#### Current Features:

+ **Store** the attribute variables of Lagrangian trajectories in a TrajArray or TrajFrame object.

+ **Add** new variables, such as particle IDs and seeding levels, to your TrajArray or TrajFrame.

+ **Filter** trajectories using any attribute variable contained in your TrajArray or TrajFrame.

+ **Get** existing features, including trajectory start/end times, start/end locations and durations.

+ **Compute** metrics, such as distance travelled, particle displacements, velocities and Lagrangian probabilities from trajectories.

+ **Plot** trajectory data in the form of time series, temperature-salinity diagrams and more.

+ **Map** trajectories, properties and probability distributions onto the Earth's surface using Cartopy.

### Background

At the centre of the LT Toolbox are the TrajArray & TrajFrame objects. At their simplest, TrajArrays and TrajFrames represent containers for xarray DataSets and polars DataFrames respectively. This means that users always have complete access to their original trajectory data and also all of xarray's and polar's existing functionality.

By looking at an example TrajArray (Fig. 1), we can see that our original DataSet must be 2-dimensional with dimensions traj (trajectory - representing one particle / float) and obs (observation - representing one time-level). This formatting follows the [NCEI template](https://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl) for trajectory data, in which attribute variables for each particle / float are stored as a function of a series of observations (obs).

<p align="centre">
    <img src="" alt="Logo" width="220" height="150"> 
 </a>
<p


For improved functionality, all attribute variables stored inside our DataSet are made accessible with the command:

```sh
traj.{variable}
```
where traj is our trajectories object and {variable} refers to the name of any attribute variable in our original DataSet. 

The greatest value of the trajectories object comes with the use of the built-in functions specifically designed for post-processing, visualisation and analysis of Lagrangain water parcel trajectories. Below offers a great example of how, in only one line, the user can add an ID attribute variable, filter trajectories according to their distinct ID and map them onto the Earth's surface:

```sh
traj.add_id().filter_between('id', 1, 100).map_trajectories()
```

<p align="center">
    <img src="" alt="Logo" width="250" height="200"> 
 </a>
<p

<!-- Getting Started -->
## Getting Started

Below are our simple installation intructions to quickly get started with the LT Toolbox locally.

### Installation

The LT Toolbox is currently under development and is awaiting packaging for formal release.

```sh 
```

<!-- Usage -->
## Usage

The LT Toolbox is structured as follows:
    
For further details on each of the modules included in the LT Toolbox view our [docs](https://lt_toolbox.readthedocs.io).

## Example

<!-- License -->
## License

Distributed under the MIT License. See LICENSE file for further details. 

<!-- Acknowledgements -->
## Acknowledgements

* Adapted README.md from opensource template: 
https://github.com/othneildrew/Best-README-Template/blob/master/README.md
