

<br />
<p align="center">
    <img src="docs/images/README_LT_Toolbox_Logo_Dark.png" alt="Logo" width="450" height="120">
  </a>

  <h3 align="center">Lagrangian Trajectories Toolbox</h3>

  <p align="center">
    Post-Processing, Visualisation & Analysis of Lagrangian trajectories
    </a>
    <br />
    <br />
    ·
    <a href="https://oj-tooth.github.io/lt_toolbox/index.html"><strong>Explore the docs</strong></a>
    ·
    <a href="https://github.com/oj-tooth/lt-toolbox/issues"><strong>Report an issue</strong></a>
    ·
  </p>
</p>

<!-- Table of Contents -->
### Table of Contents

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

The Lagrangian Trajectories Toolbox is a Python library dedicated to the post-processing, visualisation and analysis of Lagrangian particle trajectories. 

The toolbox introduces the Trajectory Frames (TrajFrames) data structure to work with Lagrangian particle trajectories. TrajFrames make use of the blazingly fast [polars](https://pola-rs.github.io/polars-book/user-guide/index.html) DataFrame library to store attribute variables in a tabular format. Traditionally, the data structure used to store trajectories has been determined by the Lagrangian trajectory code used to evaluate particle trajectories. However, TrajFrames are well suited for working with both large .csv files generated from [TRACMASS](https://www.tracmass.org) and .zarr files output from [OceanParcels](https://oceanparcels.org).

Beyond simply storing Lagrangian data, TrajFrames contain a wide range of built-in methods to enable the efficient post-processing and visualisation of particle trajectories with [plotly](https://plotly.com/python-api-reference/index.html), [matplotlib](https://matplotlib.org) and [Cartopy](https://scitools.org.uk/cartopy/docs/latest/).

Current Features of the LT Toolbox:
-----------------------------------

* **Store** the attribute variables of Lagrangian trajectories in a TrajFrame object.

* **Add** new variables, such as transit times and seeding levels, to your TrajFrame.

* **Filter** trajectories using any attribute variable contained in your TrajFrame.

* **Get** existing features, including trajectory start/end times, start/end locations and durations.

* **Compute** metrics, such as distance travelled, binned-statistics and Lagrangian probabilities from trajectories.

* **Plot** trajectories, properties and probability distributions in the form of maps, time series, temperature-salinity diagrams and more.

<!-- Getting Started -->
## Getting Started

Below are our simple installation intructions to quickly get started with the LT Toolbox locally.

### Installation

The LT Toolbox is currently under development and is awaiting packaging for formal release. A temporary installation method is provided below.

To install the current development version of the LT Toolbox, users are strongly recommended to create a new virtual environment.

Next, run the following command to install the package into your virtual environment:

```sh 
pip install git+https://github.com/oj-tooth/lt_toolbox.git
```

<!-- Usage -->
## Usage

To learn more about how to use the LT Toolbox in your own projects see our [**tutorials**](https://oj-tooth.github.io/lt_toolbox/tutorials.html).
    
For further details on each of the modules included in the LT Toolbox view our [**API Reference**](https://oj-tooth.github.io/lt_toolbox/reference.html).

## Example

Below we show an example of how we can quickly calculate and plot a Lagrangian probability map from a collection of water parcel trajectories evaluated using a 1/12 degree ocean general circulation model ([ORCA0083-GO8p7 JRA55-do](https://dx.doi.org/10.5285/399b0f762a004657a411a9ea7203493a) simulation):

```python
# Importing LT Toolbox after installing with pip:
import lt_toolbox as ltt

# Defining filepath to our example ORCA0083-GO8p7 output trajectory file:
traj_filepath = "./data/ORCA0083-GO8p7_JRA55_SPNA_1995_example.parquet"

# Open output .parquet file as a DataFrame.
dataset = pl.read_parquet(traj_filepath, use_pyarrow=True)

# Create a TrajFrame from the DataFrame:
traj = ltt.TrajFrame(source=dataset, condense=True)
```

Next, let's plot a Lagrangian probability map using our TrajFrame:

```python
# Creating a map of the binned probability of all water parcel pathways.
traj.compute_probability(bin_res=0.25, # Bin resolution in degrees
                         prob_type='traj', # Type of Lagrangian probability map.
                         group=None, # Do not group trajectories - return single map.
                         append=False, # Replace existing data in summary Dataset.
                         )

# Plotting the Lagrangian probability in discrete 2-D longitude-latitude bins:
traj.summary_data.probability.plot()
```

<p align="centre">
    <img src="docs/images/Figure3_Example_Probability.png" alt="Fig3" width="300" height="200"> 
 </a>
<p


<!-- License -->
## License

Distributed under the MIT License. See LICENSE file for further details. 

<!-- Contact -->
## Contact

The LT Toolbox is developed and maintained by Ollie Tooth whilst at the University of Oxford, UK. I have since moved to the National Oceangraphy Centre, UK, so if you would like to get in contact about this project, please email: **oliver.tooth@noc.ac.uk**.

<!-- Acknowledgements -->
## Acknowledgements

* Adapted README.md from opensource template: 
https://github.com/othneildrew/Best-README-Template/blob/master/README.md

* LT-Toolbox logo was created with the help of Noam Vogt-Vincent (@nvogtvincent) at the University of Oxford.

* The example ORCA1-N406 Lagrangian trajectories were evaluated using open-source output made available by the [**TRACMASS**](https://www.tracmass.org) project.
