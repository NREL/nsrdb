# Welcome to the National Solar Radiation Data Base (NSRDB)!
This repository contains all of the methods for the NSRDB data processing pipeline.
You can read more about the NSRDB [here](https://nsrdb.nrel.gov/) and [here](https://www.sciencedirect.com/science/article/pii/S136403211830087X).
For details on NSRDB variable units, datatypes, and attributes, see the [NSRDB variable meta data](https://github.nrel.gov/PXS/nsrdb/blob/master/config/nsrdb_vars.csv).

## The PXS All-Sky Irradiance Model
The PXS All-Sky Irradiance Model is the main physics package that calculates surface irradiance variables.
The code base and additional documentation can be found [here](https://github.nrel.gov/PXS/nsrdb/tree/master/nsrdb/all_sky).

## The NSRDB Data Model
The NSRDB Data Model is the data aggregation framework that sources, processes, and prepares data for input to All-Sky.
The code base and additional documentation can be found [here](https://github.nrel.gov/PXS/nsrdb/tree/master/nsrdb/data_model).

## Installation
1. Use conda (anaconda or miniconda with python 3.7) to create an nsrdb environment: `conda create --name nsrdb python=3.7`
2. Activate your new conda env: `conda activate nsrdb`
3. A few packages cannot be installed using pypi and must be installed using conda. Run these commands:

    a. `conda install hdf4`

    b. `conda install -c conda-forge pyhdf`

    c. `conda install netCDF4`

4. Navigate to the nsrdb directory that contains setup.py and run: `pip install -e .`
5. There is a known dependency issue between h5py and netCDF4. If you get an error using the netCDF4 module, make sure you ran the conda install of netCDF4: `conda install netCDF4`
6. Test your installation:

    a. Start ipython and test the following import: `from nsrdb.data_model import DataModel`

    b. Navigate to the tests/ directory and run the command: `pytest`
7. If you are a developer, also run `pre-commit install` in the directory containing .pre-commit-config.yaml.
