**********************************************************
Welcome to the National Solar Radiation Data Base (NSRDB)!
**********************************************************

This repository contains all of the methods for the NSRDB data processing pipeline.
You can read more about the NSRDB `here <https://nsrdb.nrel.gov/>`_ and
`here <https://www.sciencedirect.com/science/article/pii/S136403211830087X>`_.
For details on NSRDB variable units, datatypes, and attributes, see the
`NSRDB variable meta data <https://github.com/NREL/nsrdb/blob/master/config/nsrdb_vars.csv>`_.

## The PXS All-Sky Irradiance Model
The PXS All-Sky Irradiance Model is the main physics package that calculates
surface irradiance variables. The code base and additional documentation can
be found `here <https://github.com/NREL/nsrdb/tree/master/nsrdb/all_sky>`_.

## The NSRDB Data Model
The NSRDB Data Model is the data aggregation framework that sources, processes,
and prepares data for input to All-Sky. The code base and additional
documentation can be found `here <https://github.com/NREL/nsrdb/tree/master/nsrdb/data_model>`_.

Installation
============

1. Use conda (anaconda or miniconda with python 3.7) to create an nsrdb
   environment: ``conda create --name nsrdb python=3.7``
2. Activate your new conda env: ``conda activate nsrdb``
3. Navigate to the nsrdb directory that contains setup.py and run:
   ``pip install -e .``
4. There is a known dependency issue between h5py and netCDF4. If you get an
   error using the netCDF4 module, try installing from ``conda`` instead of
   ``pip``

    a. ``pip uninstall netCDF4``

    b. ``conda install netCDF4``

5. Test your installation:

    a. Start ipython and test the following import:
       ``from nsrdb.data_model import DataModel``

    b. Navigate to the tests/ directory and run the command: ``pytest``

6. If you are a developer, also run `pre-commit install` in the directory
   containing .pre-commit-config.yaml.

NOTE: If you are trying to run the ``albedo`` sub-module you will need to
install the following additional packages:

    a. ``conda install hdf4``

    b. ``conda install -c conda-forge pyhdf``

NSRDB Versions
==============

.. list-table:: NSRDB Verions History
    :widths: auto
    :header-rows: 1

    * - Version
      - Effective Date
      - Data Years*
      - Notes
    * - 1.0.0
      - 2015
      - 2005-2012
      - Initial release of PSM v1 (no FARMS)

        - Satellite Algorithm for Shortwave Radiation Budget (SASRAB) model
        - MMAC model for clear sky condition
        - The DNI for cloud scenes is then computed using the DISC model

    * - 2.0.0
      - 2016
      - 1998-2015
      - Initial release of PSM v2 (use of FARMS, downscaling of ancillary data
        introduced to account for elevation, NSRDB website distribution
        developed)

        - Clear sky: REST2, Cloudy sky: NREL FARMS model and DISC model
        - Climate Forecast System Reanalysis (CFSR) is used for ancillary data
        - Monthly 0.5º aerosol optical depth (AOD) for 1998-2014 using
          satellite and ground-based measurements. Monthly results interpolated
          to daily 4-km AOD data. Daily data calibrated using ground
          measurements to develop accurate AOD product.

    * - 3.0.0
      - 2018
      - 1998-2017
      - Initial release of PSM v3

        - Hourly AOD (1998-2016) from Modern-Era Retrospective analysis for
          Research and Applications Version 2 (MERRA2).
        - Snow-free Surface Albedo from MODIS (2001-2015), (MCD43GF CMG
          Gap-Filled Snow-Free Products from University of Massachusetts,
          Boston).
        - Snow cover from Integrated Multi-Sensor Snow and Ice Mapping System
          (IMS) daily snow cover product (National Snow and Ice Data Center).
        - GOES-East time-shift applied to cloud properties instead of solar
          radiation.
        - Modern-Era Retrospective analysis for Research and Applications,
          Version 2 (MERRA-2) is used for ancillary data (pressure, humidity,
          wind speed etc.)

    * - 3.0.1
      - 2018
      - 2017+
      - Moved from timeshift of radiation to timeshift of cloud properties.
    * - 3.0.2
      - 2/25/2019
      - 1998-2017
      - Air temperature data recomputed from MERRA2 with elevation correction
    * - 3.0.3
      - 2/25/2019
      - 1998-2017
      - Wind data recomputed to fix corrupted data in western extent
    * - 3.0.4
      - 3/29/2019
      - 1998-2017
      - Aerosol optical depth patched with physical range from 0 to 3.2
    * - 3.0.5
      - 4/8/2019
      - 1998-2017
      - Cloud pressure attributes and scale/offset fixed for 2016 and 2017
    * - 3.0.6
      - 4/23/2019
      - 1998-2017
      - Missing data for all cloud properties gap filled using heuristics method
    * - 3.1.0
      - 9/23/2019
      - 2018+
      - Complete refactor of NSRDB processing code for NSRDB 2018
    * - 3.1.1
      - 12/5/2019
      - 2018+, TMY/TDY/TGY-2018
      - Complete refactor of TMY processing code
    * - 3.2.0
      - 3/17/2021
      - 2020
      - Enabled cloud solar shading coordinate adjustment by default, enabled
        MLClouds machine learning gap fill method for missing cloud properties
        (cloud fill flag #7)

*Note: The “Data Years” column shows which years of NSRDB data were updated at
the time of version release. However, each NSRDB file should be checked for the
version attribute, which should be a more accurate record of the actual data
version.
