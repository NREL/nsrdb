================================
Welcome to the NSRDB Data Model!
================================

The NSRDB Data Model is the data aggregation framework that sources,
processes, and prepares data for input to All-Sky. All source data is
ultimately processed to match the NSRDB spatiotemporal resolution through
interolation and mapping methods.

Data Sources
------------

Albedo
~~~~~~
The surface albedo dataset is a measure of the ground reflectivity.
The albedo parameter is comprised of a slow-changing land-based albedo
parameter from the `MODIS Dataset <https://modis.gsfc.nasa.gov/data/dataprod/mod43.php>`_,
and the daily `IMS Snow Dataset <https://nsidc.org/data/g02156>`_. Both datasets
are available at a high spatial resolution close to the final NSRDB resolution,
so no spatial interpolation is required. The MODIS Dataset is on an 8-day
temporal resolution, which is paired with the IMS daily snow cover, resulting
in a daily albedo timeseries.

Asymmetry
~~~~~~~~~
The aerosol asymmetry parameter is a fractional measure of the tendancy for
aerosols to scatter light in the forward vs. reverse direction. The NSRDB
asymmetry data source is a climatology dataset that has a very coarse spatial
and temporal resolution. The NSRDB data model uses the asymmetry at the closest
coordinate and timestep for input to all-sky.

GOES Satellite Cloud Data
~~~~~~~~~~~~~~~~~~~~~~~~~
Cloud data is retrieved from the east and west `GOES satellites <https://www.nasa.gov/content/goes>`_,
and is processed by the University of Wisconsin to retrieve irradiance properties
`(paper) <https://journals.ametsoc.org/doi/pdf/10.1175/1520-0450%281980%29019%3C1005%3AASPMTE%3E2.0.CO%3B2>`_.
The NSRDB processes the cloud data by "re-gridding" the cloud data onto the
NSRDB grid. The GOES cloud data is ultimately what limits the spatiotemporal
resolution of the NSRDB, and the source cloud data matches the resolution of
the NSRDB.

MERRA2
~~~~~~~
Much of the NSRDB ancillary data (aerosols, precipitable water, ozone, wind
speeds, etc...) comes from the `NASA MERRA2 Dataset <https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/>`_.
The MERRA2 data is available at a 40-km 1-hr resolution. Spatiotemporal
interpolation is performed on the data to match the NSRDB resolution.