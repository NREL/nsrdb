=====================================
Welcome to the NSRDB Albedo Library!
=====================================

The surface albedo dataset is a measure of the ground reflectivity.
The albedo parameter is comprised of a slow-changing land-based albedo parameter from the
`MODIS Dataset (Version 6) <https://lpdaac.usgs.gov/news/release-modis-version-6-brdfalbedo-gap-filled-snow-free-product/>`_
and the `IMS Snow Dataset <https://nsidc.org/data/g02156>`_.
Both datasets are available at a high spatial resolution close to the final NSRDB resolution, so no spatial interpolation is required.
The MODIS Dataset is paired with the IMS daily snow cover, resulting in a daily albedo time series.

Albedo Processing Steps for the NSRDB
---------------------------------------

The processing of surface albedo was updated in February 2020 . The new process is detailed in the following steps:

1. The MODIS data must currently be downloaded manually, as described below. As of 2/22, the MCD43GF MODIS dataset is only available up to 2017. The 2017 dataset is used for the all NSRDB datasets newer than 2017.
2. The IMS snow data is automatically retrieved.
3. The IMS data has four categories: dry land, open water, snow, sea ice. These categories are simplified to snow (snow and sea ice) or no-snow (dry land and open water).
4. The IMS data is filtered to only contain pixels that act as a boundary between regions of snow and no-snow. This greatly reduces memory needs and computation time.
5. The MODIS data is clipped to the extents of the IMS data. The following step are performed on the clipped MODIS data:
    - The nearest IMS pixel is determined for each MODIS pixel using lat-lon values.
    - If the nearest IMS pixel is categorized as snow the MODIS albedo value is updated to the SNOW_ALBEDO value in `albedo.py`. If the nearest IMS pixel is no-snow, the MODIS pixel retains its original value.
6. The clipped MODIS data is reintegrated into the original MODIS data, creating the composite albedo data.
7. The albedo data is scaled and exported to HDF5.


Downloading MODIS data
-----------------------

The current MODIS data requires a free `Earthdata <https://wiki.earthdata.nasa.gov/display/EL/Earthdata+Login+Home>`_ login account and must be downloaded manually. Files are available at the Earthdata `Data Pool <https://e4ftl01.cr.usgs.gov/MOTA/MCD43GF.006>`_. The albedo processing requires the shortwave, white-sky albedo datasets, which have the filename format `MCD43GF_wsa_shortwave_113_2017_V006.hdf`.

`Directions <https://wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+cURL+And+Wget>`_ are available to download the MODIS data via curl and wget. Additionally, the provided Python script `modis_v6_download.py` can be used to automate downloading MODIS data after the `.netrc` and `.urs_cookies` files are setup per the directions above.
