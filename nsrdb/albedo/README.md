# Welcome to the NSRDB Albedo Library!

The surface albedo dataset is a measure of the ground reflectivity. 
The albedo parameter is comprised of a slow-changing land-based albedo parameter from the 
[MODIS Dataset](https://modis.gsfc.nasa.gov/data/dataprod/mod43.php), 
and the daily [IMS Snow Dataset](https://nsidc.org/data/g02156). 
Both datasets are available at a high spatial resolution close to the final NSRDB resolution, so no spatial interpolation is required. 
The MODIS Dataset is on an 8-day temporal resolution, which is paired with the IMS daily snow cover, resulting in a daily albedo timeseries. 

## Albedo Processing Steps for the NSRDB

The processing of surface albedo has been updated for the 2018 NSRDB. The new process is detailed in the following steps:

1. The MODIS data is retrieved and the .hdf files are converted to .h5 using the [NSRDB file utilities](https://github.nrel.gov/PXS/nsrdb/blob/master/nsrdb/utilities/file_utils.py). 
    a. As of 4/19, the MCD43GF MODIS dataset is only available up to 2015. The 2015 dataset is used for the NSRDB 2016-2018 datasets. 
2. The IMS snow data is retrieved. 
3. The full year of IMS snow data is compiled into a single .h5 file so that temporal gap filling can be performed. 
    a. The methods used are year_1k_to_h5() and gap_fill_ims() found in the [IMS snow module](https://github.nrel.gov/PXS/nsrdb/blob/master/nsrdb/albedo/ims_snow.py). 
4. The 1km MODIS albedo data is updated with a fixed snow albedo value for coordinates that have snow based 1km IMS data. 
    a. This is done using the [map_modis() code](https://github.nrel.gov/PXS/nsrdb/blob/master/nsrdb/albedo/map_ims_to_modis.py).