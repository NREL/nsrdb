# -*- coding: utf-8 -*-
"""A framework for handling global forecasting system (GFS) source data as a
real-time replacement for MERRA."""
import logging
import numpy as np
import pandas as pd

from nsrdb.data_model.base_handler import AncillaryVarHandler

logger = logging.getLogger(__name__)


class GfsVar(AncillaryVarHandler):
    """Framework for GFS source data extraction."""

    def __init__(self, name, var_meta, date, source_dir=None):
        """
        Parameters
        ----------
        name : str
            NSRDB var name.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        date : datetime.date
            Single day to extract data for.
        source_dir : str | None
            Optional data source directory. Will overwrite the source directory
            from the var_meta input.
        """

        self._grid = None
        super().__init__(name, var_meta=var_meta, date=date,
                         source_dir=source_dir)

    @property
    def time_index(self):
        """Get the GFS native time index.

        Returns
        -------
        ti : pd.DatetimeIndex
        """
        return None

    @property
    def files(self):
        """Get GFS file list.

        Returns
        -------
        files : list
        """
        return []

    def pre_flight(self):
        """Perform pre-flight checks - source file check.

        Returns
        -------
        missing : str
            Look for the source file and return the string if not found.
            If nothing is missing, return an empty string.
        """

        return ''

    @property
    def source_data(self):
        """Get a flat (1, n) array of data for a single day of MAIAC AOD.

        Returns
        -------
        data : np.ndarray
            2D numpy array (1, n) of MAIAC data for the specified var for a
            given day.
        """

        return None

    @property
    def grid(self):
        """Return the GFS source meta data with elevation in meters

        Returns
        -------
        self._grid : pd.DataFrame
            gfs source coordinates (latitude, longitude) with elevation
        """

        if self._grid is None:
            from pyhdf.SD import SD
            fp = self.files[0]
            res = SD(fp)
            attrs = res.attributes()

            lat_0 = attrs['FIRST LATITUDE']
            lon_0 = attrs['FIRST LONGITUDE']
            lat_res = attrs['LATITUDE RESOLUTION']
            lon_res = attrs['LONGITUDE RESOLUTION']
            lon_n = attrs['NUMBER OF LONGITUDES']
            lat_n = attrs['NUMBER OF LATITUDES']

            longitude = [lon_0 + (lon_res * i) for i in range(lon_n)]
            latitude = [lat_0 + (lat_res * i) for i in range(lat_n)]

            longitude, latitude = np.meshgrid(longitude, latitude)
            longitude = longitude.flatten()
            latitude = latitude.flatten()
            longitude[(longitude > 180)] -= 360
            elevation = res.select('surface height')[:].flatten() * 1000
            self._grid = pd.DataFrame({'longitude': longitude,
                                       'latitude': latitude,
                                       'elevation': elevation})

        return self._grid
