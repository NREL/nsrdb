# -*- coding: utf-8 -*-
"""A framework for handling MAIAC high-res AOD source data."""

import numpy as np
import pandas as pd
import os
import logging

from nsrdb.file_handlers.resource import Resource
from nsrdb.data_model.base_handler import AncillaryVarHandler


logger = logging.getLogger(__name__)


class MaiacVar(AncillaryVarHandler):
    """Framework for MAIAC AOD source data extraction."""

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
    def doy_index(self):
        """Get the day of year index which is one less than doy (zero indexed)

        Returns
        -------
        index : int
            Zero-indexed doy (0-364 or 0-365)
        """
        return self.doy - 1

    @property
    def doy(self):
        """Get the day of year for daily MAIAC AOD data.

        Returns
        -------
        doy : int
            Day of year integer (1-365 or 1-366).
        """
        doy = self._date.timetuple().tm_yday
        return doy

    @property
    def time_index(self):
        """Get the aod native time index.

        Returns
        -------
        ti : pd.DatetimeIndex
            Pandas datetime index for the current day at the MAIAC
            resolution (1-day).
        """
        return self._get_time_index(self._date, freq='1D')

    @property
    def files(self):
        """Get multiple MAIAC AOD filepaths based on source_dir and year.

        Returns
        -------
        files : list
           List of filepaths to h5 files containing MAIAC AOD data.
        """
        fns = [fn for fn in os.listdir(self.source_dir)
               if str(self._date.year) in fn]
        fps = [os.path.join(self.source_dir, fn) for fn in fns]
        logger.debug('Found the following MAIAC files: {}'.format(fns))
        return fps

    def pre_flight(self):
        """Perform pre-flight checks - source file check.

        Returns
        -------
        missing : str
            Look for the source file and return the string if not found.
            If nothing is missing, return an empty string.
        """
        for fp in self.files:
            with Resource(fp) as res:
                dsets = res.dsets
                msg = 'Needs "{}" dset: {}'
                assert 'latitude' in dsets, msg.format('latitude', fp)
                assert 'longitude' in dsets, msg.format('longitude', fp)
                assert 'aod' in dsets, msg.format('aod', fp)
                shape_lat = res.get_dset_properties('latitude')[0]
                shape_lon = res.get_dset_properties('longitude')[0]
                shape_aod = res.get_dset_properties('aod')[0]
                assert shape_lat == shape_lon
                assert len(shape_aod) == 3, '"aod" dset must be 3D (x,y,time)'
                assert shape_aod[0] == shape_lat[0]
                assert shape_aod[1] == shape_lat[1]
                assert (shape_aod[2] == 365) | (shape_aod[2] == 366)

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

        L = 0
        data = []
        for fp in self.files:
            with Resource(fp) as res:
                logger.debug('Getting MAIAC aod from {}'
                             .format(os.path.basename(fp)))
                data.append(res['aod', :, :, self.doy_index].flatten())
                L += len(data[-1])

        logger.debug('Stacking {} MAIAC aod data arrays'.format(len(data)))
        data = np.hstack(data)
        data = np.expand_dims(data, axis=0)
        data[np.isnan(data)] = 0.0

        assert data.shape[1] == len(self.grid)

        return data

    @property
    def grid(self):
        """Return the MAIAC AOD source coordinates

        Returns
        -------
        self._grid : pd.DataFrame
            MAIAC source coordinates (latitude, longitude) without elevation
        """

        if self._grid is None:
            for fp in self.files:
                with Resource(fp) as res:
                    temp = pd.DataFrame(
                        {'longitude': res['longitude'].flatten(),
                         'latitude': res['latitude'].flatten()})
                    if self._grid is None:
                        self._grid = temp
                    else:
                        self._grid = self._grid.append(temp, ignore_index=True)

            logger.debug('MAIAC AOD grid has {} coordinates'
                         .format(len(self._grid)))

        return self._grid