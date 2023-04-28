# -*- coding: utf-8 -*-
"""A framework for handling MAIAC high-res AOD source data."""
import logging
import numpy as np
import os
import pandas as pd

from nsrdb.data_model.base_handler import AncillaryVarHandler
from nsrdb.file_handlers.file_system import NSRDBFileSystem as NFS

logger = logging.getLogger(__name__)


class MaiacVar(AncillaryVarHandler):
    """Framework for MAIAC AOD source data extraction."""

    def __init__(self, name, var_meta, date, **kwargs):
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
        """

        self._grid = None
        super().__init__(name, var_meta=var_meta, date=date, **kwargs)

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
    def pattern(self):
        """Get the source file pattern which is sent to glob().

        Returns
        -------
        str
        """
        pat = super().pattern
        if pat is None:
            pat = os.path.join(self.source_dir, '*{}*'.format(self._date.year))

        return pat

    @property
    def file(self):
        """Get the file paths for the target NSRDB variable name based on
        the glob self.pattern.

        Returns
        -------
        list
        """

        fps = NFS(self.pattern).glob()
        if not any(fps):
            emsg = ('Could not find source files '
                    'for dataset "{}" with glob pattern: "{}". '
                    'Found {} files: {}'
                    .format(self.name, self.pattern, len(fps), fps))
            logger.error(emsg)
            raise FileNotFoundError(emsg)

        logger.debug('Found the following MAIAC files: {}'.format(fps))

        return fps

    @property
    def files(self):
        """Get multiple MAIAC AOD filepaths based on source_dir and year.

        Returns
        -------
        list
        """
        return self.file

    def pre_flight(self):
        """Perform pre-flight checks - source file check.

        Returns
        -------
        missing : str
            Look for the source file and return the string if not found.
            If nothing is missing, return an empty string.
        """
        for fp in self.files:
            with NFS(fp, use_rex=True) as res:
                dsets = res.dsets
                msg = 'Needs "{}" dset: {}'
                assert 'latitude' in dsets, msg.format('latitude', fp)
                assert 'longitude' in dsets, msg.format('longitude', fp)
                assert 'aod' in dsets, msg.format('aod', fp)
                shape_lat = res.get_dset_properties('latitude')[0]
                shape_lon = res.get_dset_properties('longitude')[0]
                shape_aod = res.get_dset_properties('aod')[0]
                assert shape_lat == shape_lon
                msg = '"aod" dset must be 3D (x,y,time)'
                assert len(shape_aod) == 3, msg
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
            with NFS(fp, use_rex=True) as res:
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
                with NFS(fp, use_rex=True) as res:
                    temp = pd.DataFrame(
                        {'longitude': res['longitude'].flatten(),
                            'latitude': res['latitude'].flatten()})
                    if self._grid is None:
                        self._grid = temp
                    else:
                        self._grid = self._grid.append(temp,
                                                       ignore_index=True)

            logger.debug('MAIAC AOD grid has {} coordinates'
                         .format(len(self._grid)))

        return self._grid
