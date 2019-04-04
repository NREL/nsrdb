# -*- coding: utf-8 -*-
"""A framework for handling Asymmetry source data."""

import pandas as pd
import h5py
import os
import logging
import datetime

from nsrdb.data_model.base_handler import AncillaryVarHandler

logger = logging.getLogger(__name__)


class AsymVar(AncillaryVarHandler):
    """Framework for Asymmetry variable data extraction."""

    def __init__(self, var_meta, name='asymmetry',
                 date=datetime.date(year=2017, month=1, day=1),
                 fname='asymmetry_clim.h5'):
        """
        Parameters
        ----------
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        name : str
            NSRDB var name.
        date : datetime.date
            Single day to extract data for.
        fname : str
            Asymmetry source data filename.
        """
        self._fname = fname
        super().__init__(var_meta, name, date)

    @property
    def fpath(self):
        """Get the Asymmetry source file path with file name."""
        return os.path.join(self.source_dir, self._fname)

    @property
    def time_index(self):
        """Get the MERRA native time index.

        Returns
        -------
        asym_ti : pd.DatetimeIndex
            Pandas datetime index for the current day at the asymmetry
            resolution (1-month).
        """
        return self._get_time_index(self._date, freq='1D')

    @property
    def source_data(self):
        """Get the asymmetry source data.

        Returns
        -------
        data : np.ndarray
            Single month of asymmetry data with shape (1 x n_sites).
        """

        with h5py.File(self.fpath, 'r') as f:
            # take the data at all sites for the zero-indexed month
            i = self._date.month - 1
            data = f[self.name][i, :]

        # reshape to (1 x n_sites)
        data = data.reshape((1, len(data)))

        return data

    @property
    def grid(self):
        """Get the asymmetry grid.

        Returns
        -------
        _asym_grid : pd.DataFrame
            Asymmetry grid data with columns 'latitude' and 'longitude'.
        """

        if not hasattr(self, '_asym_grid'):
            with h5py.File(self.fpath, 'r') as f:
                self._asym_grid = pd.DataFrame(f['meta'][...])

            if ('latitude' not in self._asym_grid or
                    'longitude' not in self._asym_grid):
                raise ValueError('Asymmetry file did not have '
                                 'latitude/longitude meta data. '
                                 'Please check: {}'.format(self.fpath))

        return self._asym_grid
