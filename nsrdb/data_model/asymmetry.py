# -*- coding: utf-8 -*-
"""A framework for handling Asymmetry source data."""
import datetime
import logging
import os

from nsrdb.data_model.base_handler import AncillaryVarHandler
from nsrdb.file_handlers.file_system import NSRDBFileSystem as NFS

logger = logging.getLogger(__name__)


class AsymVar(AncillaryVarHandler):
    """Framework for Asymmetry variable data extraction."""

    def __init__(self, name='asymmetry', var_meta=None,
                 date=datetime.date(year=2017, month=1, day=1),
                 fname='asymmetry_clim.h5', **kwargs):
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
        fname : str
            Asymmetry source data filename.
        """
        self._asym_grid = None
        self._fname = fname
        super().__init__(name, var_meta=var_meta, date=date, **kwargs)

    @property
    def pattern(self):
        """Get the source file pattern which is sent to glob().

        Returns
        -------
        str
        """
        pat = super().pattern
        if pat is None:
            pat = os.path.join(self.source_dir, self._fname)

        return pat

    def pre_flight(self):
        """Perform pre-flight checks - source file check.

        Returns
        -------
        missing : str
            Look for the source file and return the string if not found.
            If nothing is missing, return an empty string.
        """

        missing = ''
        if not NFS(self.file).isfile():
            missing = self.file

        return missing

    @property
    def time_index(self):
        """Get the asymmetry native time index.

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
        with NFS(self.file) as f:
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

        if self._asym_grid is None:
            with NFS(self.file, use_rex=True) as f:
                self._asym_grid = f.meta

            if ('latitude' not in self._asym_grid
                    or 'longitude' not in self._asym_grid):
                raise ValueError('Asymmetry file did not have '
                                 'latitude/longitude meta data. '
                                 'Please check: {}'.format(self.file))

        return self._asym_grid
