# -*- coding: utf-8 -*-
"""A framework for handling source data in the NREL resource format: .h5 source
files with meta and time_index datasets, all data is (n_time, n_sites).
"""
import os
import logging
import numpy as np

from nsrdb.data_model.base_handler import AncillaryVarHandler
from nsrdb.file_handlers.filesystem import NSRDBFileSystem as NFS

logger = logging.getLogger(__name__)


class NrelVar(AncillaryVarHandler):
    """Framework for NREL source data extraction."""

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
        self._time_index = None
        self._row_mask = None
        super().__init__(name, var_meta=var_meta, date=date, **kwargs)

    @property
    def file(self):
        """Get the file path for the target NSRDB variable name.

        Returns
        -------
        str
        """

        flist = NFS(self.source_dir).ls()
        flist = [f for f in flist if self.file_set in f]

        if len(flist) > 1:
            msg = ('Found multiple files for "{}" with file set "{}": {}'
                   .format(self.name, self.file_set, flist))
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not any(flist):
            msg = ('Could not find file for "{}" with file set "{}" in {}'
                   .format(self.name, self.file_set, self.source_dir))
            logger.error(msg)
            raise FileNotFoundError(msg)

        return os.path.join(self.source_dir, flist[0])

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
    def row_mask(self):
        """Get the boolean time_index (row) mask for the single day.

        Returns
        -------
        np.ndarray
        """

        if self._row_mask is None:
            with NFS(self.file, use_rex=True) as f:
                ti = f.time_index

            self._row_mask = ((ti.year == self._date.year)
                              & (ti.month == self._date.month)
                              & (ti.day == self._date.day))

        return self._row_mask

    @property
    def time_index(self):
        """Get the NREL source file native time index masked for the
        day of interest.

        Returns
        -------
        pd.DatetimeIndex
        """

        if self._time_index is None:
            with NFS(self.file, use_rex=True) as f:
                self._time_index = f.time_index[self.row_mask]

        return self._time_index

    @property
    def source_data(self):
        """Get single variable data of shape (n_time, n_sites) from the
        NREL source file.

        Returns
        -------
        np.ndarray
        """

        with NFS(self.file, use_rex=True) as f:
            locs = np.where(self.row_mask)[0]
            row_slice = slice(locs[0], locs[-1] + 1)
            data = f[self.name, row_slice, :]

        return data

    @property
    def grid(self):
        """Return the NREL resource file coordinates with elevation.

        Returns
        -------
        pd.DataFrame
        """

        if self._grid is None:
            with NFS(self.file, use_rex=True) as f:
                self._grid = f.meta[['latitude', 'longitude', 'elevation']]

        return self._grid
