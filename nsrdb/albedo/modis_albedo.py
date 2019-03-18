# -*- coding: utf-8 -*-
"""NSRDB MODIS Albedo processing tools.

Adapted from Galen and Nick Gilroy's original code:
    https://github.nrel.gov/dav-gis/pv_task/tree/dev/pv_task

@authors: gbuster & ngilroy
"""

import os
import logging

from nsrdb.utilities.loggers import init_logger
from nsrdb.utilities.file_utils import url_download


logger = logging.getLogger(__name__)


class RetrieveMODIS:
    """Class to manage MODIS data retrieval"""

    def __init__(self, year, target_path):
        """
        Parameters
        ----------
        year : int | str
            Year to download modis data for (last year is currently 2015).
        target_path : str
            Target path to download files to.
        """

        self._year = year
        self._target_path = target_path

    def retrieve_data(self):
        """Retrieve MODIS albedo source data for a single year.

        Parameters
        ----------
        year : int | str
            Year to download modis data for (last year is currently 2015).
        target_path : str
            Target path to download files to.
        """

        f_dir = 'ftp://rsftp.eeos.umb.edu/data02/Gapfilled/{year}/'
        fname_base = 'MCD43GF_wsa_shortwave_{day}_{year}.hdf'
        flink_base = os.path.join(f_dir, fname_base)
        days = [str(d).zfill(3) for d in range(1, 362, 8)]
        flinks = [flink_base.format(year=self._year, day=day) for day in days]
        dl_files = os.listdir(self._target_path)

        if not os.path.exists(self._target_path):
            os.makedirs(self._target_path)

        failed = []
        for url in flinks:
            dl_fname = url.split('/')[-1]

            if dl_fname in dl_files:
                logger.info('Skipping (already exists): {}'.format(dl_fname))
            else:
                logger.info('Downloading {}'.format(dl_fname))
                dfname = os.path.join(self._target_path, dl_fname)
                fail = url_download(url, dfname)
                if failed:
                    failed.append(fail)

        return failed

    @classmethod
    def run(cls, year, target_path, log_level='INFO'):
        """Retrieve MODIS albedo source data for a single year.

        Parameters
        ----------
        year : int | str
            Year to download modis data for (last year is currently 2015).
        target_path : str
            Directory to save the downloaded files.
        Returns
        -------
        failed : list
            List of files that failed to download.
        """
        # initialize logger output file for this method in modis directory.
        init_logger(__name__, log_file=None, log_level=log_level)
        init_logger('nsrdb_utilities.file_utils', log_file=None,
                    log_level=log_level)

        modis = cls(year, target_path)
        failed = modis.retrieve_data()
        return failed
