# -*- coding: utf-8 -*-
"""NSRDB MODIS Albedo processing tools.

Adapted from Galen and Nick's original code:
    https://github.nrel.gov/dav-gis/pv_task/tree/dev/pv_task

@author: gbuster
"""

import os
import logging

from nsrdb.utilities.loggers import init_logger
from nsrdb.utilities.file_utils import url_download


logger = logging.getLogger(__name__)


def retrieve_modis(year, out_dir, log_level='DEBUG'):
    """Download MODIS albedo source data for a single year.

    Parameters
    ----------
    year : int | str
        Year to download modis data for (last year is currently 2015).
    out_dir : str
        Target path to download files to.
    """

    # initialize a logger output file for this method in the ims directory.
    init_logger(__name__, log_file=None, log_level=log_level)
    init_logger('nsrdb_utilities.file_utils', log_file=None,
                log_level=log_level)

    f_dir = 'ftp://rsftp.eeos.umb.edu/data02/Gapfilled/{year}/'
    fname_base = 'MCD43GF_wsa_shortwave_{day}_{year}.hdf'
    flink_base = os.path.join(f_dir, fname_base)
    days = [str(d).zfill(3) for d in range(1, 362, 8)]
    flinks = [flink_base.format(year=year, day=day) for day in days]
    dl_files = os.listdir(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    failed = []
    for url in flinks:
        dl_fname = url.split('/')[-1]

        if dl_fname in dl_files:
            logger.info('Skipping (already exists): {}'.format(dl_fname))
        else:
            logger.info('Downloading {}'.format(dl_fname))
            dfname = os.path.join(out_dir, dl_fname)
            fail = url_download(url, dfname)
            if failed:
                failed.append(fail)

    return failed
