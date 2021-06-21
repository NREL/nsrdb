# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import os
import pytest
import numpy as np
import pandas as pd
import h5py
import datetime

from nsrdb import TESTDATADIR, DEFAULT_VAR_META, DATADIR
from nsrdb.data_model import DataModel, VarFactory
from rex.utilities.loggers import init_logger


RTOL = 0.01
ATOL = 0.0

DATE_20 = datetime.date(year=2021, month=6, day=20)
DATE_21 = datetime.date(year=2021, month=6, day=21)


def test_ancillary_single(var, date):
    """Test GFS processed variables"""

    grid = os.path.join(TESTDATADIR, 'reference_grids/', 'west_psm_extent.csv')

    # set test directory
    source_dir = os.path.join(TESTDATADIR, 'gfs_source_files')

    factory_kwargs = {'TMP_surface': {'source_dir': source_dir,
                                      'handler': 'GfsVar'}}

    data = DataModel.run_single(var, date, grid, factory_kwargs=factory_kwargs)


if __name__ == '__main__':
    init_logger('nsrdb.data_model', log_file=None, log_level='DEBUG')
    test_ancillary_single('TMP_surface', DATE_20)
    test_ancillary_single('TMP_surface', DATE_21)
