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

VARS = ('air_temperature', 'ozone', 'surface_pressure',
        'total_precipitable_water', 'wind_direction', 'wind_speed')


@pytest.mark.parametrize(('var', 'date'),
                         list(zip(VARS, (DATE_20, DATE_21) * len(VARS))))
def test_ancillary_single(var, date):
    """Test GFS processed variables"""
    grid = os.path.join(TESTDATADIR, 'reference_grids/', 'west_psm_extent.csv')

    # set test directory
    source_dir = os.path.join(TESTDATADIR, 'gfs_source_files')
    var_meta = os.path.join(source_dir, 'nsrdb_vars_gfs.csv')
    factory_kwargs = {var: {'source_dir': source_dir, 'handler': 'GfsVar'}}

    DataModel.run_single(var, date, grid, var_meta=var_meta,
                         factory_kwargs=factory_kwargs, scale=False)
