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
import datetime
import tempfile

from nsrdb import TESTDATADIR, CONFIGDIR
from nsrdb.data_model import DataModel, VarFactory
from nsrdb.file_handlers.outputs import Outputs

RTOL = 0.05
ATOL = .1


def test_output_handler(var_list=('surface_pressure', 'air_temperature',
                                  'wind_speed')):
    """Test the output handler."""
    with tempfile.TemporaryDirectory() as td:
        out_file = os.path.join(td, 'output_handler_test.h5')
        date = datetime.date(year=2017, month=1, day=1)
        grid = os.path.join(TESTDATADIR, 'reference_grids/',
                            'west_psm_extent.csv')

        # set test directory
        source_dir = os.path.join(TESTDATADIR, 'merra2_source_files/')
        var_meta = pd.read_csv(os.path.join(CONFIGDIR, 'nsrdb_vars.csv'))
        var_meta['source_directory'] = source_dir

        data_model = DataModel.run_multiple(var_list, date,
                                            grid, var_meta=var_meta,
                                            max_workers=1,
                                            return_obj=True)

        with Outputs(out_file, mode='w') as fout:
            fout.time_index = data_model.nsrdb_ti
            fout.meta = data_model.nsrdb_grid

            for k, v in data_model.processed_data.items():
                if k not in ['time_index', 'meta']:
                    var = VarFactory.get_base_handler(k, var_meta=var_meta,
                                                      date=date)
                    attrs = var.attrs

                    fout._add_dset(dset_name=k, data=v, dtype=var.final_dtype,
                                   chunks=None, attrs=attrs)

        with Outputs(out_file, mode='r', unscale=False) as fout:
            for dset in data_model.processed_data.keys():
                if dset not in ['time_index', 'meta']:
                    data = fout[dset]
                    msg = 'Output handler failed for {}'.format(dset)
                    assert np.allclose(data, data_model[dset],
                                       rtol=RTOL, atol=ATOL), msg


def execute_pytest(capture='all', flags='-rapP', purge=True):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
