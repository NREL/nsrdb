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

from nsrdb import TESTDATADIR, CONFIGDIR
from nsrdb.data_model import DataModel
from nsrdb.file_handlers.outputs import Outputs


RTOL = 0.05
ATOL = .1
PURGE_OUT = True


def test_output_handler(var_list=('surface_pressure', 'air_temperature',
                                  'wind_speed')):
    """Test the output handler."""

    out_dir = os.path.join(TESTDATADIR, 'processed_ancillary/')
    out_file = os.path.join(out_dir, 'output_handler_test.h5')
    date = datetime.date(year=2017, month=1, day=1)
    grid = os.path.join(TESTDATADIR, 'reference_grids/', 'west_psm_extent.csv')

    # set test directory
    source_dir = os.path.join(TESTDATADIR, 'merra2_source_files/')
    var_meta = pd.read_csv(os.path.join(CONFIGDIR, 'nsrdb_vars.csv'))
    var_meta['source_directory'] = source_dir

    data_model = DataModel.run_multiple(var_list, var_meta, date,
                                        grid, parallel=False,
                                        return_obj=True)

    with Outputs(out_file, mode='w') as fout:
        fout.time_index = data_model.nsrdb_ti
        fout.meta = data_model.nsrdb_grid

        for k, v in data_model.processed_data.items():
            if k not in ['time_index', 'meta']:
                mask = (var_meta['var'] == k)
                units = str(var_meta.loc[mask, 'units'].values[0])
                dtype = str(var_meta.loc[mask, 'final_dtype'].values[0])
                scale_factor = float(
                    var_meta.loc[mask, 'scale_factor'].values[0])
                attrs = {'units': units, 'scale_factor': scale_factor}

                fout._add_dset(dset_name=k, data=v, dtype=dtype,
                               chunks=None, attrs=attrs)

    with Outputs(out_file, mode='r') as fout:
        for dset in data_model.processed_data.keys():
            if dset not in ['time_index', 'meta']:
                data = fout[dset]
                msg = 'Output handler failed for {}'.format(dset)
                assert np.allclose(data, data_model[dset],
                                   rtol=RTOL, atol=ATOL), msg

    if PURGE_OUT:
        os.remove(out_file)


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
