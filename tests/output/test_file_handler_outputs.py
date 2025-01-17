# pylint: skip-file
"""
PyTest file for PV generation in Rhode Island.

Created on Thu Nov 29 09:54:51 2018

@author: gbuster
"""

import datetime
import os
import tempfile

import numpy as np
import pandas as pd

from nsrdb import DEFAULT_VAR_META, TESTDATADIR
from nsrdb.data_model import DataModel, VarFactory
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.utilities.file_utils import clean_meta
from nsrdb.utilities.pytest import execute_pytest

RTOL = 0.05
ATOL = .1


def test_clean_meta():
    """Test clean_meta routine"""
    meta_file = os.path.join(TESTDATADIR, 'meta/',
                             'surfrad_meta.csv')
    meta = pd.read_csv(meta_file, index_col=0)
    meta.loc[meta.sample(frac=0.4).index, 'state'] = np.nan
    meta = clean_meta(meta)

    assert meta['elevation'].dtype == np.int16
    assert meta['timezone'].dtype == np.int8
    assert not meta.isnull().values.any()


def test_coordinate_export():
    """Test coordinate export handling"""
    meta_file = os.path.join(TESTDATADIR, 'meta/',
                             'surfrad_meta.csv')
    meta = pd.read_csv(meta_file, index_col=0)
    with tempfile.TemporaryDirectory() as td:
        out_file = os.path.join(td, 'coordinate_export_test.h5')
        date = datetime.date(year=2017, month=1, day=1)
        with Outputs(out_file, mode='w') as fout:
            time_index = pd.date_range('1-1-{y}'.format(y=date.year),
                                       '1-1-{y}'.format(y=date.year + 1),
                                       freq='6m')[:-1]
            fout.init_h5(out_file, [], {}, {}, {},
                         time_index, meta, mode='a',
                         add_coords=True)

            coords_check = meta[['latitude', 'longitude']].to_numpy()
            coords_check = coords_check.astype(np.float32)

            assert np.array_equal(coords_check, fout['coordinates'])


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
        var_meta = pd.read_csv(DEFAULT_VAR_META)
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
            for dset in data_model.processed_data:
                if dset not in ['time_index', 'meta']:
                    data = fout[dset]
                    msg = 'Output handler failed for {}'.format(dset)
                    assert np.allclose(data, data_model[dset],
                                       rtol=RTOL, atol=ATOL), msg


if __name__ == '__main__':
    execute_pytest(__file__)
