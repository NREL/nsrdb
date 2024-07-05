"""CLI test for blend module."""

import json
import os
import tempfile
import traceback

import numpy as np
import pandas as pd

from nsrdb import cli
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.utilities.file_utils import pd_date_range
from nsrdb.utilities.pytest import execute_pytest


def test_blend_cli(runner):
    """Test the blend cli"""
    meta_out = pd.DataFrame(
        {'longitude': np.linspace(-100, 100, 500), 'latitude': np.zeros(500)}
    )
    lon_seam = 0.25
    meta_east = meta_out.copy()
    meta_west = meta_out.copy()
    meta_east['gid_full_map'] = np.arange(len(meta_out))
    meta_west['gid_full_map'] = np.arange(len(meta_out))
    time_index = pd_date_range(
        '20190101', '20200101', freq='1h', closed='left'
    )

    with tempfile.TemporaryDirectory() as td:
        meta_path = os.path.join(td, 'meta_out.csv')
        meta_out.to_csv(meta_path)
        east_dir = os.path.join(td, 'east/')
        west_dir = os.path.join(td, 'west/')
        out_dir = os.path.join(td, 'out/')
        os.mkdir(east_dir)
        os.mkdir(west_dir)

        east_fp = os.path.join(east_dir, 'nsrdb_conus_east_irradiance.h5')
        west_fp = os.path.join(west_dir, 'nsrdb_conus_west_irradiance.h5')
        out_fp = os.path.join(out_dir, 'nsrdb_conus_irradiance.h5')

        dsets = ['dni', 'dhi', 'ghi']
        attrs = {d: {'scale_factor': 1, 'units': 'unitless'} for d in dsets}
        chunks = dict.fromkeys(dsets)
        dtypes = dict.fromkeys(dsets, 'uint16')

        Outputs.init_h5(
            east_fp, dsets, attrs, chunks, dtypes, time_index, meta_east
        )
        Outputs.init_h5(
            west_fp, dsets, attrs, chunks, dtypes, time_index, meta_west
        )

        with Outputs(east_fp, mode='a') as f:
            for dset in dsets:
                f[dset] = np.zeros((8760, len(meta_out)))

        with Outputs(west_fp, mode='a') as f:
            for dset in dsets:
                f[dset] = np.ones((8760, len(meta_out)))

        config = {
            'blend': {
                'meta': meta_path,
                'out_dir': out_dir,
                'east_dir': east_dir,
                'west_dir': west_dir,
                'file_tag': 'nsrdb_conus_',
                'map_col': 'gid_full_map',
                'lon_seam': lon_seam,
            },
        }

        config_file = os.path.join(td, 'config.json')
        with open(config_file, 'w') as f:
            f.write(json.dumps(config))

        result = runner.invoke(cli.blend, ['-c', config_file, '-v'])
        assert result.exit_code == 0, traceback.print_exception(
            *result.exc_info
        )

        with Outputs(out_fp) as out:
            west_mask = out.meta.longitude < lon_seam
            east_mask = out.meta.longitude >= lon_seam
            for dset in dsets:
                data = out[dset]
                assert (data[:, west_mask] == 1).all()
                assert (data[:, east_mask] == 0).all()


if __name__ == '__main__':
    execute_pytest(__file__)
