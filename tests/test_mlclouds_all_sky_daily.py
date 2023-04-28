# pylint: skip-file
"""
PyTest file for all sky daily processing after mlclouds daily gap fill

Created on 12/3/2020

@author: gbuster
"""
import pytest
import numpy as np
import os
import tempfile
import shutil
from nsrdb import TESTDATADIR
from nsrdb.nsrdb import NSRDB
from rex import MultiFileNSRDB

pytest.importorskip("mlclouds")
pytest.importorskip("phygnn")
from nsrdb.gap_fill.mlclouds_fill import MLCloudsFill


PROJECT_DIR = os.path.join(TESTDATADIR, 'mlclouds_pipeline/')
ARCHIVE_DIR = os.path.join(PROJECT_DIR, 'daily_files_archive/')
GRID = os.path.join(PROJECT_DIR, 'surfrad_meta.csv')


@pytest.mark.parametrize('date', ('20190102', '20190103', '20190104'))
def test_all_sky_daily(date):
    """Test the mlclouds fill on daily files then all sky from those files."""
    with tempfile.TemporaryDirectory() as td:
        project_tdir = os.path.join(td, 'mlclouds_pipeline/')
        daily_dir = os.path.join(td, 'mlclouds_pipeline/daily/')
        shutil.copytree(ARCHIVE_DIR, daily_dir)

        h5_source = os.path.join(daily_dir, '{}*.h5'.format(date))
        MLCloudsFill.run(h5_source)

        NSRDB.run_daily_all_sky(project_tdir, date[0:4], GRID, date,
                                freq='5min', max_workers=1)

        dsets = ('dni', 'ghi', 'dhi', 'fill_flag', 'clearsky_dhi',
                 'clearsky_dni', 'clearsky_ghi')
        with MultiFileNSRDB(h5_source) as res:
            assert all([d in res.dsets for d in dsets])
            dni = res['dni']
            ghi = res['ghi']
            dhi = res['dhi']
            sza = res['solar_zenith_angle']
            fill_flag = res['fill_flag']
            cloud_fill_flag = res['cloud_fill_flag']

        # DNI can be all 0 if its a very cloudy day
        # tested below with the diffuse_mask

        assert ~(ghi == 0).all(axis=0).any()
        assert ~(dhi == 0).all(axis=0).any()
        assert ~(sza == 0).all(axis=0).any()
        assert (fill_flag > 0).sum() > 50
        assert (cloud_fill_flag > 0).sum() > 50
        day_mask = (sza < 89)
        assert day_mask.any()
        assert (ghi[day_mask] > 0).all()
        diffuse_mask = day_mask & (dni == 0)
        not_diffuse_mask = day_mask & (dhi == 0)
        assert (dhi[diffuse_mask] > 0).all()
        assert (dni[not_diffuse_mask] > 0).all()

        assert np.isin(cloud_fill_flag, (0, 1, 2, 7)).all()
        assert np.isin(fill_flag, (0, 1, 2, 5, 7)).all()


def execute_pytest(capture='all', flags='-rapP'):
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
