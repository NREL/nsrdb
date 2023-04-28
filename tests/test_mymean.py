# -*- coding: utf-8 -*-
"""
PyTest file for multi year mean.

Created on Feb 13th 2019

@author: gbuster
"""
import os
import pytest
import numpy as np
import tempfile

from nsrdb import TESTDATADIR
from nsrdb.mymean.mymean import MyMean
from nsrdb.file_handlers.outputs import Outputs


NSRDB_DIR = os.path.join(TESTDATADIR, 'validation_nsrdb/')


def test_mymean():
    """Test multiyear mean"""
    with tempfile.TemporaryDirectory() as td:
        flist = [os.path.join(NSRDB_DIR, 'nsrdb_surfrad_{}.h5'.format(y))
                 for y in range(1998, 2001)]
        fout = os.path.join(td, 'mymean.h5')
        dset = 'ghi'
        MyMean.run(flist, fout, dset, process_chunk=2, parallel=False)

        with Outputs(fout, mode='r') as out:
            print(out.get_attrs(dset=dset))
            data = out[dset]

        truth = None
        for f in flist:
            with Outputs(f, mode='r') as out:
                temp = out[dset]
            if truth is None:
                truth = temp.mean(axis=0)
            else:
                truth += temp.mean(axis=0)

        truth = truth / len(flist) / 1000 * 24

        assert np.allclose(data, truth, atol=1)


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
