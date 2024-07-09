"""
PyTest file for multi year mean.

Created on Feb 13th 2019

@author: gbuster
"""
import os
import tempfile

import numpy as np

from nsrdb import TESTDATADIR
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.mymean.mymean import MyMean
from nsrdb.utilities.pytest import execute_pytest

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


if __name__ == '__main__':
    execute_pytest(__file__)
