# pylint: skip-file
"""
PyTest file for all sky utilities.

Created on June 7th 2019.

@author: gbuster
"""

import os
import pytest
import numpy as np
from nsrdb.all_sky.utilities import rayleigh


RTOL = 0.001
ATOL = 0.001


def test_rayleigh():
    """Test the rayleigh violation check."""

    dhi = np.array([[0.72444256, 0.23359043],
                    [0.81759012, 0.88451122],
                    [0.6008738, 0.03514198],
                    [0.95301128, 0.36230728],
                    [0.65396809, 0.25557211]])
    cs_dhi = np.array([[0.82444256, 0.23359043],
                       [0.81759012, 0.88451122],
                       [0.7008738, 0.03514198],
                       [0.95301128, 0.36230728],
                       [0.65396809, 0.35557211]])

    fill_out = np.array([[1, 0],
                         [0, 0],
                         [1, 0],
                         [0, 0],
                         [0, 1]], dtype=np.int16)

    fill_flag = np.zeros_like(dhi).astype(np.int16)

    fill_flag = rayleigh(dhi, cs_dhi, fill_flag, rayleigh_flag=1)

    assert np.array_equal(fill_flag, fill_out)


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
