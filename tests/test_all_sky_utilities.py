# pylint: skip-file
"""
PyTest file for all sky utilities.

Created on June 7th 2019.

@author: gbuster
"""

import os
import pytest
import numpy as np
from nsrdb.all_sky.utilities import rayleigh_check


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
    dni = np.array([[0.82444256, 0.23359043],
                    [0.81759012, 0.88451122],
                    [0.7008738, 0.03514198],
                    [0.95301128, 0.36230728],
                    [0.65396809, 0.35557211]])
    cs_dni = np.array([[0.46797247, 0.0712267],
                       [0.74319406, 0.80287394],
                       [0.77036961, 0.34277198],
                       [0.56129481, 0.80525808],
                       [0.50996644, 0.95803792]])
    ghi = np.array([[0.82444256, 0.23359043],
                    [0.81759012, 0.88451122],
                    [0.7008738, 0.03514198],
                    [0.95301128, 0.36230728],
                    [0.65396809, 0.35557211]])
    cs_ghi = np.array([[0.46797247, 0.0712267],
                       [0.74319406, 0.80287394],
                       [0.77036961, 0.34277198],
                       [0.56129481, 0.80525808],
                       [0.50996644, 0.95803792]])

    dhi_out = np.array([[0.82444256, 0.23359043],
                        [0.81759012, 0.88451122],
                        [0.7008738, 0.03514198],
                        [0.95301128, 0.36230728],
                        [0.65396809, 0.35557211]])
    dni_out = np.array([[0.46797247, 0.23359043],
                        [0.81759012, 0.88451122],
                        [0.77036961, 0.03514198],
                        [0.95301128, 0.36230728],
                        [0.65396809, 0.95803792]])
    ghi_out = np.array([[0.46797247, 0.23359043],
                        [0.81759012, 0.88451122],
                        [0.77036961, 0.03514198],
                        [0.95301128, 0.36230728],
                        [0.65396809, 0.95803792]])
    fill_out = np.array([[5, 0],
                         [0, 0],
                         [5, 0],
                         [0, 0],
                         [0, 5]], dtype=np.int16)

    fill_flag = np.zeros_like(dhi).astype(np.int16)

    dhi, dni, ghi, fill_flag = rayleigh_check(dhi, dni, ghi, cs_dhi, cs_dni,
                                              cs_ghi, fill_flag)

    assert np.array_equal(dhi, dhi_out)
    assert np.array_equal(dni, dni_out)
    assert np.array_equal(ghi, ghi_out)
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
