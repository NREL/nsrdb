# pylint: skip-file
"""
PyTest file for solar position algorithms.

Created on Jun 4th 2019

@author: gbuster
"""
from copy import deepcopy
import pytest
import numpy as np
import os
from nsrdb.gap_fill.cloud_fill import CloudGapFill

CLOUD_TYPE = np.array([[0, 0, -15],
                       [1, 1, -15],
                       [7, 3, -15],
                       [7, -15, -15],
                       [3, 8, -15],
                       [3, 8, -15],
                       [7, 3, -15],
                       [7, 3, -15],
                       ])

CLD_OPD_DCOMP = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [71, 43, 0],
                          [73, 45, 0],
                          [17, 29, 0],
                          [14, 21, 0],
                          ], dtype=np.int32)

SZA = np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [180, 0, 0],
                ])

OUT_CTYPE = np.array([[0, 0, 0],
                      [1, 1, 0],
                      [7, 3, 0],
                      [7, 3, 0],
                      [3, 8, 0],
                      [3, 8, 0],
                      [7, 3, 0],
                      [7, 3, 0]], dtype=np.int8)

OUT_PROP = np.array([[0, 0, 0],
                     [0, 0, 0],
                     [17, 29, 0],
                     [17, 29, 0],
                     [71, 43, 0],
                     [73, 45, 0],
                     [17, 29, 0],
                     [0, 21, 0]], dtype=np.int32)

OUT_FILL_FLAG = np.array([[0, 0, 2],
                          [0, 0, 2],
                          [3, 3, 2],
                          [3, 1, 2],
                          [0, 0, 2],
                          [0, 0, 2],
                          [0, 0, 2],
                          [0, 0, 2]], dtype=np.int8)


def test_type():
    """Test the cloud property gap fill algorithm."""
    cloud_type, fill_flag = CloudGapFill.fill_cloud_type(deepcopy(CLOUD_TYPE))
    assert np.array_equal(cloud_type, OUT_CTYPE)
    return cloud_type, fill_flag


def test_opd():
    """Test the cloud property gap fill algorithm."""
    _, fill_flag = CloudGapFill.fill_cloud_type(deepcopy(CLOUD_TYPE))
    cloud_prop, fill_flag = CloudGapFill.fill_cloud_prop('cld_opd_dcomp',
                                                         CLD_OPD_DCOMP,
                                                         CLOUD_TYPE, SZA,
                                                         fill_flag=fill_flag)
    assert np.array_equal(cloud_prop, OUT_PROP)
    assert np.array_equal(fill_flag, OUT_FILL_FLAG)
    return cloud_prop, fill_flag


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
