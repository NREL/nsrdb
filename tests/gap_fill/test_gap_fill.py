"""
PyTest file for solar position algorithms.

Created on Jun 4th 2019

@author: gbuster
"""
from copy import deepcopy

import numpy as np

from nsrdb.gap_fill.cloud_fill import CloudGapFill
from nsrdb.utilities.pytest import execute_pytest

CLOUD_TYPE = np.array([[0, 0, -128, 0, 0, 7],
                       [1, 1, -15, 0, 0, 7],
                       [7, 3, -15, 0, 0, 0],
                       [7, -15, -15, 0, 1, 0],
                       [3, 8, -15, 5, 1, 7],
                       [3, 8, -15, 5, 1, 7],
                       [7, 3, -15, -15, 4, -15],
                       [7, 3, -15, 5, 4, 7],
                       ])

CLD_OPD_DCOMP = np.array([[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [71, 43, 0, 0, 0, 0],
                          [73, 45, 0, 0, 0, 41],
                          [17, 29, 0, 0, 0, 0],
                          [14, 21, 0, 0, 0, 0],
                          ], dtype=np.int32)

SZA = np.array([[0, 0, 0, 0, 0, 180],
                [0, 0, 0, 0, 0, 180],
                [0, 0, 0, 0, 180, 180],
                [0, 0, 0, 0, 180, 180],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 180, 0],
                [180, 0, 0, 180, 180, 0],
                ])

OUT_CTYPE = np.array([[0, 0, 0, 0, 0, 7],
                      [1, 1, 0, 0, 0, 7],
                      [7, 3, 0, 0, 0, 0],
                      [7, 3, 0, 0, 1, 0],
                      [3, 8, 0, 5, 1, 7],
                      [3, 8, 0, 5, 1, 7],
                      [7, 3, 0, 5, 4, 7],
                      [7, 3, 0, 5, 4, 7]], dtype=np.int8)

OUT_PROP = np.array([[0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [17, 29, 0, 0, 0, 0],
                     [17, 29, 0, 0, 0, 0],
                     [71, 43, 0, 10, 0, 41],
                     [73, 45, 0, 10, 0, 41],
                     [17, 29, 0, 10, 0, 41],
                     [0, 21, 0, 0, 0, 41]], dtype=np.int32)

OUT_FILL_FLAG = np.array([[0, 0, 2, 0, 0, 0],
                          [0, 0, 2, 0, 0, 0],
                          [3, 3, 2, 0, 0, 0],
                          [3, 1, 2, 0, 0, 0],
                          [0, 0, 2, 4, 0, 3],
                          [0, 0, 2, 4, 0, 0],
                          [0, 0, 2, 1, 0, 1],
                          [0, 0, 2, 0, 0, 3]], dtype=np.int8)


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


if __name__ == '__main__':
    execute_pytest(__file__)
