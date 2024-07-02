# -*- coding: utf-8 -*-
"""
PyTest file for rest2.

Created on Feb 13th 2019

@author: gbuster
"""

import os

from nsrdb import TESTDATADIR
from nsrdb.utilities.sky_class import SkyClass


def test_sky_class():
    """Simple sky classification test"""
    fp_surf = os.path.join(TESTDATADIR, 'validation_nsrdb_2019/tbl_2019.h5')
    fp_nsrdb = os.path.join(
        TESTDATADIR, 'validation_nsrdb_2019/srf19a_*_2019.h5'
    )
    gid = 1

    with SkyClass(fp_surf, fp_nsrdb, gid) as sc:
        df_val = sc.get_comparison_df()
        df_val = sc.calculate_sky_class(df_val)
        df_val = sc.add_validation_data(df_val)

    assert 'sky_class' in df_val
    assert len(df_val) == 8760 * 12
    assert (df_val['sky_class'] == 'missing').sum() / len(df_val) > 0.5
