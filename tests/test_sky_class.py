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
    fp_surf = os.path.join(TESTDATADIR,
                           'validation_nsrdb_2019/tbl_2019.h5')
    fp_nsrdb = os.path.join(TESTDATADIR,
                            'validation_nsrdb_2019/srf19a_*_2019.h5')
    gid = 1

    df_val = SkyClass.run(fp_surf, fp_nsrdb, gid)
    assert 'sky_class' in df_val
    assert len(df_val) == 8760 * 12
    assert (df_val['sky_class'] == 'missing').sum() / len(df_val) > 0.5
