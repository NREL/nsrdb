# -*- coding: utf-8 -*-
"""Update NSRDB h5 file version attributes

@author: gbuster
"""
import os

import h5py

root_dir = '/projects/pxs/nsrdb/v3.0.1/'

versions = {"nsrdb_1998.h5": "3.0.6",
            "nsrdb_1999.h5": "3.0.6",
            "nsrdb_2000.h5": "3.0.6",
            "nsrdb_2001.h5": "3.0.6",
            "nsrdb_2002.h5": "3.0.6",
            "nsrdb_2003.h5": "3.0.6",
            "nsrdb_2004.h5": "3.0.6",
            "nsrdb_2005.h5": "3.0.6",
            "nsrdb_2006.h5": "3.0.6",
            "nsrdb_2007.h5": "3.0.6",
            "nsrdb_2008.h5": "3.0.6",
            "nsrdb_2009.h5": "3.0.6",
            "nsrdb_2010.h5": "3.0.6",
            "nsrdb_2011.h5": "3.0.6",
            "nsrdb_2012.h5": "3.0.6",
            "nsrdb_2013.h5": "3.0.6",
            "nsrdb_2014.h5": "3.0.6",
            "nsrdb_2015.h5": "3.0.6",
            "nsrdb_2016.h5": "3.0.6",
            "nsrdb_2017.h5": "3.0.6",
            "nsrdb_2018.h5": "3.1.0",
            "nsrdb_tdy-2016.h5": "3.0.1",
            "nsrdb_tdy-2017.h5": "3.0.1",
            "nsrdb_tdy-2018.h5": "3.1.1",
            "nsrdb_tgy-2016.h5": "3.0.1",
            "nsrdb_tgy-2017.h5": "3.0.1",
            "nsrdb_tgy-2018.h5": "3.1.1",
            "nsrdb_tmy-2016.h5": "3.0.1",
            "nsrdb_tmy-2017.h5": "3.0.1",
            "nsrdb_tmy-2018.h5": "3.1.1",
            "nsrdb_conus_ancillary_2018.h5": "3.1.0",
            "nsrdb_conus_clouds_2018.h5": "3.1.0",
            "nsrdb_conus_irradiance_2018.h5": "3.1.0",
            "nsrdb_conus_sam_2018.h5": "3.1.0",
            "nsrdb_full_disc_ancillary_2018.h5": "3.1.0",
            "nsrdb_full_disc_clouds_2018.h5": "3.1.0",
            "nsrdb_full_disc_irradiance_2018.h5": "3.1.0",
            "nsrdb_full_disc_sam_2018.h5": "3.1.0",
            }

for fn in os.listdir(root_dir):
    if fn in versions and fn.endswith('.h5'):
        fp = os.path.join(root_dir, fn)
        version = versions[fn]

        print('Updating "{}" to version: {}'.format(fn, version))

        with h5py.File(fp, 'a') as f:
            f.attrs['version'] = version
