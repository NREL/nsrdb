"""Test for temperature dependent albedo calculation"""

import os
from datetime import datetime as dt
import tempfile
import numpy as np
from scipy.spatial import cKDTree

from nsrdb import TESTDATADIR
from nsrdb.albedo.temperature_model import TemperatureModel
from nsrdb.albedo import albedo
from nsrdb.albedo import modis, ims
from nsrdb.albedo.albedo import (ModisClipper,
                                 IMS_EDGE_BUFFFER,
                                 ALBEDO_NODATA)

ALBEDOTESTDATADIR = os.path.join(TESTDATADIR, 'albedo')
source_dir = '/projects/pxs/ancillary/merra/'
model = TemperatureModel(source_dir)

with tempfile.TemporaryDirectory as td:
    snow_no_snow_file = f'{td}/snow_no_snow.py'


def calc_albedo(cad, data):
    """Calculate albedo using provided data.
    Load or save snow_no_snow mask from/to td"""

    # Clip MODIS data to IMS boundary
    mc = ModisClipper(cad._modis, cad._ims)

    if not os.path.exists(snow_no_snow_file):

        # Find snow/no snow region boundaries of IMS
        ims_bin_mskd, ims_pts = cad._get_ims_boundary(buffer=IMS_EDGE_BUFFFER)

        # Create cKDTree to map MODIS points onto IMS regions
        ims_tree = cKDTree(ims_pts)

        # Map MODIS pixels to IMS data
        modis_pts = cad._get_modis_pts(mc.mlon_clip, mc.mlat_clip)
        if cad._max_workers != 1:
            ind = cad._run_futures(ims_tree, modis_pts)
        else:
            ind = cad._run_single_tree(ims_tree, modis_pts)

        # Project nearest neighbors from IMS to MODIS. Array is on same grid as
        # clipped MODIS, but has snow/no snow values from binary IMS.
        snow_no_snow = ims_bin_mskd[ind].reshape(len(mc.mlat_clip),
                                                 len(mc.mlon_clip))

        with open(snow_no_snow_file, 'wb') as f:
            np.save(f, snow_no_snow)

    else:

        with open(snow_no_snow_file, 'rb') as f:
            snow_no_snow = np.load(f)

    # Update MODIS albedo for cells w/ snow
    mclip_albedo = mc.modis_clip

    mclip_albedo = model.update_snow_albedo(
        mclip_albedo, snow_no_snow, data)

    # Merge clipped composite albedo with full MODIS data
    albedo = cad._modis.data
    albedo[mc.modis_idx] = mclip_albedo

    # Reset NODATA values
    albedo[albedo == modis.MODIS_NODATA] = ALBEDO_NODATA

    # MODIS data has a scaling factor of 1000, reduce to 100
    albedo /= 10
    albedo = np.round(albedo)
    albedo = albedo.astype(np.uint8)

    return albedo


def test_albedo_model():
    """ Test temperature based albedo model """

    d = dt(2013, 1, 1)
    modis_shape = (122, 120)
    ims_shape = (32, 25)

    with tempfile.TemporaryDirectory() as td:
        cad = albedo.CompositeAlbedoDay(d, ALBEDOTESTDATADIR,
                                        ALBEDOTESTDATADIR, td)

    cad._modis = modis.ModisDay(cad.date, cad._modis_path,
                                shape=modis_shape)

    grid = model.get_grid(cad)

    cad._ims = ims.ImsDay(cad.date, cad._ims_path, shape=ims_shape)

    data = model.get_data(d, grid)

    cad = calc_albedo(cad, data)