"""Test for temperature dependent albedo calculation"""

import os
from datetime import datetime as dt
import tempfile
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


from nsrdb import TESTDATADIR
from nsrdb.albedo import temperature_model as tm
from nsrdb.albedo import albedo
from nsrdb.albedo import modis, ims
from nsrdb.albedo.albedo import (ModisClipper,
                                 IMS_EDGE_BUFFFER,
                                 ALBEDO_NODATA)

ALBEDOTESTDATADIR = os.path.join(TESTDATADIR, 'albedo')
source_dir = os.path.join(TESTDATADIR, 'merra2_source_files')

td = tempfile.gettempdir()
snow_no_snow_file = os.path.join(td, 'snow_no_snow.py')


def calc_albedo(cad):
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

    cad._mask = snow_no_snow

    # Update MODIS albedo for cells w/ snow
    mclip_albedo = mc.modis_clip

    if cad._merra_path is not None:
        cad._merra_data = tm.DataHandler.get_data(
            cad.date, cad._merra_path, snow_no_snow,
            mc.mlat_clip, mc.mlon_clip, avg=False,
            fp_out=f'{td}/merra_data.csv')

        mclip_albedo = tm.TemperatureModel.update_snow_albedo(
            mclip_albedo, snow_no_snow, cad._merra_data)
    else:
        mclip_albedo[snow_no_snow == 1] = 867
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


def test_albedo_model(with_temp_model=False, plot=True):
    """ Test temperature based albedo model """

    d = dt(2013, 1, 1)
    modis_shape = (122, 120)
    ims_shape = (32, 25)

    if with_temp_model:
        cad = albedo.CompositeAlbedoDay(d, ALBEDOTESTDATADIR,
                                        ALBEDOTESTDATADIR, td,
                                        source_dir)
    else:
        cad = albedo.CompositeAlbedoDay(d, ALBEDOTESTDATADIR,
                                        ALBEDOTESTDATADIR, td)

    cad._modis = modis.ModisDay(cad.date, cad._modis_path,
                                shape=modis_shape)

    cad._ims = ims.ImsDay(cad.date, cad._ims_path, shape=ims_shape)

    cad.albedo = calc_albedo(cad)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4), ncols=1)
        im = ax.imshow(cad.albedo, interpolation='none')
        plt.title('Albedo')
        fig.colorbar(im, ax=ax)

        plt.show()
        cad.write_tiff()

        if with_temp_model:
            fig, ax = plt.subplots(figsize=(8, 4), ncols=1)
            T = np.zeros(cad._mask.shape).flatten()
            T[cad._mask.flatten() == 1] = cad._merra_data
            T = T.reshape(cad._mask.shape)
            im = ax.imshow(T, interpolation='none')
            plt.title('Merra Temperature')
            fig.colorbar(im, ax=ax)
            plt.show()
