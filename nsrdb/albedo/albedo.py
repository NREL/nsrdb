# -*- coding: utf-8 -*-
"""
Class to compile MODIS dry land albedo and IMS snow data into composite albedo.

Created on Jan 23 2020

@author: mbannist
"""
import os
import numpy as np
import h5py
from scipy import ndimage
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import libtiff
import logging
from datetime import datetime as dt

import nsrdb.albedo.modis as modis
import nsrdb.albedo.ims as ims


# TIFF world file. Values were extracted from MODIS data with QGIS and may
# be slightly off.
WORLD = '''0.00833333
0.00000000
0.00000000
-0.00833333
-179.99583333
89.99583333
'''

# MODIS is clipped to IMS extent before calculating nearest neighbors.
# CLIP_MARGIN is a small buffer around the IMS extent to prevent the MODIS
# data from being clipped slightly too small due to mismatched projections.
CLIP_MARGIN = 0.1  # degrees lat/long


logger = logging.getLogger(__name__)


class AlbedoError (Exception):
    """ Exceptions for albedo related errors """
    pass


class CompositeAlbedoDay:
    """
    Combine IMS and MODIS data to create composite Albedo.

    This class is intended to be used with the run() class method, which
    acquires IMS and MODIS data, merges the two, and a returns an instance
    of the class. Exporting albedo data to HDF5 or TIFF must be explicitly
    requested by the user using appropriate methods.

    The composite albedo data is created by mapping the MODIS data to the IMS
    data using a KD tree to perform a nearest neighbor analysis using
    concurrent futures. Passing a KD tree based on the full IMS data to the
    futures proved problematic due to memory issues, so the IMS data is first
    reduced to the snow/no-snow boundary areas first, which reduces the size
    of the tree significantly.

    Combining the two data sources takes roughly 15 minutes for older, 4 km
    IMS data, and 50 minutes for 1 km IMS data running on a 36 core eagle node.

    Methods
    -------
    run - Load all data and create composite albedo.
    write_albedo - Write albedo data to HDF5 file.
    write_tiff - Write albedo to georeferenced TIFF.
    write_tiff_from_h5 - Load existing composite albedo data from HDF5 and
                         write to TIFF.
    """
    # Value for snow/sea ice in IMS. In thousandths, e.g. 867 == 0.867
    SNOW_ALBEDO = 867

    @classmethod
    def run(cls, date, modis_path, ims_path, albedo_path, ims_shape=None,
            modis_shape=None):
        """
        Merge MODIS and IMS data for one day.

        Parameters
        ----------
        date : datetime instance
            Date to calculate composite albedo for
        modis_path : str
            Path for MODIS data files
        ims_path : str
            Path for IMS data files
        albedo_path : str
            Path for composite albedo data files (output)
        ims_shape : (int, int)
            Shape of IMS data (rows, cols). Defaults to normal shape.
        modis : (int, int)
            Shape of MODIS data (rows, cols). Defaults to normal shape.

        Returns
        -------
            Class instance
        """
        cad = cls(date, modis_path, ims_path, albedo_path)

        logger.info(f'Loading MODIS data for {cad.date}')
        cad.modis = modis.ModisDay(cad.date, cad.modis_path, shape=modis_shape)

        logger.info(f'Loading IMS data {cad.date}')
        cad.ims = ims.ImsDay(cad.date, cad.ims_path, shape=ims_shape)

        cad.albedo = cad._calc_albedo()
        return cad

    def __init__(self, date, modis_path, ims_path, albedo_path):
        """ See parameter definitions for self.run() """
        self.date = date
        self.modis_path = modis_path
        self.ims_path = ims_path
        self.albedo_path = albedo_path

        self.modis = None  # ModisDay object
        self.ims = None  # ImsDay object
        self.albedo = None  # numpy array of albedo data, same formate as MODIS

    def write_albedo(self):
        """
        Write albedo data to HDF5 file

        Parameters
        ----------
        path : string
            Location to save albedo data to
        """

        day = str(self.date.timetuple().tm_yday).zfill(3)
        year = self.date.year
        # TODO update the file pattern
        outfilename = os.path.join(self.albedo_path,
                                   f'nsrdb_albedo_{year}_{day}.h5')

        albedo_attrs = {'units': 'unitless',
                        'scale_factor': 1000}

        logger.info(f'Writing albedo data to {outfilename}')
        with h5py.File(outfilename, 'w') as f:
            f.create_dataset('surface_albedo', shape=self.albedo.shape,
                             dtype=self.albedo.dtype, data=self.albedo)
            for k, v in albedo_attrs.items():
                f['surface_albedo'].attrs[k] = v

            f.create_dataset('latitude', shape=self.modis.lat.shape,
                             dtype=self.modis.lat.dtype, data=self.modis.lat)
            f.create_dataset('longitude', shape=self.modis.lon.shape,
                             dtype=self.modis.lon.dtype, data=self.modis.lon)

    def write_tiff(self):
        """
        Write albedo data to TIFF and world file. Georeferencing appears to be
        off by 5 meters.
        """
        day = str(self.date.timetuple().tm_yday).zfill(3)
        year = self.date.year
        outfilename = os.path.join(self.albedo_path,
                                   f'nsrdb_albedo_{year}_{day}.tif')

        self._create_tiff(self.albedo, outfilename)

    @classmethod
    def write_tiff_from_h5(cls, infilename, outfilename=None):
        """
        Write albedo data to TIFF and world file. Georeferencing appears to be
        off by 5 meters.

        Parameters
        ----------
        infilename : string
            Path and file name of albedo h5 file
        outfilename : string
            Path and file name for TIFF and world file. Defaults to infilename
            with .tif extension
        """
        if outfilename is None:
            outfilename = os.path.splitext(infilename)[0] + '.tif'

        with h5py.File(infilename, 'r') as f:
            data = np.array(f['surface_albedo'])

        cls._create_tiff(data, outfilename)

    @staticmethod
    def _create_tiff(data, filename):
        """
        Write albedo data to TIFF and world file. Georeferencing appears to be
        off by 5 meters.

        Parameters
        ----------
        data : np.array() (int16)
            Albedo data
        filename : string
            File name and full path for TIFF
        """
        tif = libtiff.TIFF.open(filename, mode='w')
        tif.write_image(data)
        tif.close()

        with open(os.path.splitext(filename)[0] + '.tfw', 'wt') as f:
            f.write(WORLD)

    def _calc_albedo(self):
        """
        Calculate composite albedo by merging MODIS and IMS

        Returns
        -------
        albedo : 2D numpy array
            MODIS data overlayed with IMS snow. Array has same shape/projection
            as MODIS
        """
        if self.modis is None or self.ims is None:
            raise AlbedoError('MODIS/IMS data must be loaded before running' +
                              ' calc_albedo()')
        logger.info(f'Calculating composite albedo for {self.date}')

        # Find snow/no snow region boundaries of IMS
        logger.info('Determining IMS snow/no snow region boundaries')
        ims_bin_mskd, ims_pts = self._get_ims_boundary()

        # Create cKDTree to map MODIS points onto IMS regions
        logger.info('Creating KD Tree')
        ims_tree = cKDTree(ims_pts)

        # Clip MODIS data to IMS boundary
        mc = ModisClipper(self.modis, self.ims)

        # Map MODIS pixels to IMS data
        logger.info('Mapping MODIS to IMS data. This might take a while.')
        modis_pts = self._get_modis_pts(mc.mlon_clip, mc.mlat_clip)
        ind = self._run_futures(ims_tree, modis_pts)

        # Project nearest neighbors from IMS to MODIS. Array is on same grid as
        # clipped MODIS, but has snow/no snow values from binary IMS.
        snow_no_snow = ims_bin_mskd[ind].reshape(len(mc.mlat_clip),
                                                 len(mc.mlon_clip))
        logger.info(f'Shape of snow/no snow grid is {snow_no_snow.shape}.')

        # Copy and update MODIS albedo for cells w/ snow
        mclip_albedo = mc.modis_clip.copy()
        mclip_albedo[snow_no_snow == 1] = self.SNOW_ALBEDO

        # Merge clipped composite albedo with full MODIS data
        albedo = self.modis.data.copy()
        albedo[mc.modis_idx] = mclip_albedo

        return albedo

    @staticmethod
    def _run_single_tree(tree, pts):
        """
        Map MODIS pixels to IMS binary data on one core.

        Parameters
        ----------
        tree : cKDTree
            KD tree created using IMS region boundary pixels
        pts : 2D numpy array
            Lon/lat locations for chunk of MODIS data cells

        Returns
        -------
        ind : 1D numpy array (int)
            Indices mapping MODIS cells to ims_bin_mskd
        """
        _, ind = tree.query(pts)
        # TODO don't return '_'. Required for "if future.result()" below
        return '_', ind

    def _run_futures(self, ims_tree, modis_pts):
        """
        Split mapping MODIS to IMS across multiple cores.

        Parameters
        ----------
        ims_tree : cKDTree
            KD tree created using IMS region boundary pixels
        modis_pts : 2D numpy array
            Lon/lat locations for MODIS data cells

        Returns
        -------
        ind : 1D numpy array (int)
            Indices mapping MODIS cells to ims_bin_mskd
        """
        futures = {}
        chunks = np.array_split(modis_pts, cpu_count())
        now = dt.now()
        with ProcessPoolExecutor() as exe:
            for i, chunk in enumerate(chunks):
                future = exe.submit(self._run_single_tree, ims_tree, chunk)
                meta = {'id': i}
                ct = chunk.T
                meta['lon_min'] = ct[0].min()
                meta['lon_max'] = ct[0].max()
                meta['lat_min'] = ct[1].min()
                meta['lat_max'] = ct[1].max()
                meta['size'] = ct.size
                futures[future] = meta
                # import pdb; pdb.set_trace()

            logger.info(f'Started all futures in {dt.now() - now}')

            for i, future in enumerate(as_completed(futures)):
                logger.info(f'Future {futures[future]} completed in ' +
                            f'{dt.now() - now}.')
                logger.info(f'{i + 1} out of {len(futures)} futures ' +
                            f'completed')
                # import pdb; pdb.set_trace()
        logger.info('done processing')

        # Merge all returned indices
        ind = np.empty((len(modis_pts)), dtype=int)
        pos = 0
        for key in futures.keys():
            size = len(key.result()[1])
            ind[pos:pos+size] = key.result()[1]
            pos += size
        return ind

    def _get_ims_boundary(self):
        """
        Create IMS boundary layer which represents the pixels that form the
        boundary between snow and no snow.

        Returns
        ------
        ims_bin_mskd : 1D numpy array
            IMS data, represented as 0 (no snow) or 1 (snow/sea ice), for
            region boundary pixels.
        ims_pts : 2D numpy array
            Lon/lat points for ims_bin_mskd
        """
        # Create binary IMS layer. Same size as original IMS.
        # TODO - change ims_bin to bool
        ims_bin = self.ims.data.copy().astype(np.int8)
        ims_bin[ims_bin < 3] = 0  # Dry land, water
        ims_bin[ims_bin > 2] = 1  # Snow, sea ice

        # Create and buffer edge mask
        ims_mask = ims_bin - ndimage.morphology.binary_dilation(ims_bin)
        ims_mask = ndimage.morphology.binary_dilation(ims_mask)
        ims_mask = ims_mask.flatten()

        # Mask data and lon/lat to boundary edges
        ims_bin_mskd = ims_bin.flatten()[ims_mask]
        ilon = self.ims.lon[ims_mask]
        ilat = self.ims.lat[ims_mask]

        # Combine lat and lon to create pts for KD tree
        ims_pts = np.vstack((ilon, ilat)).T

        return ims_bin_mskd, ims_pts

    def _get_modis_pts(self, lon_pts, lat_pts):
        """
        Create 2D numpy array representing lon/lats of MODIS pixels

        Parameters
        ----------
        lon_pts : numpy array
            Longitude points for MODIS data
        lat_pts : numpy array
            Latitude points for MODIS data

        Returns
        -------
        modis_pts : 2d numpy array
            Array of lon/lat points corresponding to all pixels in
            self.modis.data
        """
        new_mg = np.meshgrid(lon_pts, lat_pts)
        n_mg_v = np.vstack((new_mg[0].reshape(-1), new_mg[1].reshape(-1)))
        modis_pts = n_mg_v.T
        return modis_pts


class ModisClipper:
    """
    Clip MODIS data to extent of valid IMS data. This prevents
    assigning IMS values to MODIS outside of the IMS extents, and
    speeds NN mapping.

    Attributes
    ----------
    modis_idx : 2D numpy boolean array
        2D indices of MODIS values w/n IMS extent
    modis_clip : 2D numpy int16
        MODIS data clipped to IMS extent
    mlat_clip : 1D numpy float32
        MODIS latitudes clipped to IMS extent
    mlon_clip : 1D numpy float32
        MODIS longigutes clipped to IMS extent

    """
    def __init__(self, modis, ims):
        """
        Parameters
        ----------
        modis : ModisDay instance
            MODIS data to clip
        ims : ImsDay instance
            IMS data to clip MODIS extent to
        """
        self._modis = modis
        self._ims = ims

        # Ignore IMS Nodata pixels
        idata = self._ims.data.copy().flatten()
        mask = idata > 0

        # idata_good = idata[mask]
        ilat_good = self._ims.lat[mask]
        ilon_good = self._ims.lon[mask]

        # Get MODIS mask from IMS extents
        logger.info(f'Boundaries of valid IMS data: {ilon_good.min()} - ' +
                    f'{ilon_good.max()} long, {ilat_good.min()} - ' +
                    f'{ilat_good.max()} lat')
        mlat_idx = (ilat_good.min() - CLIP_MARGIN <= self._modis.lat) & \
                   (self._modis.lat <= ilat_good.max() + CLIP_MARGIN)
        mlon_idx = (ilon_good.min() - CLIP_MARGIN <= self._modis.lon) & \
                   (self._modis.lon <= ilon_good.max() + CLIP_MARGIN)
        self.modis_idx = np.ix_(mlat_idx, mlon_idx)

        # Clip out MODIS that matches IMS extents
        self.modis_clip = self._modis.data[self.modis_idx]
        self.mlat_clip = self._modis.lat[mlat_idx]
        self.mlon_clip = self._modis.lon[mlon_idx]

        logger.info(f'Boundaries of clipped MODIS data: ' +
                    f'{self.mlon_clip.min()} - ' +
                    f'{self.mlon_clip.max()} long, {self.mlat_clip.min()} - ' +
                    f'{self.mlat_clip.max()} lat')
        logger.info(f'Shape of clipped MODIS data is {self.modis_clip.shape}')
