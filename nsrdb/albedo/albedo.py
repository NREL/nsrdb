import numpy as np
import h5py
from scipy import ndimage
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import nsrdb.albedo.modis as modis
import nsrdb.albedo.ims as ims

class AlbedoError(Exception):
    pass


class CompositeAlbedoDay:
    # Value for snow/sea ice in IMS. In thousandths, e.g. 867 == 0.867
    SNOW_ALBEDO = 867

    @classmethod
    def run(cls, date, modis_path, ims_path, albedo_path):
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
        """
        cad = cls(date, modis_path, ims_path, albedo_path)

        print(f'Loading MODIS data for {cad.date}')
        cad.modis = modis.ModisDay(cad.date, cad.modis_path)

        print(f'Loading IMS data {cad.date}')
        cad.ims = ims.ImsDay(cad.date, cad.ims_path)

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

    def write_albedo(self, outfilename):
        """
        Write albedo data to HDF5 file

        Parameters
        ----------
        outfilename : string
            Name of HDF5 file to save
        """
        albedo_attrs = {'units': 'unitless',
                        'scale_factor': 1000}
        with h5py.File(outfilename, 'w') as f:
            f.create_dataset('surface_albedo', shape=self.albedo.shape,
                                dtype=self.albedo.dtype, data=self.albedo)
            for k, v in albedo_attrs.items():
                f['surface_albedo'].attrs[k] = v

            f.create_dataset('latitude', shape=self.modis.lat.shape,
                             dtype=self.modis.lat.dtype, data=self.modis.lat)
            f.create_dataset('longitude', shape=self.modis.lon.shape,
                             dtype=self.modis.lon.dtype, data=self.modis.lon)

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
        print(f'Calculating composite albedo for {self.date}')

        # Find snow/no snow region boundaries of IMS
        print('Determining IMS snow/no snow region boundaries')
        ims_bin_mskd, ims_pts = self._get_ims_boundary()

        # Create cKDTree to map MODIS points onto IMS regions
        print('Creating KD Tree')
        ims_tree = cKDTree(ims_pts)

        # Map MODIS pixels to IMS data
        print('Mapping MODIS to IMS data. This might take a while.')
        modis_pts = self._get_modis_pts()
        ind = self._run_futures(ims_tree, modis_pts)

        # Project nearest neighbors from IMS to MODIS. Array is on same grid as
        # MODIS, but has snow/no snow values from binary IMS.
        snow_no_snow = ims_bin_mskd[ind].reshape(len(self.modis.lat),
                                                 len(self.modis.lon))

        # Copy and update MODIS albedo for cells w/ snow
        albedo = self.modis.data.copy()
        albedo[snow_no_snow == 1] = self.SNOW_ALBEDO
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
        with ProcessPoolExecutor() as exe:
            for i, chunk in enumerate(chunks):
                future = exe.submit(self._run_single_tree, ims_tree, chunk)
                futures[future] = i
            print('Started all futures')

            for i, future in enumerate(as_completed(futures)):
                print(f'{i + 1} out of {len(futures)} futures completed.')
        print('done processing')

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

    def _get_modis_pts(self):
        """
        Create 2D numpy array representing lon/lats of MODIS pixels

        Returns
        -------
        modis_pts : 2d numpy array
            Array of lon/lat points corresponding to all pixels in
            self.modis.data
        """
        new_mg = np.meshgrid(self.modis.lon, self.modis.lat)
        n_mg_v = np.vstack((new_mg[0].reshape(-1), new_mg[1].reshape(-1)))
        modis_pts = n_mg_v.T
        return modis_pts

