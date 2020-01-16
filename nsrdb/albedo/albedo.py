import modis
import ims
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed


class AlbedoError(Exception):
    pass


class CompositeAlbedoDay:
    @classmethod
    def run(cls, date, modis_path, ims_path, albedo_path, nn_indicies, res):
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
        nn_indices : numpy array
            Indices mapping IMS data to MODIS data

        """
        # TODO remove res from parameters or add to docstring

        cad = cls(date, modis_path, ims_path, albedo_path, nn_indicies)
        cad.modis = cad.load_modis()
        # cad.modis.plot()
        cad.ims = cad.load_ims(res)
        #albedo = cad.calc_albedo()
        #cad.write_albedo(albedo)

        # TODO delete below
        return cad

    def __init__(self, date, modis_path, ims_path, albedo_path, nn_indicies):
        """ See parameter definitions for self.run() """
        self.date = date
        self.modis_path = modis_path
        self.ims_path = ims_path
        self.albedo_path = albedo_path
        self.nn_indicies = nn_indicies

        self.modis = None  # ModisDay object
        self.ims = None  # ImsDay object

        # TODO - handle below value in a better way
        self.snow_albedo_value = 1000

    def load_modis(self):
        """ Load and return ModisDay instance """
        print(f'loading {self.date} from modis')
        # ModisDay is responsible for getting the closest day
        return modis.ModisDay(self.date, self.modis_path)

    def load_ims(self, res):
        """ Load and return ImsDay instance """
        print(f'loading {self.date} from ims')
        # TODO improve resolution handling
        return ims.ImsDay(self.date, self.ims_path, res)

    def calc_albedo(self):
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
        ims_bin_mskd, ims_pts = self._get_ims_boundary()

        # Create cKDTree to map MODIS points onto IMS regions
        ims_tree = cKDTree(ims_pts)

        # Map MODIS pixels to IMS data
        modis_pts = self._get_modis_pts()
        ind = self._run_futures(ims_tree, modis_pts)

        # Project nearest neighbors from IMS to MODIS. Array is on same grid as
        # MODIS, but has snow/no snow values from binary IMS.
        snow_no_snow = ims_bin_mskd[ind].reshape(len(self.modis.lat),
                                                 len(self.modis.lon))

        # Copy and update MODIS albedo for cells w/ snow
        albedo = self.modis.data.copy()
        albedo[snow_no_snow == 1] = self.snow_albedo_value
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
        # TODO - calculate number of cores automatically
        chunks = np.array_split(modis_pts, 36)
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
        Create IMS boundary layer which represents the pixels that from the
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

    @staticmethod
    def write_albedo(albedo):
        """
        Write albedo data to self.albedo_path.

        Parameters
        ----------
        albedo : Albedo instance
            Albedo data from self.calc_albedo()
        """
        pass


#############################################
##  Code below here is all obsolete

class CalcCompositeAlbedo:
    """
    Calculate composite albedo from MODIS and IMS for an arbitrary period of
    time.
    """
    @classmethod
    def run(cls, start_date, end_date, modis_path, ims_path, albedo_path):
        """
        Parameters
        ----------
        start_date : datetime instance
            Beginning of time frame to calculate albedo for
        start_date : datetime instance
            Beginning of time frame to calculate albedo for
        modis_path : str
            Path for MODIS data files
        ims_path : str
            Path for IMS data files
        albedo_path : str
            Path for composite albedo data files (output)
        """
        cca = cls(start_date, end_date, modis_path, ims_path, albedo_path)
        cca.date_range = cca.calc_date_range()
        cca.download_modis()
        cca.download_ims()
        cca.fill_ims_gaps()

        # Calculate mapping between IMS and MODIS
        nn_indicies = cca.calc_indicies()

        for date in cca.date_range:
            CompositeAlbedoDay.run(date, modis_path, ims_path, albedo_path,
                                   nn_indicies)

    def _init__(self, start_date, end_date, modis_path, ims_path, albedo_path):
        """ See parameter definitions in self.run() """
        self.start_date = start_date
        self.end_date = end_date
        self.modis_path = modis_path
        self.ims_path = ims_path
        self.albedo_path = albedo_path

    def calc_date_range(self):
        """ Calculate and return date range for analysis """
        pass

    def download_modis(self):
        """ Download MODIS data for date range """
        pass

    def download_ims(self):
        """ Download IMS data for date range """
        pass

    def fill_ims_gaps(self):
        """ Fill any temporal gaps in IMS data """
        pass
