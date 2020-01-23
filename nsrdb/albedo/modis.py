import os
import logging
import pyhdf
from pyhdf.SD import SD
import matplotlib.pyplot as plt
import urllib

from nsrdb.utilities.file_utils import url_download

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NODATA = 32767


class ModisError(Exception):
    pass


class ModisDay:
    """ Load MODIS data for a single day from disk """
    LAT_SIZE = 21600
    LON_SIZE = 43200

    def __init__(self, date, modis_path, shape=None):
        """
        Parameters
        ----------
        date : datetime instance
            Date to grab data for. Time is ignored
        modis_path : str
            Path to MODIS data files
        shape : (int, int)
            Shape of MODIS data. Defaults to normal values. Used for testing.
        """
        # logger.debug(f'Importing {dfile} into ModisDay')
        self.modis_path = modis_path
        if shape is None:
            self._shape = (self.LAT_SIZE, self.LON_SIZE)
        else:
            self._shape = shape

        self._filename = ModisFileAcquisition.get_file(date, modis_path)
        self.data, self.lon, self.lat = self._load_day()

    def _load_day(self):
        """
        Load albedo values, lon, and lat from HDF

        Returns
        -------
        data : 2D numpy array
            Array with MODIS dry earth albedo values. CRS appears to be WGS84
        lon : 1D numpy array
            Array with longitude values for MODIS data
        lat : 1D numpy array
            Array with latitude values for MODIS data
        """
        try:
            hdf = SD(self._filename)
        except pyhdf.error.HDF4Error as e:
            raise ModisError(f'Issue loading {self._filename}: {e}')

        try:
            print('Loading MODIS data')
            data = hdf.select('Albedo_Map_0.3-5.0')[:]
            print('Loading MODIS metadata')
            lat = hdf.select('Latitude')[:]
            lon = hdf.select('Longitude')[:]
            print('Completed loading MODIS data and metadata')
        except pyhdf.error.HDF4Error as e:
            raise ModisError(f'Error loading {self._filename}: {e}. File ' +
                             'does not have expected datasets and may be ' +
                             'too old.')

        if len(lat) != self._shape[0] or len(lon) != self._shape[1] or \
                data.shape != self._shape:
            msg = f'Data/metadata shapes are not correct for ' +\
                  f'{self._filename}. Data shape={data.shape}, ' +\
                  f'Lon shape={lon.shape}, Lat shape={lat.shape}. Data ' +\
                  f'shape is expected to be {self._shape}.'
            raise ModisError(msg)

        return data, lon, lat

    def plot(self):
        """ Plot data as map. Nodata is corrected so colors are sane """
        vals = self.data.copy()
        vals[vals > 1000] = 0
        plt.imshow(vals)
        plt.title(self._filename)
        plt.colorbar()
        plt.show()

    # TODO fix below
    # def __repr__(self):
        # return f'{self.__class__.__name__}(year={self.year}, day={self.day})'


class ModisFileAcquisition:
    """
    Class to acquire MODIS data for requested day. Attempts to get data from
    disk first. If not available the data is downloaded
    exist it is downloaded.
    """
    FTP_SERVER = 'rsftp.eeos.umb.edu'
    FTP_FOLDER = '/data02/Gapfilled/{year}/'
    FILE_PATTERN = 'MCD43GF_wsa_shortwave_{day}_{year}.hdf'
    # Example file name: MCD43GF_wsa_shortwave_033_2010.hdf

    @classmethod
    def get_file(cls, date, path):
        """
        Returns filename for MODIS date file for date. Searches in 'path' and
        downloads if necessary. MODIS files are every 8 days. Returns nearest
        day to 'data'.

        Parameters
        ----------
        date : Datetime object
            Desired data
        path : string
            Location of/for MODIS data on disk

        Returns
        -------
        filename : string
            Filename with path to MODIS data file
        """
        mfa = cls(date, path)

        # See if the file is on disk
        if os.path.isfile(os.path.join(path, mfa.filename)):
            print(f'{mfa.filename} found on disk at {path}')
        else:
            # Download it
            print(f'{mfa.filename} not found on disk, attempting to download')
            mfa._download()
        return os.path.join(mfa.path, mfa.filename)

    def __init__(self, date, path):
        """ See docstring for self.get_filename() """
        self.date = date
        self.path = path

        # Extract day as day of year (e.g. 1-366), left pad with 0
        day = self._nearest_modis_day(date.timetuple().tm_yday)
        self.day = str(day).zfill(3)
        self.year = str(date.year)

        # Example file name: MCD43GF_wsa_shortwave_033_2010.hdf
        self.filename = self.FILE_PATTERN.format(day=self.day, year=self.year)

    def _download(self):
        year_path = self.FTP_FOLDER.format(year=self.year)
        url = f'ftp://{self.FTP_SERVER}{year_path}{self.filename}'
        logger.info(f'Downloading {url}')
        try:
            fail = url_download(url, os.path.join(self.path, self.filename))
        # TODO below exception catching is not working
        except urllib.error.URLError as e:
            raise ModisError(f'Error while attempting to download {url}, ' +
                             e)
        if fail:
            raise ModisError(f'Error while attempting to download {url}')
        print(f'Successfully downloaded {url}')

    @staticmethod
    def _nearest_modis_day(day):
        """
        MODIS data is available in 8 day increments, e.g. 1, 9, 17, 25, etc.
        Finds nearest MODIS day to requested day. Days perfectly between
        available days are rounded up, e.g. day=5 returns 9.

        Parameters
        ---------
        day : int
            Requested day of year [1-366]

        Returns
        -------
        xxx : int
            Nearest MODIS day to 'day'
        """
        if day > 361:
            return 361
        if (day - 1) % 8 == 0:
            # day matches available day
            return day
        if (day - 1) % 8 < 4:
            # round down
            return day - (day - 1) % 8
        if (day - 1) % 8 >= 4:
            # round up
            return day - (day - 1) % 8 + 8
