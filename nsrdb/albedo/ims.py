# import ftplib
import os
import logging
# import glob
import numpy as np
import re
import matplotlib.pyplot as plt
import urllib
import tarfile
import gzip
import shutil
from datetime import datetime, timedelta

from nsrdb.utilities.file_utils import url_download
from nsrdb.utilities.loggers import init_logger


logger = logging.getLogger(__name__)


class ImsError(Exception):
    pass


class ImsDay:
    """ Load IMS data for a single day from disk """
    # Available resolutions
    RES_1KM = '1km'
    RES_4KM = '4km'

    # Number of pixels for width/height for IMS data resolutions
    # '24km': 1024,
    pixels = {RES_4KM: 6144,
              RES_1KM: 24576}

    def __init__(self, date, ims_path, shape=None, log_level='INFO',
                 log_file=None):
        """
        Parameters
        ----------
        date : datetime instance
            Date to grab data for. Time is ignored.
        ims_path : str
            Path to IMS data files.
        shape : (int, int)
            Tuple of data shape in (rows, cols) format. Defaults to standard
            values for 1- and 4-km grid. Used for testing.
        log_level : str
            Level to log messages at.
        log_file : str
            File to log messages to
        """
        # TODO - possibly accept metadata so it doesn't have to be read
        init_logger(__name__, log_file=log_file, log_level=log_level)

        self.ims_path = ims_path

        self.day = date.timetuple().tm_yday
        self.year = date.year

        # Download data (if necessary)
        _ifa = ImsFileAcquisition(date, ims_path, log_level=log_level,
                                  log_file=log_file)
        _ifa.get_files()
        self._filename = _ifa.filename
        self._lat_file = _ifa.lat_file
        self._lon_file = _ifa.lon_file

        self.res = _ifa.res
        if shape is None:
            # Assume standard shape based on resolution
            self._shape = (self.pixels[self.res], self.pixels[self.res])
        else:
            # Use customize data shape (i.e., for testing)
            self._shape = shape

        logger.info('Loading IMS data')
        self.data = self._load_data()
        logger.info('Loading IMS metadata')
        self.lon, self.lat = self._load_meta()
        logger.info('Completed loading IMS data and metadata')

    def _load_data(self):
        """
        Load IMS values from asc file. For format see
        https://nsidc.org/data/g02156 -> "User Guide" tab

        Returns
        -------
        ims_data : 2D numpy array (int8)
            IMS snow values [0, 1, 2, 3, 4] in polar projection
        """
        raw = []
        with open(self._filename, 'r') as dat:
            lines = dat.readlines()
            for line in lines:
                # asc file has text header then rows of [0,1,2,3]
                if re.search('[a-z]', line.strip()) is None:
                    raw.extend([int(l) for l in list(line.strip())])

        # IMS data sanity check
        length = self._shape[0]*self._shape[1]
        if len(raw) != length:
            msg = f'Data length in {self._filename} is expected to be ' + \
                f'{length} but is {len(raw)}.'
            raise ImsError(msg)

        # Reshape data to square, size dependent on resolution, and flip
        ims_data = np.flipud(np.array(raw).reshape(self._shape))
        ims_data = ims_data.astype(np.int8)

        logger.info(f'IMS data shape is {ims_data.shape}')
        return ims_data

    def _load_meta(self):
        """
        Load IMS metadata (lat/lon values) from .double or .bin  file. For
        format see https://nsidc.org/data/g02156 -> "User Guide" tab

        Returns
        -------
        lon, lat : 1D numpy arrays
            Longitude and latitudes for IMS data. IMS data is stored in a polar
            projection so lon/lat is specifically defined for each pixel. The
            lon/lat arrays are the same length as the IMS data.
        """
        length = self._shape[0]*self._shape[1]
        # The 1km and 4km data are stored in different formats
        if self.res == self.RES_1KM:
            with open(self._lat_file, 'rb') as f:
                lat = np.fromfile(f, dtype='<d', count=length)\
                    .astype(np.float32)
            with open(self._lon_file, 'rb') as f:
                lon = np.fromfile(f, dtype='<d', count=length)\
                    .astype(np.float32)
        else:
            # 4km
            lat = np.fromfile(self._lat_file, dtype='<f4')
            lon = np.fromfile(self._lon_file, dtype='<f4')

        # Longitude might be stored as 0-360, fix
        lon = np.where(lon > 180, lon - 360, lon)

        # Meta data sanity checks
        if lon.shape != (length,):
            msg = f'Shape of {self._lon_file} is expected to be ({length},)' +\
                   f' but is {lon.shape}.'
            raise ImsError(msg)
        if lat.shape != (length,):
            msg = f'Shape of {self._lat_file} is expected to be ({length},)' +\
                   f' but is {lat.shape}.'
        return lon, lat

    def plot(self):
        """ Plot values as map. """
        plt.imshow(self.data)
        plt.title(self._filename)
        plt.colorbar()
        plt.show()

    def __repr__(self):
        return f'{self.__class__.__name__}(year={self.year}, day={self.day})'


def get_dt(year, day):
    """
    Return datetime instance for year and day [1-366]

    Parameters
    ---------
    year : int
        Desired year
    day : int
        Desired day of year, from 1 to 366

    Returns
    -------
    datetime instance
    """
    return datetime(year, 1, 1) + timedelta(days=day - 1)


class ImsFileAcquisition:
    """
    Class to acquire IMS data for requested day. Attempts to get data from
    disk first. If not available the data is downloaded exist it is downloaded.

    Files are acquired by calling self.get_files() after class is initialized.

    It should be noted that for dates on and after 2014, 336, (Ver 1.3) the
    file date is one day after the data date.
    """
    FTP_SERVER = 'sidads.colorado.edu'
    FTP_FOLDER = '/DATASETS/NOAA/G02156/{res}/{year}/'
    FTP_METADATA_FOLDER = '/DATASETS/NOAA/G02156/metadata/'
    # Example file name: ims2015010_4km_v1.3.asc.gz
    FILE_PATTERN = 'ims{year}{day}_{res}_{ver}.asc'

    EARLIEST_1KM = get_dt(2014, 336)
    EARLIEST_VER_1_3 = get_dt(2014, 336)
    EARLIEST_SUPPORTED = datetime(2004, 2, 22)

    def __init__(self, date, path, log_level='INFO', log_file=None):
        """
        Attributes
        ----------
        self.filename : str
            Path and filename for IMS data file
        self.lon_file : str
            Path and filename for longitude data file
        self.lat_file : str
            Path and filename for latitude data file
        self.res: str
            Resolution of data

        Parameters
        ----------
        date : Datetime object
            Desired date
        path : str
            Path of/for IMS data on disk
        log_level : str
            Level to log messages at.
        log_file : str
            File to log messages to
        """
        init_logger(__name__, log_file=log_file, log_level=log_level)
        init_logger('nsrdb.utilities.file_utils', log_file=log_file,
                    log_level=log_level)

        if date < self.EARLIEST_SUPPORTED:
            raise ImsError(f'Dates before {self.EARLIEST_SUPPORTED} are not' +
                           ' current supported.')
        self.date = date  # data date
        self.path = path

        if self.date >= self.EARLIEST_1KM:
            self.res = '1km'
        else:
            self.res = '4km'

        if self.date >= self.EARLIEST_VER_1_3:
            self.ver = 'v1.3'
        else:
            self.ver = 'v1.2'

        # For ver == 1.3, date in filename is the day after the data date
        # See https://nsidc.org/data/g02156 -> User Guide
        file_date = self.date
        if self.ver == 'v1.3':
            file_date += timedelta(days=1)

        self._file_day = str(file_date.timetuple().tm_yday).zfill(3)
        self._file_year = str(file_date.year)

        self._pfilename = self.FILE_PATTERN.format(year=self._file_year,
                                                   day=self._file_day,
                                                   res=self.res,
                                                   ver=self.ver)

        self.filename = os.path.join(path, self._pfilename)

        self._mf = MetaFiles(self.res)
        self.lon_file = os.path.join(path, self._mf.lon_file)
        self.lat_file = os.path.join(path, self._mf.lat_file)

    def get_files(self):
        """
        Check if IMS data and metadata is on disk and download if necessary.
        """
        # Data
        if os.path.isfile(self.filename):
            logger.info(f'{self._pfilename} found on disk at {self.path}')
        else:
            logger.info(f'{self._pfilename} not found on disk, attempting to' +
                        ' download')
            self._download_data()
            self._uncompress(self._pfilename + '.gz')

        # Metadata
        if os.path.isfile(self.lon_file) and os.path.isfile(self.lat_file):
            logger.info(f'IMS metadata found on disk at {self.path}')
        else:
            logger.info(f'IMS metadata not found on disk, attempting to ' +
                        'download')
            self._download_metadata(self._mf.lon_remote, self._mf.lat_remote)
            self._uncompress(self._mf.lon_remote)
            self._uncompress(self._mf.lat_remote)

    def _download_data(self):
        """ Download IMS data file and save to disk"""
        year_path = self.FTP_FOLDER.format(year=self._file_year, res=self.res)
        url = f'ftp://{self.FTP_SERVER}{year_path}{self._pfilename}.gz'
        self.__download(url, os.path.join(self.path, self._pfilename + '.gz'))

    def _download_metadata(self, lon_file, lat_file):
        """
        Download IMS meta data

        Parameters
        ----------
        lon_file, lat_file : str
            Names of metadata files
        """
        url = f'ftp://{self.FTP_SERVER}{self.FTP_METADATA_FOLDER}' + \
              f'{lon_file}'
        self.__download(url, os.path.join(self.path, lon_file))

        url = f'ftp://{self.FTP_SERVER}{self.FTP_METADATA_FOLDER}' + \
              f'{lat_file}'
        self.__download(url, os.path.join(self.path, lat_file))

    def _uncompress(self, filename):
        """
        Ungzip/untar file to self.path

        Parameters
        ----------
        filename : str
            File to untar/ungzip
        """
        logger.info(f'Uncompressing {filename}')
        filename = os.path.join(self.path, filename)

        if filename.split('.')[-2] == 'tar':
            with tarfile.open(filename) as tar:
                tar.extractall(self.path)
        else:
            with gzip.open(filename, 'r') as f_in:
                with open(filename[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    @staticmethod
    def __download(url, lfile):
        """
        Download and save file via ftp

        Parameters
        ----------
        url : str
            full url to download
        lfile : str
            Path and filename to save data as
        """
        logger.info(f'Downloading {url}')
        try:
            fail = url_download(url, lfile)
        # TODO below exception catching is not working
        except urllib.error.URLError as e:
            raise ImsError(f'Error while attempting to download {url}, ' +
                           str(e))
        if fail:
            raise ImsError(f'Error while attempting to download {url}')
        logger.info(f'Successfully downloaded {url}')


class MetaFiles:
    """ IMS metadata filename handler """
    def __init__(self, res):
        """
        Parameters
        ----------
        res : str ['1km', '4km']
            Desired resolution
        """
        self.res = res

    @property
    def lon_remote(self):
        if self.res == '1km':
            return 'IMS1kmLons.24576x24576x1.tar.gz'
        else:
            return 'imslon_4km.bin.gz'

    @property
    def lat_remote(self):
        if self.res == '1km':
            return 'IMS1kmLats.24576x24576x1.tar.gz'
        else:
            return 'imslat_4km.bin.gz'

    @property
    def lon_file(self):
        if self.res == '1km':
            return 'IMS1kmLons.24576x24576x1.double'
        else:
            return 'imslon_4km.bin'

    @property
    def lat_file(self):
        if self.res == '1km':
            return 'IMS1kmLats.24576x24576x1.double'
        else:
            return 'imslat_4km.bin'


class ImsGapFill:
    """
    Fill gaps in IMS data set. First attempt to perform temporal gap fill,
    then attempt spatial.
    """
    def __init__(self, year):
        pass

    @classmethod
    def fill_gaps(cls, year):
        """
        Fill gaps in IMS data set for one year.

        And then save to disk????
        """
        pass
