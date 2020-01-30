import ftplib
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


logger = logging.getLogger(__name__)


class ImsError(Exception):
    """ General exception for IMS processing """
    pass


class ImsDataNotFound(ImsError):
    """ Raised when IMS data is not available on ftp server. This is typically
    caused by a missing day that needs to be gap-filled.
    """
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

    def __init__(self, date, ims_path, shape=None):
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
        """
        # TODO - possibly accept metadata so it doesn't have to be read
        self.ims_path = ims_path

        self.day = date.timetuple().tm_yday
        self.year = date.year

        # Download data (if necessary)
        _ifa = ImsFileAcquisition(date, ims_path)
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
        # TODO - Consider ignoring nodata pixels (value == 0)
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
    Grab IMS files from ftp. If they don't exist, attempt to temporal gap fill.
    """
    def __init__(self, date, path):
        """
        Parameters
        ----------
        date : Datetime object
            Desired date
        path : str
            Path of/for IMS data on disk
        """

        self.date = date  # data date
        self.path = path

        self.filename = None
        self.lat_file = None
        self.lon_file = None
        self.res = None

    def get_files(self):
        """
        Grab data from ftp or disk if it exists. If day is missing, check if
        gap filled data already exists. If not, create it.
        """
        ifa = ImsRealFileAcquisition(self.date, self.path)
        self.lat_file = ifa.lat_file
        self.lon_file = ifa.lon_file
        self.res = ifa.res

        # See if data for self.date exists on server or disk
        # TODO - this logic is painfully convoluted. Fix.
        try:
            ifa.get_files()
            self.filename = ifa.filename
        except ImsDataNotFound:
            # Doesn't exist, gap fill
            logger.info(f'Data is missing from server for {self.date}, '
                        f'looking for existing gap-filled file')
            missing_fname = ifa.filename
            self.filename = missing_fname + '.gf'

            # Check if gap fill file was already created
            if os.path.isfile(missing_fname + '.gf'):
                logger.info(f'Gap-filled file found, opening {missing_fname}' +
                            f'.gf')
            else:
                logger.info('Gap-filled file not found, creating gap-filled ' +
                            'data.')
                fa = ImsGapFill(self.date, self.path, missing_fname)
                fa.fill_gap()


class ImsGapFill:
    """
    Fill temporal gaps in IMS data set.
    """
    def __init__(self, date, path, missing_fname, sr=4):
        """
        Parameters
        ----------
        date : Datetime object
            Desired date.
        path : str
            Path of/for IMS data on disk.
        missing_fname : str
            Path and name of data file for missing date
        sr : int
            Number of days to search before and after a missing date for good
            data.
        """
        self.date = date
        self.path = path
        self._missing_fname = missing_fname
        self._sr = sr
        if sr < 1:
            raise ValueError('sr must be at least 1')

    def fill_gap(self):
        """
        Fill gaps in IMS data set for one year and save to disk. There may be
        multiple consecutive days of missing data. Try to find data within
        self._sr days of desired date.
        """
        logger.info(f'Looking for IMS data before/after {self.date}')
        for i in range(1, self._sr + 1):
            day_before = self.date - timedelta(days=i)
            logger.debug(f'Trying {i} day before missing date. {day_before}')
            # TODO - The use of IRFA to see if a file is on the server seems
            # wrong and is unintuitive.
            irfa = ImsRealFileAcquisition(day_before, self.path)
            if irfa.check_ftp_for_data():
                logger.info(f'Found good data {i} day(s) before missing day ' +
                            f'on {day_before}')
                break
        else:
            raise ImsError('No data found on ftp before {self.date}')

        # Search for data after missing days.
        for i in range(1, self._sr + 1):
            day_after = self.date + timedelta(days=i)
            logger.debug(f'Trying {i} day after missing date. {day_after}')
            irfa = ImsRealFileAcquisition(day_after, self.path)
            if irfa.check_ftp_for_data():
                logger.info(f'Found good data {i} day(s) after missing day ' +
                            f'on {day_after}')
                break
        else:
            raise ImsError(f'No data found on ftp before {self.date}')

        logger.info(f'Creating gap-fill data for {self.date}')
        # TODO - Currently just using data from day before. Improve algorithm
        i = ImsDay(day_before, self.path)

        # Data is stored in asc file starting at bottom left cell. Flip.
        self._write_data(np.flipud(i.data))

    def _write_data(self, data):
        """ Write gap filled data to disk with .gf extension """
        meta_header = f'Temporal gap filled data for {self.date}\n'

        # Write masked IMS data to disk
        with open(self._missing_fname + '.gf', 'wt') as f:
            f.write(meta_header)
            # write each row of data as a string
            for r in data:
                txt = ''.join(r.astype(str))
                f.write(txt)
                f.write('\n')


class ImsRealFileAcquisition:
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

    def __init__(self, date, path):
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
        """

        if date < self.EARLIEST_SUPPORTED:
            raise ImsError(f'Dates before {self.EARLIEST_SUPPORTED} are not' +
                           ' currently supported.')
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
        # Metadata
        if os.path.isfile(self.lon_file) and os.path.isfile(self.lat_file):
            logger.info(f'IMS metadata found on disk at {self.path}')
        else:
            logger.info(f'IMS metadata not found on disk, attempting to ' +
                        'download')
            self._download_metadata(self._mf.lon_remote, self._mf.lat_remote)
            self._uncompress(self._mf.lon_remote)
            self._uncompress(self._mf.lat_remote)

        # Data
        if os.path.isfile(self.filename):
            logger.info(f'{self._pfilename} found on disk at {self.path}')
        else:
            logger.info(f'{self._pfilename} not found on disk, attempting to' +
                        ' download')
            if not self.check_ftp_for_data():
                raise ImsDataNotFound(f'{self._pfilename}.gz not found on ' +
                                      f'{self.FTP_SERVER}')
            self._download_data()
            self._uncompress(self._pfilename + '.gz')

    def check_ftp_for_data(self):
        """
        Check for existence of data on ftp server. Returns True if file exists,
        otherwise False.
        """
        ftp_path = self.FTP_FOLDER.format(year=self._file_year, res=self.res)

        ftp = ftplib.FTP(self.FTP_SERVER)
        ftp.login()
        ftp.cwd(ftp_path)
        rfiles = []
        ftp.retrlines('LIST', rfiles.append)
        # LIST provides ls -ls style output, simplify to filenames
        rfiles = [x.split()[8] for x in rfiles]
        return f'{self._pfilename}.gz' in rfiles

    def _download_data(self):
        """ Download IMS data file and save to disk"""
        ftp_path = self.FTP_FOLDER.format(year=self._file_year, res=self.res)
        url = f'ftp://{self.FTP_SERVER}{ftp_path}{self._pfilename}.gz'
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
