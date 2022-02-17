"""
Classes to acquire and load IMS snow data. Entrance via ImsDay.

Mike Bannister
2/18/2020
"""
import ftplib
import gzip
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import shutil
import tarfile
from time import sleep
from datetime import datetime, timedelta

from nsrdb.utilities.file_utils import url_download

logger = logging.getLogger(__name__)

# Earliest available date for IMS data
EARLIEST_AVAILABLE = datetime(1997, 2, 4)  # 035

# Available resolutions
IMS_RES_1KM = '1km'
IMS_RES_4KM = '4km'
IMS_RES_24KM = '24km'

FTP_SERVER = 'sidads.colorado.edu'
FTP_FOLDER = '/DATASETS/NOAA/G02156/{res}/{year}/'
FTP_METADATA_FOLDER = '/DATASETS/NOAA/G02156/metadata/'

FTP_RETRY_DELAY = 60  # seconds
FTP_RETRIES = 15  # number of times to retry ftp before giving up


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


def uncompress(filename):
    """
    Ungzip/untar file

    Parameters
    ----------
    filename : str
        File to untar/ungzip
    """
    logger.info(f'Uncompressing {filename}')
    path = os.path.dirname(os.path.realpath(filename))

    if filename.split('.')[-2] == 'tar':
        with tarfile.open(filename) as tar:
            tar.extractall(path)
    else:
        with gzip.open(filename, 'r') as f_in:
            with open(filename[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def download(url, lfile):
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

    tries = 0
    while True:
        fail = url_download(url, lfile)
        if not fail:
            break

        if tries >= FTP_RETRIES:
            msg = 'Error downloading IMS data after multiple tries'
            logger.exception(msg)
            raise ImsError(msg)
        logger.info('Server busy, waiting and trying to download again')
        sleep(FTP_RETRY_DELAY)
        tries += 1

    logger.info(f'Successfully downloaded {url}')


class ImsError(Exception):
    """ General exception for IMS processing """


class ImsDataNotFound(ImsError):
    """
    Raised when IMS data is not available on ftp server. This is typically
    caused by a missing day that needs to be gap-filled.
    """


class ImsDay:
    """ Load IMS data for a single day. Download if necessary """

    def __init__(self, date, path, shape=None):
        """
        Parameters
        ----------
        date : datetime instance
            Date to grab data for. Time is ignored.
        path : str
            Path to/for IMS data files.
        shape : (int, int) | None
            Tuple of data shape in (rows, cols) format. Defaults to standard
            values for 1-, 4-, and 24-km grid. Used for testing.
        """
        if date < EARLIEST_AVAILABLE:
            msg = f'Requested date of {date} predates earliest IMS ' +\
                  f'availability of {EARLIEST_AVAILABLE}'
            logger.exception(msg)
            raise ImsError(msg)

        self.path = path
        self._day = date.timetuple().tm_yday
        self._year = date.year

        ida = ImsDataAcquisition(date, path, shape=shape)
        ida.get_files()
        self._lat_file = ida.lat_file
        self._lon_file = ida.lon_file
        self.shape = ida.shape
        self.res = ida.res
        self.data = ida.data

        logger.info('Loading IMS metadata')
        lon_file, lat_file = self._find_meta()
        self.lon, self.lat = self._load_meta(lon_file, lat_file)
        logger.info('Completed loading IMS data and metadata')

    def _find_meta(self):
        """
        See if meta data is on disk and download if not

        Returns
        -------
        lon_file, lat_file : str
            Names of metadata files
        """
        mf = MetaFiles(self.res)
        lon_file = os.path.join(self.path, mf.lon_file)
        lat_file = os.path.join(self.path, mf.lat_file)

        if os.path.isfile(lon_file) and os.path.isfile(lat_file):
            logger.info('IMS metadata found on disk at '
                        f'{os.path.realpath(lon_file)} and '
                        f'{os.path.realpath(lat_file)}.')
        else:
            logger.info('IMS metadata not found on disk, attempting to '
                        'download')
            self._download_metadata(mf.lon_remote, mf.lat_remote)
            uncompress(os.path.join(self.path, mf.lon_remote))
            uncompress(os.path.join(self.path, mf.lat_remote))

        return lon_file, lat_file

    def _download_metadata(self, lon_file, lat_file):
        """
        Download IMS meta data

        Parameters
        ----------
        lon_file, lat_file : str
            Names of metadata files
        """
        url = f'ftp://{FTP_SERVER}{FTP_METADATA_FOLDER}{lon_file}'
        download(url, os.path.join(self.path, lon_file))

        url = f'ftp://{FTP_SERVER}{FTP_METADATA_FOLDER}{lat_file}'
        download(url, os.path.join(self.path, lat_file))

    def _load_meta(self, lon_file, lat_file):
        """
        Load IMS metadata (lat/lon values) from .double or .bin  file. For
        format see https://nsidc.org/data/g02156 -> "User Guide" tab

        Parameters
        ----------
        lon_file, lat_file : str
            Names of metadata files

        Returns
        -------
        lon, lat : 1D numpy arrays
            Longitude and latitudes for IMS data. IMS data is stored in a polar
            projection so lon/lat is specifically defined for each pixel. The
            lon/lat arrays are the same length as the IMS data.
        """
        length = self.shape[0] * self.shape[1]
        # The 1km and 4km/24km metadata are stored in different formats
        if self.res == IMS_RES_1KM:
            with open(lat_file, 'rb') as f:
                lat = np.fromfile(f, dtype='<d', count=length)\
                    .astype(np.float32)
            with open(lon_file, 'rb') as f:
                lon = np.fromfile(f, dtype='<d', count=length)\
                    .astype(np.float32)
        else:
            lat = np.fromfile(lat_file, dtype='<f4')
            lon = np.fromfile(lon_file, dtype='<f4')

        # Longitude might be stored as 0-360, fix
        lon = np.where(lon > 180, lon - 360, lon)

        # Meta data sanity checks
        if lon.shape != (length,):
            msg = (f'Shape of {lon_file} is expected to be ({length},)'
                   f' but is {lon.shape}.')
            logger.error(msg)
            raise ImsError(msg)
        if lat.shape != (length,):
            msg = (f'Shape of {lat_file} is expected to be ({length},)'
                   f' but is {lat.shape}.')
            logger.error(msg)
            raise ImsError(msg)
        return lon, lat

    def plot(self):
        """ Plot values as map. """
        plt.imshow(self.data)
        plt.title(f'{self._year} - {self._day}')
        plt.colorbar()
        plt.show()

    def __repr__(self):
        return f'{self.__class__.__name__}(year={self._year}, day={self._day})'


class ImsDataAcquisition:
    """
    Get IMS data for a date, either from disc, server, or via gapfill.
    """
    def __init__(self, date, path, shape=None):
        """
        Parameters
        ----------
        date : Datetime object
            Desired date
        path : str
            Path of/for IMS data on disk
        shape : (int, int) | None
            Tuple of data shape in (rows, cols) format. Defaults to standard
            values for 1-, 4-, or 24-km grid. Used for testing.
        """

        self.date = date
        self.path = path
        self.shape = shape

        self.lat_file = None
        self.lon_file = None
        self.res = None
        self.data = None

    def get_files(self):
        """
        Grab data from ftp or disk if it exists. If day is missing, use
        gap-filled data.
        """
        ifa = ImsFileAcquisition(self.date, self.path, shape=self.shape)

        try:
            ifa.get_file()
            self.data = ifa.data
        except ImsDataNotFound:
            logger.info(f'Data is missing or bad for {self.date}, '
                        'attempting to gap fill')
            fa = ImsGapFill(self.date, self.path)
            ifa = fa.fill_gap()

        self.res = ifa.res
        self.data = ifa.data
        self.shape = ifa.shape


class ImsGapFill:
    """
    Fill temporal gaps in IMS data set.
    """
    def __init__(self, date, path, search_range=4):
        """
        Parameters
        ----------
        date : Datetime object
            Desired date.
        path : str
            Path of/for IMS data on disk.
        search_range : int
            Number of days to search before and after a missing date for good
            data.
        """
        self.date = date
        self.path = path
        if search_range < 1:
            raise ValueError('search_range must be at least 1')
        self._search_range = search_range

    def fill_gap(self):
        """
        Fill gaps in IMS data set for one year and save to disk. There may be
        multiple consecutive days of missing data. Try to find data within
        self._search_range days of desired date. Will use existing gap-fill
        file if found.

        Returns
        _______
        ImsFileAcquisition instance
            File data for closest day found
        """
        logger.info(f'Attempting to gap-fill data for {self.date}')

        ifa = ImsFileAcquisition(self.date, self.path, gap_fill=True)
        gap_fill_fname = ifa.filename

        try:
            ifa.get_file()
            logger.info('Gap-fill data found on disk')
            return ifa
        except ImsDataNotFound:
            pass

        ifa = self._find_closest_day()

        # Data is stored in asc file starting at bottom left cell. Flip.
        self._write_data(np.flipud(ifa.data), gap_fill_fname)
        logger.info(f'Finished gap-fill data for {self.date}')
        return ifa

    def _find_closest_day(self):
        """
        Find closest day with IMS data to self.date. Raise ImsError if none
        found.

        Returns
        -------
        ImsFileAcquisition instance
            File data for closest day found
        """
        dates_list = []
        for i in range(1, self._search_range + 1):
            day_before = self.date - timedelta(days=i)
            day_after = self.date + timedelta(days=i)
            dates_list.extend([(day_before, i, 'before'),
                               (day_after, i, 'after')])

        for date, i, direction in dates_list:
            logger.debug(f'Trying {i} day {direction} missing date: {date}')
            ifa = ImsFileAcquisition(date, self.path)

            try:
                ifa.get_file()
            except ImsDataNotFound:
                continue

            logger.info(f'Found good data {i} day(s) {direction} missing day '
                        f'on {date}')
            return ifa

        msg = f'No valid data found on ftp within search range of {self.date}'
        logger.error(msg)
        raise ImsError(msg)

    def _write_data(self, data, gap_fill_fname):
        """
        Write gap filled data to disk with .gf extension

        Parameters
        ----------
        data : numpy.ndarray
            Data to save in gap-filled file
        gap_fill_fname : str
            File name to save data in
        """
        with open(gap_fill_fname, 'wt', encoding='utf-8') as f:
            f.write(f'Temporal gap filled data for {self.date}\n')
            # write each row of data as a string
            for r in data:
                txt = ''.join(r.astype(str))
                f.write(txt)
                f.write('\n')


class ImsFileAcquisition:
    """
    Class to IMS data file for requested day. Attempts to get data from
    disk first. If not on disk the data is downloaded.

    Files are acquired and loaded by calling self.get_file() after class is
    initialized. ImsDataNotFound is raised if there is any issue obtaining
    or loading data.

    It should be noted that for dates on and after 2014, 336, (Ver 1.3) the
    file date is one day after the data date.
    """
    # Example file name: ims2015010_4km_v1.3.asc
    FILE_PATTERN = 'ims{year}{day}_{res}_{ver}.asc'

    EARLIEST_4KM = get_dt(2004, 55)
    EARLIEST_1KM = get_dt(2014, 336)
    EARLIEST_VER_1_3 = get_dt(2014, 336)
    EARLIEST_VER_1_2 = get_dt(2004, 54)

    # Number of pixels for width/height for IMS data resolutions
    PIXELS = {IMS_RES_24KM: 1024, IMS_RES_4KM: 6144, IMS_RES_1KM: 24576}

    def __init__(self, date, path, shape=None, gap_fill=False):
        """
        Attributes
        ----------
        self.filename : str
            Path and filename for IMS data file
        self.res : str
            Resolution of data
        self.ver : str
            Version of IMS data
        self.shape : (int, int)
            Shape of data in (rows, cols)
        self.data : numpy.ndarray
            IMS data

        Parameters
        ----------
        date : Datetime object
            Desired date
        path : str
            Path of/for IMS data on disk
        shape : (int, int) | None
            Tuple of data shape in (rows, cols) format. Defaults to standard
            values for 1- and 4-km grid. Used for testing.
        gap_fill : bool
            Look for gap-filled file on disk. Don't attempt to download.
        """
        self.date = date  # data date
        self.path = path
        self._gap_fill = gap_fill

        if self.date < self.EARLIEST_4KM:
            self.res = IMS_RES_24KM
        elif self.date < self.EARLIEST_1KM:
            self.res = IMS_RES_4KM
        else:
            self.res = IMS_RES_1KM

        if shape is None:
            self.shape = (self.PIXELS[self.res], self.PIXELS[self.res])
        else:
            self.shape = shape

        if self.date < self.EARLIEST_VER_1_2:
            self.ver = 'v1.1'
        elif self.date < self.EARLIEST_VER_1_3:
            self.ver = 'v1.2'
        else:
            self.ver = 'v1.3'

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
        if gap_fill:
            self._pfilename += '.gf'

        self.filename = os.path.join(path, self._pfilename)
        self._data = None

    @property
    def data(self):
        """
        Return IMS data if loaded

        Returns
        -------
        numpy.ndarray
            IMS Data
        """
        if self._data is None:
            msg = 'Data is not loaded please call get_file() first.'
            logger.exception(msg)
            raise ImsError(msg)
        return self._data

    def get_file(self):
        """
        Check if IMS data is on disk and download if necessary.
        """
        if os.path.isfile(self.filename):
            logger.info(f'{self._pfilename} found on disk at '
                        f'{os.path.abspath(self.path)}.')
        else:
            if self._gap_fill:
                raise ImsDataNotFound(f'Gap-fill file {self._pfilename} not '
                                      'found on disk')

            logger.info(f'{self._pfilename} not found on disk, attempting to'
                        ' download')
            if not self._check_ftp_for_data():
                raise ImsDataNotFound(f'{self._pfilename}.gz not found on '
                                      f'{FTP_SERVER}')
            self._download_data()
            uncompress(os.path.join(self.path, self._pfilename + '.gz'))

        self._data = self._load_data()

    def _check_ftp_for_data(self):
        """
        Check for existence of data on ftp server. Returns True if file exists,
        otherwise False.
        """
        ftp_path = FTP_FOLDER.format(year=self._file_year, res=self.res)

        tries = 0
        while True:
            try:
                ftp = ftplib.FTP(FTP_SERVER)
                ftp.login()
                ftp.cwd(ftp_path)
                rfiles = []
                ftp.retrlines('LIST', rfiles.append)
                break
            except (ftplib.error_temp, TimeoutError) as e:
                if tries >= FTP_RETRIES:
                    msg = f'Error contacting FTP server for IMS data: {e}'
                    logger.exception(msg)
                    raise ImsError(msg) from e
                logger.info('FTP server busy, waiting and trying again')
                sleep(FTP_RETRY_DELAY)
                tries += 1

        # ftp LIST provides ls -ls style output, simplify to filenames
        rfiles = [x.split()[8] for x in rfiles]
        return f'{self._pfilename}.gz' in rfiles

    def _download_data(self):
        """ Download IMS data file and save to disk"""
        ftp_path = FTP_FOLDER.format(year=self._file_year, res=self.res)
        url = f'ftp://{FTP_SERVER}{ftp_path}{self._pfilename}.gz'
        download(url, os.path.join(self.path, self._pfilename + '.gz'))

    def _load_data(self):
        """
        Load IMS values from asc file. The file has a text header that should
        be ignored. Data rows may have spaces between values (unpacked format)
        or may be a be a continuous string of single digit values (packed
        format) For more format information see https://nsidc.org/data/g02156
        -> "User Guide" tab.

        Returns
        -------
        ims_data : 2D numpy array (int8)
            IMS snow values [0, 1, 2, 3, 4] in polar projection
        """
        logger.info('Loading IMS data')
        raw = []
        with open(self.filename, 'r', encoding='utf-8') as dat:
            lines = dat.readlines()
            packed = None
            for line in lines:
                line = line.strip()

                if re.search('[a-z]', line) is not None or line == '':
                    continue

                if packed is None:
                    packed = ' ' not in line

                if packed:
                    vals = [int(val) for val in list(line)]
                else:
                    vals = [int(val) for val in line.split()]
                raw.extend(vals)

        # IMS data sanity check
        length = self.shape[0] * self.shape[1]
        if len(raw) != length:
            msg = (f'Data length in {self.filename} is expected to be '
                   f'{length} but is {len(raw)}.')
            logger.warning(msg)
            raise ImsDataNotFound(msg)

        # Changed unpacked snow/ice values to match packed format
        raw = np.array(raw)
        raw[raw == 164] = 3  # Sea ice
        raw[raw == 165] = 4  # Snow

        # Reshape data to square, size dependent on resolution, and flip
        ims_data = np.flipud(raw.reshape(self.shape))
        ims_data = ims_data.astype(np.int8)

        logger.debug(f'IMS data shape is {ims_data.shape}')
        return ims_data


class MetaFiles:
    """ IMS metadata filename handler """
    def __init__(self, res):
        """
        Parameters
        ----------
        res : str ['1km', '4km', '24km']
            Desired resolution
        """
        assert res in [IMS_RES_1KM, IMS_RES_4KM, IMS_RES_24KM]
        self.res = res

    @property
    def lon_remote(self):
        """
        Return proper compressed filename for longitude
        """
        if self.res == '1km':
            return 'IMS1kmLons.24576x24576x1.tar.gz'
        else:
            return f'imslon_{self.res}.bin.gz'

    @property
    def lat_remote(self):
        """
        Return proper compressed filename for latitude
        """
        if self.res == '1km':
            return 'IMS1kmLats.24576x24576x1.tar.gz'
        else:
            return f'imslat_{self.res}.bin.gz'

    @property
    def lon_file(self):
        """
        Return proper filename for longitude
        """
        if self.res == '1km':
            return 'IMS1kmLons.24576x24576x1.double'
        else:
            return f'imslon_{self.res}.bin'

    @property
    def lat_file(self):
        """
        Return proper filename for latitude
        """
        if self.res == '1km':
            return 'IMS1kmLats.24576x24576x1.double'
        else:
            return f'imslat_{self.res}.bin'
