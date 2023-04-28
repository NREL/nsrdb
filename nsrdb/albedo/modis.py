"""
Classes to acquire and load MODIS albedo data. Entrance via ModisDay.

Mike Bannister
2/18/2020
"""
import calendar
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from nsrdb.albedo.ims import get_dt

logger = logging.getLogger(__name__)

MODIS_NODATA = 32767

# Last year of MODIS data. Any dates after this year will use the data for the
# appropriate day from this year.
LAST_YEAR = 2017

# First available MODIS data is 3/3/2000.
FIRST_YEAR = 2000
FIRST_DAY = 63


class ModisError(Exception):
    """
    Custom exception for MODIS processing
    """


class ModisDay:
    """ Load MODIS data for a single day from disk """
    LAT_SIZE = 21600
    LON_SIZE = 43200

    SCALE = 0.001

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
        self.modis_path = modis_path
        if shape is None:
            self._shape = (self.LAT_SIZE, self.LON_SIZE)
        else:
            self._shape = shape

        self._date = date
        self._filename = ModisFileAcquisition.get_file(date, modis_path)
        self.data, self.lon, self.lat = self._load_day()

    def _load_day(self):
        """
        Load albedo values, lon, and lat from HDF

        Returns
        -------
        data : 2D numpy array (float32)
            Array with MODIS dry earth albedo values. CRS appears to be WGS84
        lon : 1D numpy array (float32)
            Array with longitude values for MODIS data
        lat : 1D numpy array (float32)
            Array with latitude values for MODIS data
        """
        import pyhdf
        from pyhdf.SD import SD

        try:
            hdf = SD(self._filename)
        except pyhdf.error.HDF4Error as e:
            raise ModisError(f'Issue loading {self._filename}: {e}') from e

        try:
            logger.info('Loading MODIS data')
            attrs = hdf.select('Albedo_Map_0.3-5.0').attributes()
            data = hdf.select('Albedo_Map_0.3-5.0')[:]
            logger.info('Loading MODIS metadata')
            lat = hdf.select('Latitude')[:]
            lon = hdf.select('Longitude')[:]
            logger.info('Completed loading MODIS data and metadata')
        except pyhdf.error.HDF4Error as e:
            raise ModisError(f'Error loading {self._filename}: {e}. File '
                             'does not have expected datasets and may be '
                             'too old.') from e
        if 'scale_factor' in attrs:
            scale = attrs['scale_factor']
            if scale != self.SCALE:
                msg = (f'Scaling factor of MODIS data is {scale}, but is '
                       f'expected to be {self.SCALE}')
                logger.error(msg)
                raise ModisError(msg)
        else:
            # This should only occur in testing
            logger.warning(f'MODIS data in {self._filename} is missing the '
                           'scale factor attribute.')

        if len(lat) != self._shape[0] or len(lon) != self._shape[1] or \
                data.shape != self._shape:
            msg = (f'Data/metadata shapes are not correct for '
                   f'{self._filename}. Data shape={data.shape}, '
                   f'Lon shape={lon.shape}, Lat shape={lat.shape}. Data '
                   f'shape is expected to be {self._shape}.')
            raise ModisError(msg)

        logger.info(f'MODIS data shape is {data.shape}')
        logger.info(f'Boundaries of MODIS data: '
                    f'{lon.min()} - {lon.max()} long, {lat.min()} - '
                    f'{lat.max()} lat')
        return data.astype(np.float32), lon, lat

    def plot(self):
        """ Plot data as map. Nodata is corrected so colors are sane """
        vals = self.data.copy()
        vals[vals > 1000] = 0
        plt.imshow(vals)
        plt.title(self._filename)
        plt.colorbar()
        plt.show()

    def __repr__(self):
        year = self._date.year
        day = self._date.timetuple().tm_yday
        return f'{self.__class__.__name__}(year={year}, day={day})'


class ModisFileAcquisition:
    """
    Class to acquire MODIS data for requested day. Attempts to get data from
    disk first. If not available the data is downloaded
    exist it is downloaded.
    """
    HTTP_SERVER = 'e4ftl01.cr.usgs.gov'
    HTTP_FOLDER = '/MOTA/MCD43GF.006/'
    FILE_PATTERN = 'MCD43GF_wsa_shortwave_{day}_{year}_V006.hdf'
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
            Desired date for MODIS data.
        path : str
            Location of/for MODIS data on disk.

        Returns
        -------
        filename : str
            Filename with path to MODIS data file
        """
        mfa = cls(date, path)

        # See if the file is on disk
        if os.path.isfile(os.path.join(path, mfa.filename)):
            logger.info(f'{mfa.filename} found on disk at {path}')
        else:
            # Download it
            logger.info(f'{mfa.filename} not found on disk, attempting to '
                        'download')
            mfa._download()
        return os.path.join(mfa.path, mfa.filename)

    def __init__(self, date, path):
        """
        Parameters
        ----------
        date : Datetime object
            Desired date for MODIS data.
        path : str
            Location of/for MODIS data on disk.
        """
        year = date.year
        day = date.timetuple().tm_yday

        msg = 'The available date handling code makes certain assumptions' +\
              ' about whether the first and last available years are leap' +\
              ' years. The new values of FIRST_YEAR or LAST_YEAR violate' +\
              ' those assumptions.'
        assert calendar.isleap(FIRST_YEAR), msg
        assert not calendar.isleap(LAST_YEAR), msg

        # Is date after last available and last day of leap year?
        if year > LAST_YEAR and day == 366:
            self.date = get_dt(LAST_YEAR, 365)
            logger.info('Using day 365 of {} in place of day 366'
                        .format(LAST_YEAR))

        # Is date after last available?
        elif year > LAST_YEAR and day < 366:
            self.date = get_dt(LAST_YEAR, day)

        # Is date before first available day of first available year?
        elif date < get_dt(FIRST_YEAR, FIRST_DAY) and day < FIRST_DAY:
            self.date = get_dt(FIRST_YEAR + 1, day)

        # Is date before first available year?
        elif date < get_dt(FIRST_YEAR, FIRST_DAY) and day >= FIRST_DAY:
            self.date = get_dt(FIRST_YEAR, day)

        # Date falls within available dates
        else:
            self.date = date

        if self.date != date:
            logger.info('MODIS albedo data does not yet exist for '
                        '{}/{}. Using data for {}/{} instead.'
                        ''.format(date.year, date.timetuple().tm_yday,
                                  self.date.year,
                                  self.date.timetuple().tm_yday))

        self.path = path

        # Extract day as day of year (e.g. 1-366), left pad with 0
        self.day = str(self.date.timetuple().tm_yday).zfill(3)
        self.year = str(self.date.year)

        # Example file name: MCD43GF_wsa_shortwave_033_2010.hdf
        self.filename = self.FILE_PATTERN.format(day=self.day, year=self.year)

    @staticmethod
    def _download():
        """
        Download Ver 6 MODIS hdf file from server
        """
        # Example: https://e4ftl01.cr.usgs.gov/MOTA/MCD43GF.006/2012.01.15/
        #                  MCD43GF_wsa_shortwave_015_2012_V006.hdf
        msg = 'Automatic downloading of MODIS data is currently not supported.'
        logger.exception(msg)
        raise NotImplementedError(msg)
