# import ftplib
import os
import sys
# import logging
# import glob
import numpy as np
import re
import matplotlib.pyplot as plt


# TODO - remove below code after testing is finished
nsrdb_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(nsrdb_path)

# from nsrdb.utilities.file_utils import url_download


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

    def __init__(self, date, ims_path, res, meta=None):
        """
        Parameters
        ----------
        date : datetime instance
            Date to grab data for. Time is ignored
        ims_path : str
            Path to IMS data files
        res: str
            Desired IMS resolution [self.RES_1KM or self.RES_4KM]
        meta: tuple
            Lat/lon information as (lat, lon). Loading meta data is slow and
            meta data should be reused if possible.
        """
        # logger.debug(f'Importing {dfile} into ImsDay')
        self.ims_path = ims_path

        # Extract day as day of year (e.g. 1-366), left pad with 0
        self.day = str(date.timetuple().tm_yday).zfill(3)
        self.year = str(date.year)

        # TODO - should this be set here or determined automatically?
        self.res = res
        if res not in [self.RES_1KM, self.RES_4KM]:
            raise ImsError(f'Error loading {self.file_name} meta-data, only 1-'
                           ' and 4-km data is allowed.')

        # Example file name: ims2015009_4km_v1.3.asc
        partial_fname = f'ims{self.year}{self.day}_{self.res}_v1.3.asc'
        self.file_name = os.path.join(ims_path, partial_fname)

        self.data = None  # IMS data, numpy array
        self.lat = None  # Latitude mapping for data
        self.lon = None  # Longitude mapping for data

        self._load_data()

        if meta is None:
            self._load_meta()
        else:
            self.lat = meta[0]
            self.lon = meta[1]

    def _load_data(self):
        """
        Load IMS values from asc file. For format see
        https://nsidc.org/data/g02156 -> "User Guide" tab
        """
        raw = []
        with open(self.file_name, 'r') as dat:
            lines = dat.readlines()
            for line in lines:
                # asc file has text header then rows of [0,1,2,3]
                if re.search('[a-z]', line.strip()) is None:
                    raw.extend([int(l) for l in list(line.strip())])

            # arr = np.flipud(np.array(out).reshape((6144, 6144))) \
            # .flatten().astype(np.int8)

        # IMS data sanity check
        if self.res == self.RES_1KM and len(raw) != 24576**2:
            msg = f'Data length in {self.file_name} is expected to be ' + \
                  f'{24576**2} but is {len(raw)}.'
            raise ImsError(msg)
        if self.res == self.RES_4KM and len(raw) != 6144**2:
            msg = f'Data length in {self.file_name} is expected to be ' + \
                  f'{6144**2} but is {len(raw)}.'
            raise ImsError(msg)

        # Reshape data to square, size dependent on resolution, and flip
        # TODO - do we really need to flipud the data?
        grid = (self.pixels[self.res], self.pixels[self.res])
        ims_data = np.flipud(np.array(raw).reshape(grid))

        self.data = ims_data

    def _load_meta(self):
        """
        Load IMS meta data (lat/lon values) from .double or .bin  file. For
        format see https://nsidc.org/data/g02156 -> "User Guide" tab
        """
        ims_lat_file, ims_lon_file = self._get_lat_lon_files()

        # The 1km and 4km data are stored in different formats
        if self.res == self.RES_1KM:
            count = self.pixels[self.res]**2
            with open(ims_lat_file, 'rb') as f:
                lat = np.fromfile(ims_lat_file, dtype='<d', count=count)\
                    .astype(np.float32)
            with open(ims_lon_file, 'rb') as f:
                lon = np.fromfile(f, dtype='<d', count=count)\
                    .astype(np.float32)
        else:
            # 4km
            lat = np.fromfile(ims_lat_file, dtype='<f4')
            lon = np.fromfile(ims_lon_file, dtype='<f4')

        # Longitude might be stored as 0-360, fix
        lon = np.where(lon > 180, lon - 360, lon)

        # Meta data sanity checks
        if self.res == self.RES_1KM and lon.shape != (603979776,):
            msg = f'Shape of {ims_lon_file} is expected to be (603979776,)' + \
                   f' but is {lon.shape}.'
            raise ImsError(msg)
        if self.res == self.RES_1KM and lat.shape != (603979776,):
            msg = f'Shape of {ims_lat_file} is expected to be (603979776,)' + \
                   f' but is {lat.shape}.'
            raise ImsError(msg)
        if self.res == self.RES_4KM and lon.shape != (37748736,):
            msg = f'Shape of {ims_lon_file} is expected to be (37748736,)' + \
                   f' but is {lon.shape}.'
            raise ImsError(msg)
        if self.res == self.RES_4KM and lat.shape != (37748736,):
            msg = f'Shape of {ims_lat_file} is expected to be (37748736,)' + \
                   f' but is {lat.shape}.'
            raise ImsError(msg)

        self.lon = lon
        self.lat = lat

    def _get_lat_lon_files(self):
        if self.res == self.RES_1KM:
            lat = 'IMS1kmLats.24576x24576x1.double'
            lon = 'IMS1kmLons.24576x24576x1.double'
        else:
            # 4km
            lat = 'imslat_4km.bin'
            lon = 'imslon_4km.bin'

        lat_file = os.path.join(self.ims_path, lat)
        lon_file = os.path.join(self.ims_path, lon)
        return lat_file, lon_file

    def plot(self):
        """ Plot values as map. """
        plt.imshow(self.data)
        plt.title(self.file_name)
        plt.colorbar()
        plt.show()

    def __repr__(self):
        return f'{self.__class__.__name__}(year={self.year}, day={self.day})'


# TODO - Ims needs to be updated to handle arbitrary dates
class Ims:
    """
    Class to load and represent IMS snow data. If data does not exist it is
    downloaded.
    """
    def __init__(self, year, data_dir):
        if year not in data_dir:
            self._download(year)
        self._load_year(year)

    def _load_year(self):
        pass

    def _download(self, year):
        pass

    @staticmethod
    def download_all_years():
        pass


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
