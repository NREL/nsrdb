import ftplib
import os
import sys
import logging
import glob
import pyhdf
from pyhdf.SD import SD
import matplotlib.pyplot as plt


# TODO - remove below code after testing is finished
nsrdb_path = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(nsrdb_path)

#from nsrdb.utilities.file_utils import url_download

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NODATA = 32767


class ModisError(Exception):
    pass


# TODO - This whole class needs to be updated to handle an arbitrary date
# range
class ModisYear:
    """
    Class to load and represent MODIS snow-free albedo data. If data does not
    exist it is downloaded.
    """
    FTP_SERVER = 'rsftp.eeos.umb.edu'
    FTP_FOLDER = '/data02/Gapfilled/{}/'
    DATA_PATTERN = 'MCD43GF_wsa_shortwave_'
    # Example file name: MCD43GF_wsa_shortwave_033_2010.hdf

    def __init__(self, year, data_path, check_file_count=False):
        """
        Load MODIS data for year from disk. If it doesn't exist, download
        via ftp.

        Parameters
        ----------
        year : int | str
            Year to download modis data
        data_path : str
            Path for MODIS data files
        check_file_count : bool
            If False, only check if any files for year exist in data_path. If
            True, verify that all files for year on server are in data_path.
        """
        self.year = year
        self.data_path = data_path
        self.days = None  # list of ModisDay instances

        data_files = glob.glob(os.path.join(data_path,
                                            self.DATA_PATTERN +
                                            f'*_{year}.hdf'))
        if len(data_files) > 0:
            # TODO - check_file_count
            self._load_year(data_files)
        else:
            self._download()
            data_files = glob.glob(os.path.join(data_path,
                                                self.DATA_PATTERN +
                                                f'*_{year}.hdf'))
            self._load_year(data_files)

    def _load_year(self, data_files):
        logger.debug(f'files to load: {data_files}')
        self.days = [ModisDay(dfile) for dfile in data_files]
        self.days.sort(key=lambda x: x.day)

    def _download(self):
        """
        Download MODIS files for a year via FTP
        """
        # TODO - check if any files are missing on server
        ftp = ftplib.FTP(self.FTP_SERVER)
        ftp.login()
        year_path = self.FTP_FOLDER.format(self.year)
        ftp.cwd(year_path)

        # Grab list of data files for desired year
        all_files = []
        ftp.retrlines('LIST', all_files.append)
        data_files = [x.split(' ')[-1] for x in all_files if self.DATA_PATTERN
                      in x]

        # Download data
        failed = []
        # TODO - remove slicing
        #for dfile in data_files[:3]:
        for dfile in data_files:
            url = f'ftp://{self.FTP_SERVER}{year_path}{dfile}'
            lfile = os.path.join(self.data_path, dfile)
            logger.info(f'Downloading {url}')
            fail = url_download(url, lfile)
            if fail:
                failed.append(url)
        return failed

    @staticmethod
    def download_all_years():
        pass

    def __len__(self):
        return len(self.days)

    def __getitem__(self, position):
        return self.days[position]


class ModisDay:
    """ Load MODIS data for a single day from disk """
    def __init__(self, date, modis_path):
        """
        Parameters
        ----------
        date : datetime instance
            Date to grab data for. Time is ignored
        modis_path : str
            Path to MODIS data files
        """
        # logger.debug(f'Importing {dfile} into ModisDay')
        self.modis_path = modis_path

        # Extract day as day of year (e.g. 1-366), left pad with 0
        self.day = str(date.timetuple().tm_yday).zfill(3)
        self.year = str(date.year)

        # TODO - MODIS is avaible in 8 day increments. Grab the nearest day
        # Example file name: MCD43GF_wsa_shortwave_033_2010.hdf
        partial_fname = f'MCD43GF_wsa_shortwave_{self.day}_{self.year}.hdf'
        self.file_name = os.path.join(modis_path, partial_fname)

        self._load_day()

    def _load_day(self):
        """ Load albedo values, lon, and lat from HDF """
        try:
            hdf = SD(self.file_name)
        except pyhdf.error.HDF4Error as e:
            raise ModisError(f'Issue loading {self.file_name}: {e}')

        self.lat = hdf.select('Latitude')[:]
        self.lon = hdf.select('Longitude')[:]
        self.data = hdf.select('Albedo_Map_0.3-5.0')[:]

        if len(self.lat) != 21600 or len(self.lon) != 43200 or \
                self.data.shape != (21600, 43200):
            msg = f'Shape of {self.file_name} is expected to be (21600, ' + \
                   f'43200) but is {self.data.shape}. Lat/lon may be off.'
            raise ModisError(msg)

    def plot(self):
        """ Plot data as map. Nodata is corrected so colors are sane """
        vals = self.data.copy()
        vals[vals > 1000] = 0
        plt.imshow(vals)
        plt.title(self.file_name)
        plt.colorbar()
        plt.show()

    def __repr__(self):
        return f'{self.__class__.__name__}(year={self.year}, day={self.day})'


if __name__ == '__main__':
    #ModisYear(2015, 'scratch')

    from datetime import datetime as dt
    from datetime import timedelta
    #year = ModisYear(2012, 'scratch')
    dates = [(2013, 1)] # , (2015, 9), (2013, 145)]
    for y, d in dates:
        date = dt(y, 1, 1) + timedelta(d - 1)
        print(date)
        m = ModisDay(date, 'scratch')
        m.plot()



        # d.plot()
    # print(m._download())
