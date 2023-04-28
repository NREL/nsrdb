# -*- coding: utf-8 -*-
"""A framework for handling global forecasting system (GFS) source data as a
real-time replacement for GFS."""
import logging
import numpy as np
import os
import pandas as pd
import re
import time

from nsrdb.data_model.base_handler import AncillaryVarHandler, BaseDerivedVar
from nsrdb.file_handlers.file_system import NSRDBFileSystem as NFS

logger = logging.getLogger(__name__)

DATADIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.join(os.path.dirname(DATADIR), 'data')


class GfsFiles:
    """
    Class to handle Gfs file selection
    """
    def __init__(self, source_dir, time_index):
        """
        Parameters
        ----------
        source_dir : str
            Directory containing GFS files
        time_index : pandas.DatetimeIndex
            DatetimeIndex for day of interest
        """
        self._time_index = time_index
        self._files = self._get_files(source_dir)

    @property
    def time_index(self):
        """
        Time index for the date of interest that files are needed for

        Returns
        -------
        pandas.DatetimeIndex
        """
        return self._time_index

    @property
    def date_stamp(self):
        """
        Date to get GFS files for

        Returns
        -------
        str
            Date in GFS file format of YYYY_MMDD
        """
        return self._make_date_stamp(self.time_index[0])

    @property
    def files(self):
        """
        GFS files that match time_index

        Returns
        -------
        list
        """
        return self._files

    @staticmethod
    def _make_date_stamp(date):
        """
        Get the GFS datestamp corresponding to the specified datetime date

        Parameters
        ----------
        date : pandas.Timestamp
            Timestamp for date of interest

        Returns
        -------
        date : str
            Date stamp that should be in the GFS file, format is YYYY_MMDD
        """
        date = '{y}_{m:02d}{d:02d}'.format(y=date.year,
                                           m=date.month,
                                           d=date.day)

        return date

    @staticmethod
    def _get_fcst_offset(file_name):
        """
        Get offset from date beginning based on model start / forecast time

        Parameters
        ----------
        file_name : str
            File name

        Returns
        -------
        offset : pd.Timedelta
            Time delta from date 00:00:00
        """
        # pylint: disable=anomalous-backslash-in-string
        pattern = re.compile('\d{02}z')  # noqa: W605
        matcher = pattern.search(file_name)

        offset = pd.Timedelta(matcher.group().replace('z', 'h'))

        return offset

    @staticmethod
    def _get_model_offset(file_name):
        """
        Get offset from date beginning based on model time

        Parameters
        ----------
        file_name : str
            File name

        Returns
        -------
        offset : pd.Timedelta
            Time delta from date 00:00:00
        """
        # pylint: disable=anomalous-backslash-in-string
        pattern = re.compile('\d{02}hr')  # noqa: W605
        matcher = pattern.search(file_name)

        offset = pd.Timedelta(matcher.group())

        return offset

    @classmethod
    def _get_file_time(cls, file_name, start_time):
        """
        Get file time stamp and model offset

        Parameters
        ----------
        file_name : str
            File name
        start_time : pandas.Timestamp
            Start time to offset GFS files from

        Returns
        -------
        timestamp : pandas.Timestamp
            Timestamp of file
        offset : pd.TimeDelta
            Offset from forecast start time (model offset)
        """
        model_offset = cls._get_fcst_offset(file_name)
        fcst_offset = cls._get_model_offset(file_name)
        timestamp = start_time + model_offset + fcst_offset

        return timestamp, fcst_offset

    @classmethod
    def _search_files(cls, source_dir, start_time):
        """
        Search source dir for files that match given date/start_time

        Parameters
        ----------
        source_dir : str
            Directory containing GFS files
        start_time : pd.Timestamp
            Timestamp for startime to get GFS files for, this should be the
            desired date to find GFS files for with a time of 00:00:00

        Returns
        -------
        files : pandas.DataFrame
            DataFrame of all GFS files in source_dir for the given start_time
            with the corresponding timestamp of the file and offset from the
            GFS forecast start-time. Offset is used to pull the shortest
            forecast time for each timestep in time_index
        """
        files = []
        date_stamp = cls._make_date_stamp(start_time)
        for f in NFS(source_dir).ls():
            if date_stamp in f:
                fpath = os.path.join(source_dir, f)
                timestamp, offset = cls._get_file_time(f, start_time)
                files.append(pd.Series({'timestamp': timestamp,
                                        'offset': offset}, name=fpath))

        if files:
            files = pd.concat(files, axis=1).T
        return files

    def _get_files(self, source_dir):
        """
        Get GFS files from source_dir that match the timesteps in time_index

        Parameters
        ----------
        source_dir : str
            Directory containing GFS files

        Returns
        -------
        gfs_files : list
            List of file paths for each timestep in time_index. Files with the
            shortest forecast time ('offset') are chosen for each timestep
        """
        files = self._search_files(source_dir, self.time_index[0])

        start_time = self.time_index[0] - pd.Timedelta('1D')
        files2 = self._search_files(source_dir, start_time)
        both_df_check = (isinstance(files, pd.DataFrame)
                         and isinstance(files2, pd.DataFrame))
        if both_df_check:
            files = pd.concat([files, files2])
        elif isinstance(files2, pd.DataFrame):
            files = files2

        if isinstance(files, list):
            msg = ("Could not find any GFS files for {} in {}"
                   .format(self.time_index[0], source_dir))
            logger.error(msg)
            raise FileNotFoundError(msg)

        files = files.groupby('timestamp')

        gfs_files = []
        missing = []
        for timestamp in self.time_index:
            try:
                grp = files.get_group(timestamp)
                gfs_files.append(grp.sort_values('offset').index[0])
            except KeyError:
                missing.append(timestamp)

        if missing:
            m = ('Could not find the required GFS file with date stamp "{}" '
                 'in directory {}, the following timestamps were missing:\n{}'
                 .format(self.date_stamp, source_dir, missing))
            logger.error(m)
            raise FileNotFoundError(m)

        return gfs_files

    @classmethod
    def get(cls, source_dir, time_index):
        """
        Get GFS files from source_dir that match the desired time_index

        Parameters
        ----------
        source_dir : str
            Directory containing GFS files
        time_index : pandas.DatetimeIndex
            DatetimeIndex for day of interest

        Returns
        -------
        list
            List of file paths for each timestep in time_index. Files with the
            shortest forecast time ('offset') are chosen for each timestep
        """
        return cls(source_dir, time_index).files


class GfsVarSingle:
    """Handler single-file/single-timestep GFS data extraction."""

    def __init__(self, fpath):
        """
        Parameters
        ----------
        fpath : str
            Path to a single timestep GFS file
        """
        # pylint: disable=no-member
        self._fs = NFS(fpath)
        self._dataset = self._fs.open()
        self._dsets = None
        self._units = None
        self._timestep = None

    def __enter__(self):
        """
        Enter method to allow use of with
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        Closes dataset on exiting with
        """
        self._fs._close()

        if type is not None:
            raise

    def __getitem__(self, name):
        """
        Exract variable 'name' from dataset.

        Parameters
        ----------
        name : str
            Variable key

        Returns
        -------
        out : ndarray
            Variables values
        """
        if name == 'wind_speed':
            out = self.wind_speed
        elif name == 'wind_direction':
            out = self.wind_direction
        else:
            out = self._dataset.variables[name][...]

        if isinstance(out, np.ma.MaskedArray):
            out = out.data

        return out

    @property
    def datasets(self):
        """
        Get the list of dataset names in the GFS files.

        Returns
        -------
        list
        """
        if self._dsets is None:
            self._dsets = sorted(self._dataset.variables)

        return self._dsets

    @property
    def units(self):
        """
        Get a dict lookup of GFS datasets and source units

        Returns
        -------
        dict
        """
        if self._units is None:
            self._units = {}
            for k, v in self._dataset.variables.items():
                if 'units' in v.ncattrs():
                    self._units[k] = v.getncattr('units')

        return self._units

    @property
    def timestep(self):
        """
        File timestep in UTC

        Returns
        -------
        TimeStamp
        """
        if self._timestep is None:
            self._timestep = self['time'][0]
            self._timestep = time.strftime('%Y-%m-%d %H:%M:%S',
                                           time.gmtime(self._timestep))
            self._timestep = pd.to_datetime(self._timestep)

        return self._timestep

    @property
    def wind_speed(self):
        """
        10m Wind speed

        Returns
        -------
        ndarray
        """
        u_vector = self['UGRD_10maboveground'][...]
        v_vector = self['VGRD_10maboveground'][...]

        return np.sqrt(u_vector**2 + v_vector**2)

    @property
    def wind_direction(self):
        """
        10m Wind Direction

        Returns
        -------
        ndarray
        """
        u_vector = self['UGRD_10maboveground'][...]
        v_vector = self['VGRD_10maboveground'][...]

        return np.degrees(np.arctan2(u_vector, v_vector)) + 180


class GfsVar(AncillaryVarHandler):
    """Framework for GFS source data extraction."""

    def __init__(self, name, var_meta, date, **kwargs):
        """
        Parameters
        ----------
        name : str
            NSRDB var name.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        date : datetime.date
            Single day to extract data for.
        """

        self._grid = None
        self._files = None
        super().__init__(name, var_meta=var_meta, date=date, **kwargs)

    @property
    def time_index(self):
        """
        Get the albedo native time index.

        Returns
        -------
        alb_ti : pd.DatetimeIndex
            Pandas datetime index for the current day at the albedo
            resolution (1-month).
        """
        return self._get_time_index(self._date, freq='3h')

    @property
    def date_stamp(self):
        """
        Get the GFS datestamp corresponding to the specified datetime date

        Returns
        -------
        date : str
            Date stamp that should be in the GFS file, format is YYYY_MMDD
        """
        date = '{y}_{m:02d}{d:02d}'.format(y=self._date.year,
                                           m=self._date.month,
                                           d=self._date.day)

        return date

    @property
    def file(self):
        """
        Overrides the base handler file attribute

        Returns
        -------
        list
        """
        return self.files

    @property
    def files(self):
        """
        Get the GFS file paths for the target NSRDB day.

        Returns
        -------
        files: list
            List of GFS file paths needed to fill given NSRDB day.
        """
        if self._files is None:
            self._files = GfsFiles.get(self.source_dir, self.time_index)

        return self._files

    @property
    def shape(self):
        """
        Output dataset shape, (time, sites)

        Returns
        -------
        tuple
        """
        return (len(self.time_index), len(self.grid))

    @property
    def grid(self):
        """
        Return the GFS source coordinates with elevation.

        Returns
        -------
        self._GFS_grid : pd.DataFrame
            GFS source coordinates with elevation
        """
        if self._grid is None:
            with GfsVarSingle(self.files[0]) as f:
                lon2d, lat2d = np.meshgrid(f['longitude'][:], f['latitude'][:])
                elev = f['HGT_surface'][:]

            self._grid = pd.DataFrame({'longitude': lon2d.ravel(),
                                       'latitude': lat2d.ravel(),
                                       'elevation': elev.ravel()})

        return self._grid

    @property
    def source_data(self):
        """
        Get single day data from the GFS source file.

        Returns
        -------
        data : np.ndarray
            Flattened GFS time-series data. Note that the data originates as a
            2D spatially gridded numpy array with shape (lat x lon).
        """
        data = np.zeros(self.shape, dtype=np.float32)
        for i, fp in enumerate(self.files):
            with GfsVarSingle(fp) as f:
                if self.name in ('wind_speed', 'wind_direction'):
                    if self.name == 'wind_speed':
                        data[i] = f.wind_speed.ravel()
                    else:
                        data[i] = f.wind_direction.ravel()

                else:
                    data[i] = f[self.dset_name].ravel()

        return data

    def pre_flight(self):
        """
        Perform pre-flight checks - source file check.

        Returns
        -------
        missing : str
            Look for the source file and return the string if not found.
            If nothing is missing, return an empty string.
        """
        time_index = []
        for file in self.files:
            with GfsVarSingle(file) as f:
                time_index.append(f.timestep)

        time_index = pd.to_datetime(time_index)
        missing = ~self.time_index.isin(time_index)
        if np.any(missing):
            missing = ("The following GFS time-steps are missing: {}"
                       .format(self.time_index[missing]))
        else:
            missing = ''

        return missing


class GfsDewPoint(BaseDerivedVar):
    """Class to derive the dew point from other GFS vars."""

    DEPENDENCIES = ('air_temperature', 'relative_humidity')

    @staticmethod
    def derive(air_temperature, relative_humidity):
        """Derive the dew point from ancillary vars.

        Parameters
        ----------
        air_temperature : np.ndarray
            Temperature in Celsius
        relative_humidity : np.array
            Relative Humidity

        Returns
        -------
        dp : np.ndarray
            Dew point in Celsius.
        """
        logger.info('Deriving Dew Point from temperature, '
                    'and relative humidity')

        # ensure that Temperature is in C (scale from Kelvin if not)
        convert_t = False
        if np.max(air_temperature) > 100:
            convert_t = True
            air_temperature -= 273.15

        arg1 = np.log(relative_humidity / 100.0)
        arg2 = (17.625 * air_temperature) / (243.04 + air_temperature)
        dp = (243.04 * (arg1 + arg2) / (17.625 - arg1 - arg2))

        # ensure that temeprature is reconverted to Kelvin
        if convert_t:
            air_temperature += 273.15

        return dp
