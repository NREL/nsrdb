# -*- coding: utf-8 -*-
"""A framework for handling global forecasting system (GFS) source data as a
real-time replacement for GFS."""
import logging
import numpy as np
import os
import pandas as pd
import time

from nsrdb.data_model.base_handler import AncillaryVarHandler
from nsrdb.file_handlers.filesystem import NSRDBFileSystem as NFS

logger = logging.getLogger(__name__)

DATADIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.join(os.path.dirname(DATADIR), 'data')


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
    GFS_ELEV = os.path.join(DATADIR, 'gfs_grid_srtm_500m_stats.csv')

    def __init__(self, name, var_meta, date, source_dir=None):
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
        source_dir : str | None
            Optional data source directory. Will overwrite the source directory
            from the var_meta input.
        """

        self._grid = None
        self._files = None
        super().__init__(name, var_meta=var_meta, date=date,
                         source_dir=source_dir)

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
    def files(self):
        """
        Get the GFS file paths for the target NSRDB day.

        Returns
        -------
        files: list
            List of GFS file paths needed to fill given NSRDB day.
        """
        if self._files is None:
            files = self._get_gfs_files(self.source_dir, self.date_stamp)

            self._files = self._check_files(files)

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

    @staticmethod
    def _get_gfs_files(source_dir, date_stamp):
        """
        Search source_dir for all GFS files for the given date stamp.
        Find files with the shortest forecast lead time.

        Parameters
        ----------
        source_dir : str
            Source directory containing GFS files
        date_stamp : str
            Date stamp that should be in the GFS file, format is YYYY_MMDD

        Returns
        -------
        files : list
            List of GFS files to use for given NSRDB date stamp
        """
        date_files = {'00z': [], '06z': [], '12z': [], '18z': []}
        for f in NFS(source_dir).ls():
            if date_stamp in f:
                fpath = os.path.join(source_dir, f)
                for k, v in date_files.items():
                    if k in f:
                        v.append(fpath)

        files = []
        i = 2  # if all models are available date 2 files from each (00 and 03)
        for j, m in zip(range(1, 5), sorted(date_files)[::-1]):
            files.extend(date_files[m][:i])
            i = 2 + (j * 2 - len(files))

        return sorted(files)

    def _check_files(self, files):
        """
        Check to make sure there are enough available files to model the
        desired NSRDB date stamp. If needed pull files from the prior days 18z
        model run

        Parameters
        ----------
        files : list
            List of GFS files to use for given NSRDB date stamp

        Returns
        -------
        files : list
            List of GFS files to use for given NSRDB date stamp
        """
        if len(files) < len(self.time_index):
            date = self.time_index[0] - pd.Timedelta('1d')
            date = '{y}_{m:02d}{d:02d}'.format(y=date.year,
                                               m=date.month,
                                               d=date.day)

            files = []
            hr_list = ["{:03d}hr".format(h) for h in range(6, 30, 3)]
            for f in NFS(self.source_dir).ls():
                hr_check = any(hr in f for hr in hr_list)
                if date in f and '18z' in f and hr_check:
                    files.append(os.path.join(self.source_dir, f))

        if len(files) < len(self.time_index):
            m = ('Could not find required GFS file with date stamp "{}" '
                 'in directory: {}'
                 .format(self.date_stamp, self.source_dir))
            logger.error(m)
            raise FileNotFoundError(m)

        return files

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
