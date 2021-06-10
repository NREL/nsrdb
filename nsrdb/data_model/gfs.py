# -*- coding: utf-8 -*-
"""A framework for handling global forecasting system (GFS) source data as a
real-time replacement for GFS."""
import logging
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
import time

from cloud_fs import FileSystem as FS

from nsrdb.data_model.base_handler import AncillaryVarHandler

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
        self._fpath = FS(fpath).open()
        # pylint: disable=no-member
        self._dataset = nc.Dataset(self._fpath, mode='r')
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
        self.close()

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
            out = out.out

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

    def close(self):
        """
        Close netCDF Dataset
        """
        if self._dataset.isopen():
            self._dataset.close()

        self._fpath.close()


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
        files = self._get_gfs_files(self.source_dir, self.date_stamp)

        if len(files) < len(self.time_index):

            m = ('Could not find required GFS file with date stamp "{}" '
                 'in directory: {}'
                 .format(self.date_stamp, self.source_dir))
            logger.error(m)
            raise FileNotFoundError(m)

        return files

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

            self._grid = pd.DataFrame({'longitude': lon2d.ravel(),
                                       'latitude': lat2d.ravel()})

            # add elevation to coordinate set
            elev = pd.read_csv(self.GFS_ELEV)
            self._grid = self._grid.merge(elev,
                                          on=['latitude', 'longitude'],
                                          how='left')

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
        for i, file in enumerate(self.files):
            with GfsVarSingle(file) as f:
                data[i] = f[self.name].ravel()

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
        """
        date_files = {'00z': [], '06z': [], '12z': [], '18z': []}
        flist = FS(source_dir).ls()
        for f in sorted(flist):
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
            pass

        return missing
