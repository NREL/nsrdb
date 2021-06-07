# -*- coding: utf-8 -*-
"""A framework for handling global forecasting system (GFS) source data as a
real-time replacement for GFS."""
import logging
import netCDF4 as nc
import numpy as np
import os
import pandas as pd

from cloud_fs import FileSystem as FS

from nsrdb import DATADIR
from nsrdb.data_model.base_handler import AncillaryVarHandler

logger = logging.getLogger(__name__)


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
        """Get the albedo native time index.

        Returns
        -------
        alb_ti : pd.DatetimeIndex
            Pandas datetime index for the current day at the albedo
            resolution (1-month).
        """
        return self._get_time_index(self._date, freq='3h')

    @property
    def date_stamp(self):
        """Get the GFS datestamp corresponding to the specified datetime date

        Returns
        -------
        date : str
            Date stamp that should be in the GFS file, format is YYYY_MMDD
        """

        y = str(self._date.year)
        m = str(self._date.month).zfill(2)
        d = str(self._date.day).zfill(2)
        date = '{y}_{m}{d}'.format(y=y, m=m, d=d)

        return date

    @property
    def file(self):
        """Get the GFSfile path for the target NSRDB date.

        Returns
        -------
        fpath : str
            GFS file path.
        """
        fpath = None
        flist = FS(self.source_dir).ls()
        for f in flist:
            if self.date_stamp in f:
                fpath = os.path.join(self.source_dir, f)
                break

        if fpath is None:
            m = ('Could not find GFS file with date stamp "{}" '
                 'in directory: {}'
                 .format(self.date_stamp, self.source_dir))
            logger.error(m)
            raise FileNotFoundError(m)

        return fpath

    @property
    def source_dsets(self):
        """Get the list of dataset names in the GFS files.

        Returns
        -------
        list
        """
        # pylint: disable=no-member
        with FS(self.file).open() as fp:
            with nc.Dataset(fp, mode='r') as f:
                dsets = sorted(f.variables)

        return dsets

    def source_units(self):
        """Get a dict lookup of GFS datasets and source units

        Returns
        -------
        dict
        """
        units = {}
        # pylint: disable=no-member
        with FS(self.file).open() as fp:
            with nc.Dataset(fp, mode='r') as f:
                for k, v in f.variables.items():
                    if 'units' in v.ncattrs():
                        units[k] = v.getncattr('units')

        return units

    def pre_flight(self):
        """Perform pre-flight checks - source file check.

        Returns
        -------
        missing : str
            Look for the source file and return the string if not found.
            If nothing is missing, return an empty string.
        """
        missing = ''
        if not FS(self.file).isfile():
            missing = self.file

        return missing

    @property
    def source_data(self):
        """Get single day data from the GFS source file.

        Returns
        -------
        data : np.ndarray
            Flattened GFS data. Note that the data originates as a 2D
            spatially gridded numpy array with shape (lat x lon).
        """
        # open NetCDF file
        with FS(self.file).open() as fp:
            # pylint: disable=no-member
            with nc.Dataset(fp, 'r') as f:
                # depending on variable, might need extra logic
                if self.name in ['wind_speed', 'wind_direction']:
                    u_vector = f['UGRD_10maboveground'][:]
                    v_vector = f['VGRD_10maboveground'][:]
                    if self.name == 'wind_speed':
                        data = np.sqrt(u_vector**2 + v_vector**2)
                    else:
                        data = np.degrees(
                            np.arctan2(u_vector, v_vector)) + 180
                else:
                    data = f[self.dset_name][:]

        return data

    @property
    def grid(self):
        """Return the GFS source coordinates with elevation.

        Returns
        -------
        self._GFS_grid : pd.DataFrame
            GFS source coordinates with elevation
        """
        if self._grid is None:
            with FS(self.file).open() as fp:
                # pylint: disable=no-member
                with nc.Dataset(fp, 'r') as f:
                    lon2d, lat2d = np.meshgrid(f['lon'][:], f['lat'][:])

            self._grid = pd.DataFrame({'longitude': lon2d.ravel(),
                                       'latitude': lat2d.ravel()})

            # add elevation to coordinate set
            elev = pd.read_csv(self.GFS_ELEV)
            self._grid = self._grid.merge(elev,
                                          on=['latitude', 'longitude'],
                                          how='left')

        return self._grid
