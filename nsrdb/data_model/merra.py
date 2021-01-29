# -*- coding: utf-8 -*-
"""A framework for handling MERRA2 source data."""

import numpy as np
import pandas as pd
import os
import netCDF4
import logging

from nsrdb import DATADIR
from nsrdb.data_model.base_handler import AncillaryVarHandler


logger = logging.getLogger(__name__)


class MerraVar(AncillaryVarHandler):
    """Framework for MERRA source data extraction."""

    # default MERRA paths.
    MERRA_ELEV = os.path.join(DATADIR, 'merra_grid_srtm_500m_stats.csv')

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

        self._merra_grid = None
        super().__init__(name, var_meta=var_meta, date=date,
                         source_dir=source_dir)

    @property
    def date_stamp(self):
        """Get the MERRA datestamp corresponding to the specified datetime date

        Returns
        -------
        date : str
            Date stamp that should be in the MERRA file, format is YYYYMMDD
        """

        y = str(self._date.year)
        m = str(self._date.month).zfill(2)
        d = str(self._date.day).zfill(2)
        date = '{y}{m}{d}'.format(y=y, m=m, d=d)
        return date

    @property
    def file(self):
        """Get the MERRA file path for the target NSRDB variable name.

        Returns
        -------
        fmerra : str
            MERRA file path containing the target NSRDB variable.
        """

        path = os.path.join(self.source_dir, self.dset)
        flist = os.listdir(path)
        fmerra = None

        for f in flist:
            if self.date_stamp in f:
                fmerra = os.path.join(path, f)
                break

        if fmerra is None:
            emsg = ('Could not find MERRA source file for dataset "{}" with '
                    'date stamp {} in dir: {}'
                    .format(self.name, self.date_stamp, path))
            logger.error(emsg)
            raise FileNotFoundError(emsg)

        return fmerra

    def pre_flight(self):
        """Perform pre-flight checks - source file check.

        Returns
        -------
        missing : str
            Look for the source file and return the string if not found.
            If nothing is missing, return an empty string.
        """

        missing = ''
        if not os.path.isfile(self.file):
            missing = self.file
        return missing

    @property
    def merra_name(self):
        """Get the MERRA variable name from the NSRDB variable name.

        Returns
        -------
        merra_name : str
            MERRA var name.
        """
        return str(self.var_meta.loc[self.mask, 'merra_name'].values[0])

    @property
    def time_index(self):
        """Get the MERRA native time index.

        Returns
        -------
        MERRA_time_index: pd.DatetimeIndex
            Pandas datetime index for the current day at the MERRA2 resolution
            (1-hour).
        """
        return self._get_time_index(self._date, freq='1h')

    @staticmethod
    def _format_2d(data):
        """Format MERRA data as a flat 2D array: (time X sites).

        MERRA data is sourced as a 3D array: (time X sitex X sitey).

        Parameters
        ----------
        data : np.ndarray
            3D numpy array of MERRA data. 1st dim is time, 2nd and 3rd are
            both spatial.

        Returns
        -------
        flat_data : np.ndarray
            2D numpy array of flattened MERRA data. 1st dim is time, 2nd is
            spatial.
        """
        flat_data = np.zeros(shape=(data.shape[0],
                                    data.shape[1] * data.shape[2]),
                             dtype=np.float32)
        for i in range(data.shape[0]):
            flat_data[i, :] = data[i, :, :].ravel()
        return flat_data

    @property
    def source_data(self):
        """Get single variable data from the MERRA source file.

        Returns
        -------
        data : np.ndarray
            2D numpy array (time X space) of MERRA data for the specified var.
        """

        # cloud variables when satellite data is not available
        cld_vars = ['cloud_type', 'cld_press_acha', 'cloud_press_acha',
                    'cld_opd_dcomp', 'cld_reff_dcomp']

        # open NetCDF file
        with netCDF4.Dataset(self.file, 'r') as f:

            # depending on variable, might need extra logic
            if self.name in ['wind_speed', 'wind_direction']:
                u_vector = f['U2M'][:]
                v_vector = f['V2M'][:]
                if self.name == 'wind_speed':
                    data = np.sqrt(u_vector**2 + v_vector**2)
                else:
                    data = np.degrees(
                        np.arctan2(u_vector, v_vector)) + 180

            elif self.merra_name == 'TOTSCATAU':
                # Single scatter albedo is total scatter / aod
                data = f[self.merra_name][:] / f['TOTEXTTAU'][:]

            elif self.name in cld_vars:
                # Special handling of cloud properties when
                # satellite data is not available.
                opd = f['TAUTOT'][:]

                if self.name == 'cld_opd_dcomp':
                    data = opd
                else:
                    opd_hi = f['TAUHGH'][:]
                    ctype = np.zeros_like(opd)
                    ctype = np.where((opd > 0) & (opd_hi <= 0), 3, ctype)
                    ctype = np.where((opd > 0) & (opd_hi > 0), 6, ctype)

                    if self.name == 'cloud_type':
                        data = ctype
                    elif self.name == 'cld_reff_dcomp':
                        data = np.zeros_like(opd)
                        data = np.where(ctype == 3, 8.0, data)
                        data = np.where(ctype == 6, 20.0, data)
                    elif self.name in ['cld_press_acha', 'cloud_press_acha']:
                        data = np.zeros_like(opd)
                        data = np.where(ctype == 3, 800.0, data)
                        data = np.where(ctype == 6, 250.0, data)

            else:
                data = f[self.merra_name][:]

        # make the data a flat 2d array
        data = self._format_2d(data)

        return data

    @property
    def grid(self):
        """Return the MERRA source coordinates with elevation.

        It seems that all MERRA files DO NOT have the same grid.

        Returns
        -------
        self._merra_grid : pd.DataFrame
            MERRA source coordinates with elevation
        """

        if self._merra_grid is None:

            with netCDF4.Dataset(self.file, 'r') as nc:
                lon2d, lat2d = np.meshgrid(nc['lon'][:], nc['lat'][:])

            self._merra_grid = pd.DataFrame({'longitude': lon2d.ravel(),
                                             'latitude': lat2d.ravel()})

            # merra grid has some bad values around 0 lat/lon
            # quick fix is to set to zero
            self._merra_grid.loc[(self._merra_grid['latitude'] > -0.1)
                                 & (self._merra_grid['latitude'] < 0.1),
                                 'latitude'] = 0
            self._merra_grid.loc[(self._merra_grid['longitude'] > -0.1)
                                 & (self._merra_grid['longitude'] < 0.1),
                                 'longitude'] = 0

            # add elevation to coordinate set
            merra_elev = pd.read_csv(self.MERRA_ELEV)
            self._merra_grid = self._merra_grid.merge(merra_elev,
                                                      on=['latitude',
                                                          'longitude'],
                                                      how='left')

            # change column name from merra default
            if 'mean_elevation' in self._merra_grid.columns.values:
                self._merra_grid = self._merra_grid.rename(
                    {'mean_elevation': 'elevation'}, axis='columns')

        return self._merra_grid


class RelativeHumidity:
    """Class to derive the relative humidity from other MERRA2 vars."""

    @staticmethod
    def derive(t, h, p):
        """Derive the relative humidity from ancillary vars.

        Parameters
        ----------
        t : np.ndarray
            Temperature in Celsius
        h : np.ndarray
            Specific humidity in kg/kg
        p : np.ndarray
            Pressure in Pa

        Returns
        -------
        rh : np.ndarray
            Relative humidity in %.
        """
        logger.info('Deriving Relative Humidity from temperature, '
                    'humidity, and pressure')

        # ensure that Pressure is in Pa (scale from mbar if not)
        convert_p = False
        if np.max(p) < 10000:
            convert_p = True
            p *= 100
        # ensure that Temperature is in C (scale from Kelvin if not)
        convert_t = False
        if np.max(t) > 100:
            convert_t = True
            t -= 273.15

        # determine ps (saturation vapor pressure):
        # Ref: https://www.conservationphysics.org/atmcalc/atmoclc2.pdf
        ps = 610.79 * np.exp(t / (t + 238.3) * 17.2694)
        # determine w (mixing ratio)
        # Ref: http://snowball.millersville.edu/~adecaria/ESCI241/
        # esci241_lesson06_humidity.pdf
        w = h / (1 - h)
        # determine ws (saturation mixing ratio)
        # Ref: http://snowball.millersville.edu/~adecaria/ESCI241/
        # esci241_lesson06_humidity.pdf
        # Ref: https://www.weather.gov/media/epz/wxcalc/mixingRatio.pdf
        ws = 621.97 * (ps / 1000.) / (p - (ps / 1000.))
        # determine RH
        # Ref: https://www.weather.gov/media/epz/wxcalc/mixingRatio.pdf
        rh = w / ws * 100.
        # check values
        rh[rh > 100] = 100
        rh[rh < 2] = 2

        # ensure that pressure is reconverted to mbar
        if convert_p:
            p /= 100
        # ensure that temeprature is reconverted to Kelvin
        if convert_t:
            t += 273.15

        return rh


class DewPoint:
    """Class to derive the dew point from other MERRA2 vars."""

    @staticmethod
    def derive(t, h, p):
        """Derive the dew point from ancillary vars.

        Parameters
        ----------
        t : np.ndarray
            Temperature in Celsius
        h : np.ndarray
            Specific humidity in kg/kg
        p : np.ndarray
            Pressure in Pa

        Returns
        -------
        dp : np.ndarray
            Dew point in Celsius.
        """
        logger.info('Deriving Dew Point from temperature, '
                    'humidity, and pressure')

        # ensure that Temperature is in C (scale from Kelvin if not)
        convert_t = False
        if np.max(t) > 100:
            convert_t = True
            t -= 273.15

        rh = RelativeHumidity.derive(t, h, p)
        dp = (243.04 * (np.log(rh / 100.) + (17.625 * t / (243.04 + t)))
              / (17.625 - np.log(rh / 100.) - ((17.625 * t) / (243.04 + t))))

        # ensure that temeprature is reconverted to Kelvin
        if convert_t:
            t += 273.15

        return dp
