# -*- coding: utf-8 -*-
"""A framework for handling MERRA2 source data."""
import logging
import numpy as np
import os
import pandas as pd

from nsrdb import DATADIR
from nsrdb.data_model.base_handler import AncillaryVarHandler, BaseDerivedVar
from nsrdb.file_handlers.file_system import NSRDBFileSystem as NFS

logger = logging.getLogger(__name__)


class MerraVar(AncillaryVarHandler):
    """Framework for MERRA source data extraction."""

    # default MERRA paths.
    MERRA_ELEV = os.path.join(DATADIR, 'merra_grid_srtm_500m_stats.csv')

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
        super().__init__(name, var_meta=var_meta, date=date, **kwargs)

    @property
    def pattern(self):
        """Get the source file pattern which is sent to glob().

        Returns
        -------
        str
        """
        pat = super().pattern
        if pat is None:
            pat = os.path.join(self.source_dir, self.file_set,
                               '*{}*'.format(self.date_stamp))

        return pat

    @property
    def next_pattern(self):
        """Get the source file pattern for the next date, which is sent to
        glob().

        Returns
        -------
        str
        """
        pat = super().pattern
        if pat is None:
            pat = os.path.join(self.source_dir, self.file_set,
                               '*{}*'.format(self.next_date_stamp))

        return pat

    def _get_date_stamp(self, key):
        """Get the MERRA datestamp corresponding to the specified datetime date

        Parameters
        ----------
        key : str
            Used to specify whether to get the date stamp for the current date
            (e.g. 'date') or the next date (e.g. 'next_date')

        Returns
        -------
        date : str
            Date stamp that should be in the MERRA file, format is YYYYMMDD
        """
        date_attr = getattr(self, key)
        date = '{y}{m:02d}{d:02d}'.format(y=date_attr.year,
                                          m=date_attr.month,
                                          d=date_attr.day)

        return date

    @property
    def date_stamp(self):
        """Get the MERRA datestamp corresponding to the current datetime date

        Returns
        -------
        date : str
            Date stamp that should be in the MERRA file, format is YYYYMMDD
        """
        return self._get_date_stamp('date')

    @property
    def next_date_stamp(self):
        """Get the MERRA datestamp corresponding to the next datetime date.
        This is used to get the file for the next date for temporal
        interpolation

        Returns
        -------
        date : str
            Date stamp that should be in the MERRA file, format is YYYYMMDD
        """
        return self._get_date_stamp('next_date')

    def pre_flight(self):
        """Perform pre-flight checks - source file check.

        Returns
        -------
        missing : str
            Look for the source file and return the string if not found.
            If nothing is missing, return an empty string.
        """

        missing = ''
        if not NFS(self.file).isfile():
            missing = self.file

        return missing

    @property
    def time_index(self):
        """Get the MERRA native time index.

        Returns
        -------
        MERRA_time_index: pd.DatetimeIndex
            Pandas datetime index for the current day at the MERRA2 resolution
            (1-hour).
        """
        time_index = self._get_time_index(self.date, freq='1h')
        if self.next_file_exists:
            next_time_index = self._get_time_index(self.next_date, freq='1h')
            time_index = time_index.union(next_time_index)
        return time_index

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

    def _get_data(self, file):
        """Get single variable data from the given MERRA source file

        Returns
        -------
        data : np.ndarray
            2D numpy array (time X space) of MERRA data for the specified var.
        """
        # cloud variables when satellite data is not available
        cld_vars = ['cloud_type', 'cld_press_acha', 'cloud_press_acha',
                    'cld_opd_dcomp', 'cld_reff_dcomp']

        # open NetCDF file
        with NFS(file) as f:
            # depending on variable, might need extra logic
            if self.name in ['wind_speed', 'wind_direction']:
                u_vector = f['U2M'][:]
                v_vector = f['V2M'][:]
                if self.name == 'wind_speed':
                    data = np.sqrt(u_vector**2 + v_vector**2)
                else:
                    data = np.degrees(
                        np.arctan2(u_vector, v_vector)) + 180

            elif self.dset_name == 'TOTSCATAU':
                # Single scatter albedo is total scatter / aod
                data = f[self.dset_name][:] / f['TOTEXTTAU'][:]

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
                    elif self.name in ['cld_press_acha',
                                       'cloud_press_acha']:
                        data = np.zeros_like(opd)
                        data = np.where(ctype == 3, 800.0, data)
                        data = np.where(ctype == 6, 250.0, data)

            else:
                data = f[self.dset_name][:]

        # make the data a flat 2d array
        data = self._format_2d(data)

        return data

    @property
    def source_data(self):
        """Get single variable data from the MERRA source file for the current
        date and the next day if available.

        Returns
        -------
        data : np.ndarray
            2D numpy array (time X space) of MERRA data for the specified var.
        """
        data = self._get_data(self.file)
        if self.next_file_exists:
            data = np.concatenate([data, self._get_data(self.next_file)],
                                  axis=0)
        return data

    @property
    def grid(self):
        """Return the MERRA source coordinates with elevation.

        It seems that all MERRA files DO NOT have the same grid.

        Returns
        -------
        self._grid : pd.DataFrame
            MERRA source coordinates with elevation
        """

        if self._grid is None:
            with NFS(self.file) as f:
                lon2d, lat2d = np.meshgrid(f['lon'][:], f['lat'][:])

            self._grid = pd.DataFrame({'longitude': lon2d.ravel(),
                                       'latitude': lat2d.ravel()})

            # merra grid has some bad values around 0 lat/lon
            # quick fix is to set to zero
            self._grid.loc[(self._grid['latitude'] > -0.1)
                           & (self._grid['latitude'] < 0.1),
                           'latitude'] = 0
            self._grid.loc[(self._grid['longitude'] > -0.1)
                           & (self._grid['longitude'] < 0.1),
                           'longitude'] = 0

            # add elevation to coordinate set
            merra_elev = pd.read_csv(self.MERRA_ELEV)
            self._grid = self._grid.merge(merra_elev,
                                          on=['latitude', 'longitude'],
                                          how='left')

            # change column name from merra default
            if 'mean_elevation' in self._grid.columns.values:
                self._grid = self._grid.rename(
                    {'mean_elevation': 'elevation'}, axis='columns')

        return self._grid


class RelativeHumidity(BaseDerivedVar):
    """Class to derive the relative humidity from other MERRA2 vars."""

    DEPENDENCIES = ('air_temperature', 'specific_humidity', 'surface_pressure')

    @staticmethod
    def derive(air_temperature, specific_humidity, surface_pressure):
        """Derive the relative humidity from ancillary vars.

        Parameters
        ----------
        air_temperature : np.ndarray
            Temperature in Celsius
        specific_humidity : np.ndarray
            Specific humidity in kg/kg
        surface_pressure : np.ndarray
            Pressure in Pa

        Returns
        -------
        rh : np.ndarray
            Relative humidity in %.
        """
        logger.info('Deriving Relative Humidity from temperature, '
                    'humidity, and pressure')

        if np.issubdtype(surface_pressure.dtype, np.integer):
            surface_pressure = surface_pressure.astype(np.float32)

        # ensure that Pressure is in Pa (scale from mbar if not)
        convert_p = False
        if np.max(surface_pressure) < 10000:
            convert_p = True
            surface_pressure *= 100
        # ensure that Temperature is in C (scale from Kelvin if not)
        convert_t = False
        if np.max(air_temperature) > 100:
            convert_t = True
            air_temperature -= 273.15

        # determine ps (saturation vapor pressure):
        # Ref: https://www.conservationphysics.org/atmcalc/atmoclc2.pdf
        ps = 610.79 * np.exp(air_temperature / (air_temperature + 238.3)
                             * 17.2694)
        # determine w (mixing ratio)
        # Ref: http://snowball.millersville.edu/~adecaria/ESCI241/
        # esci241_lesson06_humidity.pdf
        w = specific_humidity / (1 - specific_humidity)
        # determine ws (saturation mixing ratio)
        # Ref: http://snowball.millersville.edu/~adecaria/ESCI241/
        # esci241_lesson06_humidity.pdf
        # Ref: https://www.weather.gov/media/epz/wxcalc/mixingRatio.pdf
        ws = 621.97 * (ps / 1000.) / (surface_pressure - (ps / 1000.))
        # determine RH
        # Ref: https://www.weather.gov/media/epz/wxcalc/mixingRatio.pdf
        rh = w / ws * 100.
        # check values
        rh[rh > 100] = 100
        rh[rh < 2] = 2

        # ensure that pressure is reconverted to mbar
        if convert_p:
            surface_pressure /= 100
        # ensure that temeprature is reconverted to Kelvin
        if convert_t:
            air_temperature += 273.15

        return rh


class DewPoint(BaseDerivedVar):
    """Class to derive the dew point from other MERRA2 vars."""

    DEPENDENCIES = ('air_temperature', 'specific_humidity', 'surface_pressure')

    @staticmethod
    def derive(air_temperature, specific_humidity, surface_pressure):
        """Derive the dew point from ancillary vars.

        Parameters
        ----------
        air_temperature : np.ndarray
            Temperature in Celsius
        specific_humidity : np.ndarray
            Specific humidity in kg/kg
        surface_pressure : np.ndarray
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
        if np.max(air_temperature) > 100:
            convert_t = True
            air_temperature -= 273.15

        rh = RelativeHumidity.derive(air_temperature, specific_humidity,
                                     surface_pressure)

        arg1 = np.log(rh / 100.0)
        arg2 = (17.625 * air_temperature) / (243.04 + air_temperature)
        dp = (243.04 * (arg1 + arg2) / (17.625 - arg1 - arg2))

        # ensure that temeprature is reconverted to Kelvin
        if convert_t:
            air_temperature += 273.15

        return dp
