# -*- coding: utf-8 -*-
"""A framework for handling MERRA2 source data."""
import logging
import numpy as np

from nsrdb.data_model.base_handler import BaseDerivedVar
from nsrdb.solar_position.spa import SPA

logger = logging.getLogger(__name__)


class SolarZenithAngle(BaseDerivedVar):
    """Class to derive the solar zenith angle."""

    DEPENDENCIES = ('surface_pressure', 'air_temperature')

    @staticmethod
    def derive(time_index, lat_lon, elev, surface_pressure, air_temperature,
               big_meta_threshold=1e6, n_chunks=10):
        """
        Compute the solar zenith angle after atmospheric refraction correction

        Parameters
        ----------
        time_index : ndarray | pandas.DatetimeIndex | str
            Datetime stamps of interest
        lat_lon : ndarray
            (latitude, longitude) for site(s) of interest. Must be a 2D array
            with shape (n_sites, 2).
        elev : ndarray
            Elevation above sea-level for site(s) of interest. Must be a 1D
            array with length equal to the number of sites.
        surface_pressure : ndarray
            Pressure at all sites in millibar (mbar is same as hPa)
        air_temperature : ndarray
            Temperature at all sites in C
        big_meta_threshold : int | float
            Threshold over which a meta data is considered "big" and the
            SZA is computed in chunks to reduce memory usage.
        n_chunks : int
            Number of compute chunks to split the meta data into for "big"
            meta data projects.

        Returns
        -------
        theta : ndarray
            Solar zenith after atmospheric refraction correction in degrees
        """
        logger.info('Deriving Solar Zenith Angle.')

        convert_pressure = False
        if np.max(surface_pressure) > 1e4:
            # ensure that surface_pressure is in mbar (assume Pa if not)
            convert_pressure = True
            surface_pressure /= 100

        if len(elev) > big_meta_threshold:
            # SPA is memory intensive so do chunk compute for large grids
            logger.debug('Calculation of SZA is running in {} chunks '
                         'due to large meta data.'.format(n_chunks))
            apparent_sza = np.zeros((len(time_index), len(elev)),
                                    dtype=np.float32)
            chunks_index = np.array_split(np.arange(len(elev)), n_chunks)
            chunks_lat_lon = np.array_split(lat_lon, n_chunks)
            chunks_elev = np.array_split(elev, n_chunks)
            chunks_p = np.array_split(surface_pressure, n_chunks, axis=1)
            chunks_t = np.array_split(air_temperature, n_chunks, axis=1)

            for i, index in enumerate(chunks_index):

                chunk_sza = SPA.apparent_zenith(
                    time_index, chunks_lat_lon[i], elev=chunks_elev[i],
                    pressure=chunks_p[i], temperature=chunks_t[i])

                islice = slice(index[0], index[-1] + 1)
                apparent_sza[:, islice] = chunk_sza
                logger.debug('Calculation chunk {} of {} complete for SZA.'
                             .format(i + 1, n_chunks))

        else:
            apparent_sza = SPA.apparent_zenith(time_index, lat_lon,
                                               elev=elev,
                                               pressure=surface_pressure,
                                               temperature=air_temperature)
            apparent_sza = apparent_sza.astype(np.float32)

        if convert_pressure:
            surface_pressure *= 100

        return apparent_sza
