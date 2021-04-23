# -*- coding: utf-8 -*-
"""A framework for handling MERRA2 source data."""
import logging
import numpy as np

from nsrdb.solar_position.spa import SPA

logger = logging.getLogger(__name__)


class SolarZenithAngle:
    """Class to derive the solar zenith angle."""

    @staticmethod
    def derive(time_index, lat_lon, elev, pressure, temperature,
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
        pressure : ndarray
            Pressure at all sites in millibar (mbar is same as hPa)
        temperature : ndarray
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
        if np.max(pressure) > 1e4:
            # ensure that pressure is in mbar (assume Pa if not)
            convert_pressure = True
            pressure /= 100

        if len(elev) > big_meta_threshold:
            # SPA is memory intensive so do chunk compute for large grids
            logger.debug('Calculation of SZA is running in {} chunks '
                         'due to large meta data.'.format(n_chunks))
            apparent_sza = np.zeros((len(time_index), len(elev)),
                                    dtype=np.float32)
            chunks_index = np.array_split(np.arange(len(elev)), n_chunks)
            chunks_lat_lon = np.array_split(lat_lon, n_chunks)
            chunks_elev = np.array_split(elev, n_chunks)
            chunks_pressure = np.array_split(pressure, n_chunks, axis=1)
            chunks_temperature = np.array_split(temperature, n_chunks, axis=1)

            for i, index in enumerate(chunks_index):
                chunk_sza = SPA.apparent_zenith(
                    time_index, chunks_lat_lon[i], elev=chunks_elev[i],
                    pressure=chunks_pressure[i],
                    temperature=chunks_temperature[i])
                islice = slice(index[0], index[-1] + 1)
                apparent_sza[:, islice] = chunk_sza
                logger.debug('Calculation chunk {} of {} complete for SZA.'
                             .format(i + 1, n_chunks))

        else:
            apparent_sza = SPA.apparent_zenith(time_index, lat_lon,
                                               elev=elev,
                                               pressure=pressure,
                                               temperature=temperature)
            apparent_sza = apparent_sza.astype(np.float32)

        if convert_pressure:
            pressure *= 100

        return apparent_sza
