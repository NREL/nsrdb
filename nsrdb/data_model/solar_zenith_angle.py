# -*- coding: utf-8 -*-
"""A framework for handling MERRA2 source data."""
import numpy as np
from nsrdb.solar_position.spa import SPA


class SolarZenithAngle:
    """Class to derive the dew point from other MERRA2 vars."""

    @staticmethod
    def derive(time_index, lat_lon, elev, pressure, temperature):
        """
        Compute the solar zenith angle after atmospheric refraction correction

        Parameters
        ----------
        time_index : ndarray | pandas.DatetimeIndex | str
            Datetime stamps of interest
        lat_lon : ndarray
            (latitude, longitude) for site(s) of interest
        elev : ndarray
            Elevation above sea-level for site(s) of interest
        pressure : ndarray
            Pressure at all sites in millibar
        temperature : ndarray
            Temperature at all sites in C

        Returns
        -------
        theta : ndarray
            Solar zenith after atmospheric refraction correction in degrees
        """

        convert_pressure = False
        if np.max(pressure) > 1e4:
            # ensure that pressure is in mbar (assume Pa if not)
            convert_pressure = True
            pressure /= 100

        apparent_sza = SPA.apparent_zenith(time_index, lat_lon,
                                           elev=elev,
                                           pressure=pressure,
                                           temperature=temperature)

        if convert_pressure:
            pressure *= 100

        return apparent_sza
