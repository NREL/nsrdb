"""Temperature based albedo model using MERRA data
for calculations"""

import pandas as pd
import numpy as np

from nsrdb.data_model import DataModel


class TemperatureModel:
    """Class to handle MERRA data and compute albedo"""

    def __init__(self, source_dir):
        """Initialize TemperatureModel class with
        source_dir for MERRA data

        Parameters
        ----------
        source_dir : str
            source directory for MERRA data
        """
        self.source_dir = source_dir
        self.kwargs = {'air_temperature': {'elevation_correct': False},
                       'source_directory': source_dir}
        self.data = None

    @staticmethod
    def get_snow_albedo(T):
        """Calculate albedo from temperature

        Parameters
        ----------
        T : ndarray
            temperature field to use for albedo calculations

        Returns
        -------
        ndarray
            albedo field computed from temperature field
        """
        albedo = np.zeros(T.shape)
        albedo[T < -5] = 0.8
        mask = (-5 <= T) & (T < 0)
        albedo[mask] = 0.65 - 0.03 * T[mask]
        albedo[T == 0] = 0.65
        albedo *= 1000
        return albedo

    @staticmethod
    def get_ice_albedo(T):
        """Calculate albedo from temperature

        Parameters
        ----------
        T : ndarray
            temperature field to use for albedo calculations

        Returns
        -------
        ndarray
            albedo field computed from temperature field
        """
        albedo = np.zeros(T.shape)
        albedo[T < 0] = 0.65
        mask = (T >= 0) & (T < 5)
        albedo[mask] = 0.45 + 0.04 * T[mask]
        albedo[T >= 5] = 0.45
        albedo *= 1000
        return albedo

    def get_data(self, date, grid):
        """Get temperature data from MERRA

        Parameters
        ----------
        date : datetime.datetime
            date for which to get temperature data

        Returns
        -------
        ndarray (lat, lon)
            temperature data array on lat/lon grid
        """
        self.data = DataModel.run_single(var='air_temperature',
                                         date=date,
                                         nsrdb_grid=grid,
                                         nsrdb_freq='60min', scale=False,
                                         factory_kwargs=self.kwargs)
        return self.data

    @staticmethod
    def get_grid(cad):
        """Get grid from composite albedo day instance

        Parameters
        ----------
        cad : CompositeAlbedoDay
            CompositeAlbedoDay class instance
            containing albedo calculation methods
            and grid information

        Returns
        -------
        pd.DataFrame
            dataframe with latitudes and longitudes for grid
        """
        lats = []
        lons = []
        for lat in cad._modis.lat:
            for lon in cad._modis.lon:
                lats.append(lat)
                lons.append(lon)
        return pd.DataFrame({'latitude': lats, 'longitude': lons})

    def update_snow_albedo(self, albedo, mask, data):
        """Update albedo array with calculation results

        Parameters
        ----------
        albedo : ndarray
            albedo data array to update
            (n_lats, n_lons)

        mask : ndarray
            mask with 1 at snowy grid cells
            and 0 at cells without snow
            (n_lats, n_lons)

        data : ndarray
            temperature array used to calculate albedo
            (temporal, n_lats * n_lons)

        """
        updated_albedo = self.get_snow_albedo(data)
        updated_albedo = updated_albedo.mean(axis=0)
        updated_albedo = updated_albedo.reshape(mask.shape)

        albedo[mask == 1] = updated_albedo[mask == 1]

        return albedo