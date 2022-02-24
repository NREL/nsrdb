"""Temperature based albedo model using MERRA data
for calculations"""

import pandas as pd
import numpy as np

from nsrdb.data_model import DataModel
from nsrdb import DEFAULT_VAR_META


class DataHandler:
    """DataHandler for MERRA data used in albedo
    computations
    """

    @staticmethod
    def get_grid(lat, lon):
        """Get grid from composite albedo day instance

        Parameters
        ----------
        lat : ndarray
            array of latitudes for modis grid
        lon : ndarray
            array of longitudes for modis grid

        Returns
        -------
        pd.DataFrame
            dataframe with latitudes and longitudes for grid
        """
        lats = [[latitude] * len(lon) for latitude in lat]
        lons = [lon] * len(lat)
        lats = np.array(lats).flatten()
        lons = np.array(lons).flatten()

        return pd.DataFrame({'latitude': lats, 'longitude': lons})

    @staticmethod
    def get_data(date, merra_path, mask, lat, lon):
        """Get temperature data from MERRA

        Parameters
        ----------
        date : datetime.datetime
            date for which to get temperature data
        merra_path : str
            path to merra temperature data
        mask : ndarray
            snow_no_snow mask with 1 for snowy cells
            and 0 for clear cells
        lat : ndarray
            array of latitudes for modis grid
        lon : ndarray
            array of longitudes for modis grid

        Returns
        -------
        ndarray (lat, lon)
            temperature data array on lat/lon grid
        """
        kwargs = {'source_directory': merra_path,
                  'air_temperature': {'elevation_correct': False}}
        var_meta = pd.read_csv(DEFAULT_VAR_META)
        var_meta['source_directory'] = merra_path
        grid = DataHandler.get_grid(lat, lon)
        grid = grid.loc[mask.reshape(-1) == 1]
        data = DataModel.run_single(var='air_temperature',
                                    date=date,
                                    nsrdb_grid=grid,
                                    var_meta=var_meta,
                                    nsrdb_freq='60min', scale=False,
                                    factory_kwargs=kwargs)
        return data


class TemperatureModel:
    """Class to handle MERRA data and compute albedo"""

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

    @classmethod
    def update_snow_albedo(cls, albedo, mask, data):
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
        updated_albedo = np.zeros((mask.shape[0] * mask.shape[1]))
        tmp_mask = mask.reshape(-1) == 1
        tmp_albedo = cls.get_snow_albedo(data)
        updated_albedo[tmp_mask] = tmp_albedo.mean(axis=0)

        updated_albedo = updated_albedo.reshape(mask.shape)
        albedo[mask == 1] = updated_albedo[mask == 1]

        return albedo
