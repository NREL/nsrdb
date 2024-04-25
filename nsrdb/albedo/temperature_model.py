"""
Temperature based albedo model using MERRA data
for calculations

Created on Feb 23 2022

@author : bnb32
"""

import logging
from concurrent.futures import as_completed
from datetime import datetime as dt

import numpy as np
import pandas as pd
from rex.utilities.execution import SpawnProcessPool

from nsrdb import DEFAULT_VAR_META
from nsrdb.data_model import DataModel

logger = logging.getLogger(__name__)


class DataHandler:
    """DataHandler for MERRA data used in albedo
    computations
    """

    @staticmethod
    def get_grid(lat, lon, mask):
        """Get grid from composite albedo day instance

        Parameters
        ----------
        lat : ndarray
            array of latitudes for modis grid
        lon : ndarray
            array of longitudes for modis grid
        mask : ndarray
            snow_no_snow mask with 1 for snowy
            cells and 0 for clear cells

        Returns
        -------
        pd.DataFrame
            dataframe with latitudes and longitudes for grid
        """
        lons, lats = np.meshgrid(lon, lat)
        lats = np.array(lats).flatten()
        lons = np.array(lons).flatten()
        snow_mask = mask.flatten() == 1
        lats = lats[snow_mask]
        lons = lons[snow_mask]

        return pd.DataFrame({'latitude': lats, 'longitude': lons})

    @staticmethod
    def get_data(date, merra_path, mask, lat, lon,
                 avg=True, fp_out=None,
                 max_workers=None, n_chunks=64):
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
        max_workers : int | None
            maximum number of workers for loading
            MERRA data
        n_chunks : int
            number of chunks to split full grid into
            for parallel data loading

        Returns
        -------
        ndarray (lat, lon)
            temperature data array on lat/lon grid
        """

        kwargs = {'source_directory': merra_path,
                  'air_temperature': {'elevation_correct': False}}
        var_meta = pd.read_csv(DEFAULT_VAR_META)
        var_meta['source_directory'] = merra_path
        grid = DataHandler.get_grid(lat, lon, mask)

        futures = {}
        grid_chunks = np.array_split(grid, n_chunks)
        now = dt.now()
        loggers = ['nsrdb']
        with SpawnProcessPool(loggers=loggers,
                              max_workers=max_workers) as exe:
            for i, chunk in enumerate(grid_chunks):
                future = exe.submit(DataModel.run_single,
                                    var='air_temperature',
                                    date=date,
                                    nsrdb_grid=chunk,
                                    var_meta=var_meta,
                                    nsrdb_freq='60min',
                                    scale=False,
                                    factory_kwargs=kwargs)
                meta = {'id': i}
                ct = chunk
                meta['lon_min'] = ct['longitude'].min()
                meta['lon_max'] = ct['longitude'].max()
                meta['lat_min'] = ct['longitude'].min()
                meta['lat_max'] = ct['longitude'].max()
                meta['size'] = ct.size
                futures[future] = meta

            logger.info(
                f'Started fetching all merra data chunks in {dt.now() - now}')

            for i, future in enumerate(as_completed(futures)):
                logger.info(f'Future {futures[future]} completed in '
                            f'{dt.now() - now}.')
                logger.info(f'{i+1} out of {len(futures)} futures '
                            'completed')
        logger.info('done processing')

        logger.info('Combining chunks into full temperature array')
        T = np.empty(len(grid), dtype=float)
        pos = 0
        for key in futures:
            if avg:
                res = key.result().mean(axis=0)
            else:
                res = key.result().max(axis=0)
            size = len(res)
            T[pos:pos + size] = res
            pos += size

        if fp_out is not None:
            df = grid
            df['Temperature'] = T
            df.to_csv(fp_out)

        return T


class TemperatureModel:
    """Class to handle MERRA data and compute albedo. Uses
    equations 1 and 2 from Journal of Geophysical Research
    article, A comparison of simulated and observed fluctuations
    in summertime Arctic surface albedo, by Becky Ross and
    John E. Walsh
    """

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
        albedo = T.copy()
        albedo[T < -5] = 0.8
        mask = (T >= -5) & (T < 0)
        albedo[mask] = 0.65 - 0.03 * T[mask]
        albedo[T >= 0] = 0.65
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
        albedo = T.copy()
        albedo[T < 0] = 0.65
        mask = (T >= 0) & (T < 5)
        albedo[mask] = 0.45 + 0.04 * T[mask]
        albedo[T >= 5] = 0.45
        albedo *= 1000
        return albedo

    @classmethod
    def update_snow_albedo(cls, albedo, mask, T):
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

        T : ndarray
            temperature array used to calculate albedo
            (temporal, n_lats * n_lons)

        """
        updated_albedo = albedo.flatten()
        snow_mask = mask.flatten() == 1
        updated_albedo[snow_mask] = cls.get_snow_albedo(T)
        updated_albedo = updated_albedo.reshape(mask.shape)
        albedo[mask == 1] = updated_albedo[mask == 1]

        return albedo
