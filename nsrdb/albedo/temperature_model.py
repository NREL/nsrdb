"""Temperature based albedo model using MERRA data
for calculations"""

import pandas as pd

from nsrdb.data_model import DataModel
from nsrdb import DEFAULT_VAR_META


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
        self.var_meta = pd.read_csv(DEFAULT_VAR_META)
        self.var_meta['source_directory'] = source_dir
        self.data = None

    @staticmethod
    def get_albedo(T):
        """Calculate albedo from temperature

        Parameters
        ----------
        T : float | ndarray
            temperature field to use for albedo calculations

        Returns
        -------
        float | ndarray
            albedo field computed from temperature field
        """
        return 1000 * (0.4 + 0.7 * (T - 272.15))

    def get_data(self, date):
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
                                         nsrdb_grid='albedo_snowy_latlonelev',
                                         freq='1hr', scale=False,
                                         var_meta=self.var_meta)
        return self.data

    def update_albedo(self, albedo, snow_no_snow, date):
        """Update albedo array using snow_no_snow mask

        Parameters
        ----------
        albedo : ndarray
            albedo array on lat/lon grid
        snow_no_snow : ndarray
            mask on lat/lon grid with 1 for snow
        date : datetime.datetime
            date for which to get temperature data
            to use in albedo calculations

        Returns
        -------
        albedo : ndarray
            updated albedo array
        """
        T = self.get_data(date)
        albedo[snow_no_snow] = self.get_albedo(T)[snow_no_snow]
        return albedo
