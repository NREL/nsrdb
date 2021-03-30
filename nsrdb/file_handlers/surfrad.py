# -*- coding: utf-8 -*-
"""Surfrad ground measurement file handler

Created on Tue Jul  9 10:52:44 2019

@author: gbuster
"""
import numpy as np
import pandas as pd

from nsrdb.file_handlers.resource import Resource


class Surfrad(Resource):
    """Framework to open surfrad ground measurement h5 files and format
    dataframes for easy validation against NSRDB data."""

    @staticmethod
    def get_rolling(df, window=61):
        """Get a rolling avg dataset sampled at a given timestep interval.

        Rolling average is centered and ignores nan values.

        Parameters
        ----------
        df : pd.DataFrame
            Timeseries data.
        window : int
            Timesteps that the moving average window will be over.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with same datetimeindex as input df. Each value is
            a moving average of the input df.
        """

        return df.rolling(window, center=True, min_periods=1).mean()

    @staticmethod
    def get_window_size(df, window_minutes=61):
        """calculate the index window size to take a moving average over.

        Parameters
        ----------
        df : pd.DataFrame
            Timeseries data frame with datetime index.
        window_minutes : int
            Minutes that the moving average window will be over.

        Returns
        -------
        window : int
            Number of index values that the window will be over
        """

        one_hr_mask = ((df.index.hour < 1)
                       & (df.index.day == 1)
                       & (df.index.month == 1))
        n_steps = len(np.where(one_hr_mask)[0])
        window = int(np.ceil((window_minutes / 60) * n_steps))

        return window

    @property
    def native_df(self):
        """Get the native measurement data in dataframe format.

        Returns
        -------
        native_df : pd.DataFrame
            Time series dataframe with irradiance data.
        """

        # get the measurement irradiance variables
        dhi_msr = self['dhi'].astype(float).flatten()
        dni_msr = self['dni'].astype(float).flatten()
        ghi_msr = self['ghi'].astype(float).flatten()

        dhi_msr[dhi_msr < 0] = np.nan
        dni_msr[dni_msr < 0] = np.nan
        ghi_msr[ghi_msr < 0] = np.nan

        native_df = pd.DataFrame({'dhi': dhi_msr, 'dni': dni_msr,
                                  'ghi': ghi_msr}, index=self.time_index)
        native_df = native_df.sort_index()
        return native_df

    def get_df(self, dt_out='5min', window_minutes=61):
        """Get rolling avg irradiance df to benchmark against

        The output time index will be the full year index with dt_out, but
        any missing data in the surfrad file will be passed through as nan.

        Parameters
        ----------
        dt_out : str
            Pandas timestep size (30min, 5min, 1min) for the output from
            this method.
        window_minutes : int
            Minutes that the moving average window will be over. This will be
            calculated while considering the source time resolution of the
            SURFRAD measurements.

        Returns
        -------
        df_out : pd.DataFrame
            Dataframe with datetimeindex with dt_out timestep size. Each value
            is a moving average of the measurement data.
        """

        year = self.native_df.index.year[0]

        # final time index
        ti = pd.date_range('1-1-{y}'.format(y=year),
                           '1-1-{y}'.format(y=year + 1),
                           freq=dt_out, tz='UTC')[:-1]
        df_out = pd.DataFrame(index=ti)
        df_temp = pd.DataFrame(index=ti).join(self.native_df, how='outer')

        for var in df_temp:
            window = self.get_window_size(df_temp[var],
                                          window_minutes=window_minutes)

            df_var = self.get_rolling(df_temp[var], window)

            df_out = df_out.join(df_var, how='left')

        return df_out
