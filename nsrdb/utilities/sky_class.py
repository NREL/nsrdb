"""
NSRDB Sky Classification utility using SURFRAD + Clearsky Irradiance.
"""
import logging
import numpy as np
import pandas as pd
from scipy.stats import mode

from farms import SZA_LIM
from farms.utilities import ti_to_radius, calc_beta
from rest2.rest2 import rest2
from rex import MultiFileResource, Resource

from nsrdb.data_model.solar_zenith_angle import SolarZenithAngle
from nsrdb.file_handlers.surfrad import Surfrad

logger = logging.getLogger(__name__)


class SkyClass:
    """Utility class for retrieving SURFRAD validation data alongside NSRDB
    data, determining the sky class by comparison to predicted clearsky
    irradiance, and providing data in a ready-to-validate dataframe.
    """

    # REST2 inputs with NSRDB variable names
    REST_VARS = ('surface_pressure', 'surface_albedo', 'ssa', 'asymmetry',
                 'solar_zenith_angle', 'radius', 'alpha', 'beta', 'ozone',
                 'total_precipitable_water')

    # REST2 input names (order must match REST_VARS)
    ALIASES = ('p', 'albedo', 'ssa', 'g',
               'z', 'radius', 'alpha', 'beta', 'ozone',
               'w')

    def __init__(self, fp_surf, fp_nsrdb, nsrdb_gid,
                 clearsky_ratio=0.9, clear_time_frac=0.8,
                 cloudy_time_frac=0.2, window_minutes=61,
                 min_irradiance=0, sza_lim=89):
        """
        Parameters
        ----------
        fp_surf : str
            Filepath to surfrad h5 file.
        fp_nsrdb : str
            Filepath to NSRDB file. can be a MultiFileResource path with:
            /dir/prefix*suffix.h5
        nsrdb_gid : int
            GID (meta data index) for the site of interest in the fp_nsrdb
            file that matches the fp_surf file.
        clearsky_ratio : float
            Clearsky ratio (ground measurement / clearsky irradiance) above
            which a timestep is considered clear
        clear_time_frac : float
            Fraction of clear timesteps in an averaging window above which the
            whole window is considered clear. Between clear_time_frac and
            cloudy_time_frac is considered broken clouds.
        cloudy_time_frac : float
            Fraction of cloudy timesteps in an averaging window below which the
            whole window is considered cloudy. Between clear_time_frac and
            cloudy_time_frac is considered broken clouds.
        window_minutes : int
            Minutes that the moving average of the sky classification will be
            over. This will be calculated while considering the source time
            resolution of the SURFRAD measurements.
        min_irradiance : float | int
            Minimum irradiance value, timesteps with either ground measured or
            NSRDB irradiance less than this value will be classified as
            missing.
        sza_lim : int | float
            Maximum solar zenith angle, timesteps with sza > sza_lim will
            be classified as missing
        """

        self._fp_surf = fp_surf
        self._fp_nsrdb = fp_nsrdb
        self._gid = nsrdb_gid
        self._cs_ratio = clearsky_ratio
        self._clear_frac = clear_time_frac
        self._cloud_frac = cloudy_time_frac
        self._window_min = window_minutes
        self._min_irrad = min_irradiance
        self._sza_lim = sza_lim

        self._handle_surf = Surfrad(self._fp_surf)
        Handler = MultiFileResource if '*' in self._fp_nsrdb else Resource
        self._handle_nsrdb = Handler(self._fp_nsrdb)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._handle_surf.close()
        self._handle_nsrdb.close()
        if type is not None:
            raise

    @property
    def surfrad(self):
        """Get the initialized Surfrad handler"""
        return self._handle_surf

    @property
    def surf_time_index(self):
        """Get the datetimeindex from the surfrad h5 file"""
        return self.surfrad.time_index

    @property
    def nsrdb(self):
        """Get the initialized Resource or MultiFileResource handler"""
        return self._handle_nsrdb

    @property
    def nsrdb_time_index(self):
        """Get the datetimeindex from the nsrdb h5 file"""
        return self.nsrdb.time_index

    @property
    def surf_ghi(self):
        """Get the surfrad ghi data with negative values as NaN"""
        ghi = self.surfrad['ghi'].astype(np.float32)
        ghi[ghi < 0] = np.nan
        return ghi

    def get_rest_inputs(self):
        """Get a dataframe of NSRDB variables required to run REST2.

        Returns
        -------
        rest_inputs : pd.DataFrame
            Timeseries data with time_index from the surfrad data (might be
            missing time steps) and data columns for each input variable
            required by REST and some extras (e.g. air_temperature).
        """
        rest_inputs = {v: None for v in self.REST_VARS}

        for var in self.REST_VARS:
            temp = self.nsrdb['air_temperature', :, self._gid]
            rest_inputs['air_temperature'] = temp
            if var in self.nsrdb.dsets:
                rest_inputs[var] = self.nsrdb[var, :, self._gid]

        med_p = np.median(rest_inputs['surface_pressure'])
        assert (med_p > 800) & (med_p < 1200)
        beta = calc_beta(self.nsrdb['aod', :, self._gid], rest_inputs['alpha'])
        rest_inputs['beta'] = beta

        rest_inputs = pd.DataFrame(
            rest_inputs, index=self.nsrdb_time_index)
        rest_inputs = rest_inputs.reindex(self.surf_time_index)
        rest_inputs = rest_inputs.interpolate('linear', axis=0).ffill().bfill()

        radius = ti_to_radius(self.surf_time_index, n_cols=1)
        rest_inputs['radius'] = radius

        lat_lon = self.nsrdb.meta.loc[self._gid, ['latitude', 'longitude']]
        lat_lon = lat_lon.values.reshape((1, 2)).astype(float)
        elev = np.array([self.nsrdb.meta.loc[self._gid, 'elevation']])
        sza = SolarZenithAngle.derive(self.surf_time_index, lat_lon, elev,
                                      rest_inputs['surface_pressure'].values,
                                      rest_inputs['air_temperature'].values)
        rest_inputs['solar_zenith_angle'] = sza

        return rest_inputs

    def run_rest(self, rest_inputs):
        """Run REST2 using a dataframe of input data and return clearsky GHI.

        Parameters
        ----------
        rest_inputs : pd.DataFrame
            Timeseries data with time_index from the surfrad data (might be
            missing time steps) and data columns for each input variable
            required by REST and some extras (e.g. air_temperature).

        Returns
        -------
        ghi : np.ndarray
            2D (time, 1) array of clearsky GHI values calculated by REST2.
        """
        kwargs = {self.ALIASES[i]: rest_inputs[var].values
                  for i, var in enumerate(self.REST_VARS)}
        kwargs = {k: np.expand_dims(v, axis=1) for k, v in kwargs.items()}
        ghi = rest2(**kwargs, sza_lim=SZA_LIM).ghi
        return ghi

    def get_comparison_df(self):
        """Get a timeseries dataframe comparing the ground-measured GHI vs.
        the clearsky (REST2) GHI at the ground-measurement time_index.

        Returns
        -------
        df : pd.DataFrame
            Timeseries data with time_index from the surfrad data (might be
            missing time steps) and data columns for ghi_rest (clearsky),
            ghi_ground (surfrad), and "clear", where clear is a boolean
            (1 for clear) with float dtype so it can have NaN values where
            ground measurements are missing.
        """

        df = self.get_rest_inputs()
        df['ghi_rest'] = self.run_rest(df)
        df['ghi_ground'] = self.surf_ghi
        df = df[['solar_zenith_angle', 'ghi_rest', 'ghi_ground']]

        df['clearsky_ratio_ground'] = df['ghi_ground'] / df['ghi_rest']
        df['clear'] = df['clearsky_ratio_ground'] >= self._cs_ratio
        df['clear'] = df['clear'].astype(np.float32)

        df.loc[(np.isnan(df['ghi_ground'])), 'clear'] = np.nan
        df.loc[(df.solar_zenith_angle > SZA_LIM), 'ghi_rest'] = 0
        df.loc[(df.solar_zenith_angle > SZA_LIM), 'ghi_ground'] = 0
        df.loc[(df.solar_zenith_angle > SZA_LIM), 'clear'] = 0

        n = self.surfrad.get_window_size(df, window_minutes=self._window_min)
        time_frac = df['clear'].rolling(n, center=True, min_periods=1).mean()
        time_frac = np.minimum(time_frac, 1)
        time_frac = np.maximum(time_frac, 0)
        df['clear_time_frac'] = time_frac

        csr_ground = df['clearsky_ratio_ground']
        csr_ground = csr_ground.rolling(n, center=True, min_periods=1).mean()
        csr_ground = np.minimum(csr_ground, 2)
        csr_ground = np.maximum(csr_ground, 0)
        df['clearsky_ratio_ground'] = csr_ground

        return df

    def calculate_sky_class(self, df):
        """Calculate the sky class (clear, cloudy, broken, missing) from the
        comparison df.

        Parameters
        ----------
        df : pd.DataFrame
            Timeseries data with time_index from the surfrad data (might be
            missing time steps) and data columns for ghi_rest (clearsky),
            ghi_ground (surfrad), and "clear", where clear is a boolean
            (1 for clear) with float dtype so it can have NaN values where
            ground measurements are missing.

        Returns
        -------
        df : pd.DataFrame
            Same as input but with new column "sky_class" with values
            (clear, cloudy, broken, missing) calculated from the
            clear_time_frac and cloudy_time_frac
            inputs over a time window determined by the window_minutes inputs.
            Note that sky_class == missing means that it is night or there
            is missing ground measurement data and validation should not be
            performed with those timesteps.
        """

        df['sky_class'] = 'missing'

        mask1 = df['clear_time_frac'] >= self._clear_frac
        mask2 = df['clear_time_frac'] < self._cloud_frac

        df.loc[mask1, 'sky_class'] = 'clear'
        df.loc[mask2, 'sky_class'] = 'cloudy'
        df.loc[(~mask1 & ~mask2), 'sky_class'] = 'broken'
        df.loc[np.isnan(df['clear_time_frac']), 'sky_class'] = 'missing'
        df.loc[df.solar_zenith_angle > SZA_LIM, 'sky_class'] = 'missing'

        return df

    def add_validation_data(self, df):
        """Add NSRDB and SURFRAD ghi and dni data to a DataFrame.
        """
        df = df.reindex(self.nsrdb_time_index)
        assert len(df) == len(self.nsrdb_time_index)
        ti_deltas = self.nsrdb_time_index - np.roll(self.nsrdb_time_index, 1)
        ti_deltas_minutes = pd.Series(ti_deltas).dt.seconds / 60
        ti_delta_minutes = int(mode(ti_deltas_minutes)[0])
        freq = '{}T'.format(ti_delta_minutes)
        df = df.drop(['ghi_ground', 'clear'], axis=1)
        surf_df = self.surfrad.get_df(dt_out=freq,
                                      window_minutes=self._window_min)
        surf_df = surf_df.rename({k: '{}_ground'.format(k)
                                  for k in surf_df.columns}, axis=1)
        df = df.join(surf_df, how='left')
        df['dhi_nsrdb'] = self.nsrdb['dhi', :, self._gid]
        df['dni_nsrdb'] = self.nsrdb['dni', :, self._gid]
        df['ghi_nsrdb'] = self.nsrdb['ghi', :, self._gid]
        df['cloud_type'] = self.nsrdb['cloud_type', :, self._gid]
        df['fill_flag'] = self.nsrdb['fill_flag', :, self._gid]

        mask = ((df['ghi_ground'] < self._min_irrad)
                | (df['ghi_nsrdb'] < self._min_irrad)
                | (df['solar_zenith_angle'] > self._sza_lim))
        df.loc[mask, 'sky_class'] = 'missing'

        return df

    @classmethod
    def run(cls, fp_surf, fp_nsrdb, nsrdb_gid,
            clearsky_ratio=0.9, clear_time_frac=0.8,
            cloudy_time_frac=0.2, window_minutes=61,
            min_irradiance=0, sza_lim=89):
        """
        Parameters
        ----------
        fp_surf : str
            Filepath to surfrad h5 file.
        fp_nsrdb : str
            Filepath to NSRDB file. can be a MultiFileResource path with:
            /dir/prefix*suffix.h5
        nsrdb_gid : int
            GID (meta data index) for the site of interest in the fp_nsrdb
            file that matches the fp_surf file.
        clearsky_ratio : float
            Clearsky ratio (ground measurement / clearsky irradiance) above
            which a timestep is considered clear
        clear_time_frac : float
            Fraction of clear timesteps in an averaging window above which the
            whole window is considered clear. Between clear_time_frac and
            cloudy_time_frac is considered broken clouds.
        cloudy_time_frac : float
            Fraction of cloudy timesteps in an averaging window below which the
            whole window is considered cloudy. Between clear_time_frac and
            cloudy_time_frac is considered broken clouds.
        window_minutes : int
            Minutes that the moving average of the sky classification will be
            over. This will be calculated while considering the source time
            resolution of the SURFRAD measurements.
        min_irradiance : float | int
            Minimum irradiance value, timesteps with either ground measured or
            NSRDB irradiance less than this value will be classified as
            missing.
        sza_lim : int | float
            Maximum solar zenith angle, timesteps with sza > sza_lim will
            be classified as missing

        Returns
        -------
        df : pd.DataFrame
            Timeseries of validation data from fp_nsrdb and fp_surf including
            sky classification strings (clear, cloudy, broken, missing)
            with same datetimeindex as the nsrdb file. Note that
            sky_class == missing means that it is night or there is missing
            ground measurement data and validation should not be performed
            with those timesteps.
        """

        with cls(fp_surf, fp_nsrdb, nsrdb_gid,
                 clearsky_ratio=clearsky_ratio,
                 clear_time_frac=clear_time_frac,
                 cloudy_time_frac=cloudy_time_frac,
                 window_minutes=window_minutes,
                 min_irradiance=min_irradiance,
                 sza_lim=sza_lim) as sc:

            df = sc.get_comparison_df()
            df = sc.calculate_sky_class(df)
            df = sc.add_validation_data(df)

        return df
