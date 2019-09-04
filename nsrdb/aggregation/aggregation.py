# -*- coding: utf-8 -*-
"""NSRDB aggregation methods
 - 2km 5min CONUS -> 4km 30min NSRDB PSM v3 Meta
 - 2km 15min East -> 4km 30min NSRDB PSM v3 Meta
 - 4km 30min West -> 4km 30min NSRDB PSM v3 Meta
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from scipy.stats import mode
from warnings import warn

from nsrdb.all_sky.utilities import calc_dhi
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.utilities.plots import Spatial
from nsrdb.utilities.interpolation import temporal_step
from nsrdb.utilities.loggers import init_logger
from nsrdb.utilities.execution import SLURM

logger = logging.getLogger(__name__)


# Standard configs.
NSRDB_4km_30min = {'east': {'data_sub_dir': 'east',
                            'tree_file': 'kdtree_nsrdb_meta_2km_east.pkl',
                            'meta_file': 'nsrdb_meta_2km_east.csv',
                            'spatial': '2km',
                            'temporal': '15min'},
                   'west': {'data_sub_dir': 'west',
                            'tree_file': 'kdtree_west_psm_extent.pkl',
                            'meta_file': 'west_psm_extent.csv',
                            'spatial': '4km',
                            'temporal': '30min'},
                   'conus': {'data_sub_dir': 'conus',
                             'tree_file': 'kdtree_nsrdb_meta_2km_conus.pkl',
                             'meta_file': 'nsrdb_meta_2km_conus.csv',
                             'spatial': '2km',
                             'temporal': '5min'},
                   'final': {'data_sub_dir': 'nsrdb_4km_30min',
                             'fout': 'nsrdb_2018.h5',
                             'tree_file': 'kdtree_nsrdb_meta_4km.pkl',
                             'meta_file': 'nsrdb_meta_4km.csv',
                             'spatial': '4km',
                             'temporal': '30min'},
                   }

SURFRAD = {'east': {'data_sub_dir': 'east',
                    'tree_file': 'kdtree_nsrdb_meta_2km_east.pkl',
                    'meta_file': 'nsrdb_meta_2km_east.csv',
                    'spatial': '2km',
                    'temporal': '15min'},
           'west': {'data_sub_dir': 'west',
                    'tree_file': 'kdtree_west_psm_extent.pkl',
                    'meta_file': 'west_psm_extent.csv',
                    'spatial': '4km',
                    'temporal': '30min'},
           'conus': {'data_sub_dir': 'conus',
                     'tree_file': 'kdtree_nsrdb_meta_2km_conus.pkl',
                     'meta_file': 'nsrdb_meta_2km_conus.csv',
                     'spatial': '2km',
                     'temporal': '5min'},
           'final': {'data_sub_dir': 'nsrdb_4km_30min',
                     'fout': 'nsrdb_surfrad_2018.h5',
                     'tree_file': 'kdtree_surfrad_meta.pkl',
                     'meta_file': 'surfrad_meta.csv',
                     'spatial': '4km',
                     'temporal': '30min'},
           }


class MetaManager:
    """Framework to parse the final meta data for contributing sources."""

    @staticmethod
    def meta_sources_2018(fpath_4km):
        """Make 2018 4km meta data with data source column (west/east/conus).

        WARNING: This is a script very specific to the 2018 GOES data
        arrangement, with the 4km 30min GOES West satellite, the 2km 15min GOES
        East satellite, and the 2km 5min CONUS data from GOES East. This only
        works with the psm v3 4km meta data (accessed on 8/28/2019).

        Parameters
        ----------
        fpath_4km : str
            File path to full 4km meta data.

        Returns
        -------
        meta : pd.DataFrame
            DataFrame based on fpath_4km but with a "source" column containing
            the data source string.
        """

        meta = pd.read_csv(fpath_4km, index_col=0)
        meta['source'] = 'west'

        # east 2km longitude boundary is at -125 lon (just west of CONUS)
        east_mask = (meta.longitude > -125.0)
        meta.loc[east_mask, 'source'] = 'east'

        # conus includes all of US except for Alaska and Hawaii
        conus_mask = ((meta.country == 'United States')
                      & ~meta.state.isin(['Alaska', 'Hawaii']))
        meta.loc[conus_mask, 'source'] = 'conus'

        # made a line specific to the observed 2018 GOES East extreme angle
        # boundary, above which no cloud properties are returned for the East.
        lat_boundary = 0.6 * (meta.longitude.values + 125) + 42.7
        angle_mask = ((meta.latitude > lat_boundary)
                      & (meta.source != 'west')
                      & (meta.longitude < -104.5))
        meta.loc[angle_mask, 'source'] = 'west'

        return meta

    @staticmethod
    def plot_meta_source(fpath_4km, fname, out_dir, **kwargs):
        """Make a map plot of the NSRDB Meta source data (west/east/conus).

        Parameters
        ----------
        fpath_4km : str
            File path to full 4km meta data.
        fname : str
            Filename for output map image file.
        out_dir : str
            Directory path to save map plot file.
        **kwargs : dict
            Keyword args for spatial plotting utility.
        """

        meta = MetaManager.meta_sources_2018(fpath_4km)
        sources = list(set(meta.source.unique()))
        meta['isource'] = np.nan
        for i, source in enumerate(sources):
            meta.loc[(meta.source == source), 'isource'] = i

        meta = meta[['latitude', 'longitude', 'isource']]
        Spatial.plot_geo_df(meta, fname, out_dir, **kwargs)


class Aggregation:
    """Framework for performing spatiotemporal aggregation."""

    def __init__(self, var, data_fpath, nn, w, final_ti):
        """
        Parameters
        ----------
        var : str
            Variable (dataset) name being aggregated.
        data_fpath : str
            Filepath to h5 file containing source var data.
        nn : np.ndarray
            1D array of site (column) indices in data_fpath to aggregate.
        w : int
            Window size for temporal aggregation.
        final_ti : pd.DateTimeIndex
            Final datetime index (used to ensure the aggregated profile has
            correct length).
        """
        self.var = var
        self.data_fpath = data_fpath
        self.nn = sorted(list(nn))
        self.w = w
        self.final_ti = final_ti

    @property
    def source_time_index(self):
        """Get the time index of the source data.

        Returns
        -------
        time_index : pd.Datetimeindex
            Datetimeindex of the source dataset.
        """

        with Outputs(self.data_fpath) as out:
            ti = out.time_index
        return ti

    @property
    def data(self):
        """Get the timeseries data for the specified var and sites.

        Returns
        -------
        _data : np.ndarray
            Unscaled float data array with shape (ti, nn) where ti is the
            native time index length and nn is the number of neighbors in
            the self.nn attr.
        """

        with h5py.File(self.data_fpath, 'r') as f:
            _data = f[self.var][:, self.nn].astype(np.float32)
        return _data

    @staticmethod
    def spatial_avg(data):
        """Average the source data across the spatial extent.

        Returns
        -------
        data : np.ndarray
            Unscaled float data array with shape (ti, ) where ti is the
            native time index length the data was averaged accross all
            nn neighbors.
        """

        bad_rows = np.isnan(data).all(axis=1)

        if any(bad_rows):
            data[bad_rows, :] = 0
            m = np.nanmean(data, axis=1)
            m[bad_rows] = np.nan
        else:
            m = np.mean(data, axis=1)

        return m

    @staticmethod
    def spatial_sum(data):
        """Sum the source data across the spatial extent.

        Returns
        -------
        data : np.ndarray
            Unscaled float data array with shape (ti, ) where ti is the
            native time index length the data was summed accross all
            nn neighbors.
        """

        return data.sum(axis=1)

    def time_avg(self, inp):
        """Calculate the rolling time average for an input array or df.

        Parameters
        ----------
        inp : np.ndarray | pd.DataFrame
            Input array/df with data to average.

        Returns
        -------
        out : np.ndarray | pd.DataFrame
            Array or dataframe with same size as input and each value is a
            moving average.
        """

        array = False
        if isinstance(inp, np.ndarray):
            array = True
            inp = pd.DataFrame(inp)

        out = inp.rolling(self.w, center=True, min_periods=1).mean()

        if array:
            out = out.values

        return out

    def time_sum(self, inp):
        """Calculate the rolling sum for an input array or df.

        Parameters
        ----------
        inp : np.ndarray | pd.DataFrame
            Input array/df with data to sum.

        Returns
        -------
        out : np.ndarray | pd.DataFrame
            Array or dataframe with same size as input and each value is a
            moving sum.
        """

        array = False
        if isinstance(inp, np.ndarray):
            array = True
            inp = pd.DataFrame(inp)

        out = inp.rolling(self.w, center=True, min_periods=1).sum()

        if array:
            out = out.values

        return out

    @staticmethod
    def _get_rolling_window_index(L, w):
        """Get an indexing array to index a 2D array on a rolling window.

        Parameters
        ----------
        L : int
            Length of the 2D array to apply a rolling window over
        w : int
            Window size.

        Returns
        -------
        iarr : np.ndarray
            Array of index arrays.
        """

        iarr = np.zeros((L, w), dtype=np.uint32)
        for i in range(L):
            sub = [i + n for n in range(w)]
            sub -= (np.round(w / 2) - 1)
            iarr[i, :] = sub

        iarr = np.maximum(iarr, 0)
        iarr = np.minimum(iarr, L - 1)

        return iarr

    @staticmethod
    def cloud_type_mode(data, w):
        """Get the mode of a 2D cloud type array using a rolling time window.

        Parameters
        ----------
        data : np.ndarray
            2D array of integer cloud types.
        w : int
            Temporal window over which to take the mode.

        Returns
        -------
        data : np.ndarray
            Mode of cloud type.
        """

        iarr = Aggregation._get_rolling_window_index(len(data), w)

        cloud_mode = np.ndarray((len(data), ), dtype=data.dtype)
        for i, x in enumerate(data[iarr]):
            m = mode(x.flatten())[0]
            if np.isnan(m):
                emsg = ('Bad cloud type mode from ctype array: \n\t{}'
                        .format(x))
                logger.info('Bad ctype mode at i {}'.format(i))
                logger.exception(emsg)
                raise ValueError(emsg)
            else:
                cloud_mode[i] = m

        return cloud_mode

    def reduce_timeseries(self, arr):
        """Reduce a high res timeseries to a coarse timeseries.

        Parameters
        ----------
        arr : np.ndarray
            2D numpy array

        Returns
        -------
        arr : np.ndarray
            Shortened 2D numpy array with length equal to the final ti.
        """

        m = len(arr) / len(self.final_ti)
        if m % 1 != 0:
            raise ValueError('Cannot reduce timeseries! Final ti has shape '
                             '{} and working array has shape {}.'
                             .format(self.final_ti.shape, arr.shape))
        m = int(m)

        if len(arr.shape) == 1:
            arr = arr[::m]
        else:
            arr = arr[::m, :]

        if len(arr) != len(self.final_ti):
            raise ValueError('Timeseries length reduction failed! Final ti '
                             'has shape {} and reduced array has shape {}.'
                             .format(self.final_ti.shape, arr.shape))

        return arr

    @staticmethod
    def format_out_arr(arr):
        """Format the output array (round and flatten)."""
        return np.round(arr).flatten()

    @classmethod
    def point(cls, var, data_fpath, nn, w, final_ti):
        """Run agg by selecting just the closest site and timestep.

        Parameters
        ----------
        var : str
            Variable (dataset) name being aggregated.
        data_fpath : str
            Filepath to h5 file containing source var data.
        nn : np.ndarray
            1D array of site (column) indices in data_fpath to aggregate.
        w : int
            Window size for temporal aggregation.
        final_ti : pd.DateTimeIndex
            Final datetime index (used to ensure the aggregated profile has
            correct length).

        Returns
        -------
        data : np.ndarray
            (n, ) array unscaled and rounded data from the nn with time
            series matching final_ti.
        """
        nn = [list(nn)[0]]
        a = cls(var, data_fpath, nn, w, final_ti)
        data = a.reduce_timeseries(a.data)
        data = a.format_out_arr(data)
        return data

    @classmethod
    def dhi(cls, var, i, fout):
        """Calculate the aggregated DHI from an aggregated output file.

        Parameters
        ----------
        var : str
            Variable name, either "dhi" or "clearsky_dhi".
        i : int
            Site index in fout.
        fout : str
            Filepath to the output file containing aggregated GHI, DNI, and
            SZA to calculate aggregated DHI.

        Returns
        -------
        dhi : np.ndarray
            DHI calcualted from vars in fout.
        """

        var_ghi = var.replace('dhi', 'ghi')
        var_dni = var.replace('dhi', 'dni')

        with Outputs(fout) as out:
            attrs = out.get_attrs(dset=var)
            ghi = out[var_ghi, :, i]
            dni = out[var_dni, :, i]
            sza = out['solar_zenith_angle', :, i]

        dhi = calc_dhi(dni, ghi, sza)[0]
        dhi *= float(attrs.get('scale_factor', 1))
        dhi = cls.format_out_arr(dhi)
        return dhi

    @classmethod
    def fill_flag(cls, var, data_fpath, nn, w, final_ti):
        """Run fill flag aggregation, returning the percentage of timesteps
        that were filled.

        Parameters
        ----------
        var : str
            Variable (dataset) name being aggregated (fill_flag).
        data_fpath : str
            Filepath to h5 file containing source var data.
        nn : np.ndarray
            1D array of site (column) indices in data_fpath to aggregate.
        w : int
            Window size for temporal aggregation.
        final_ti : pd.DateTimeIndex
            Final datetime index (used to ensure the aggregated profile has
            correct length).

        Returns
        -------
        data : np.ndarray
            (n, ) array unscaled and rounded data from the nn with time
            series matching final_ti.
        """
        a = cls(var, data_fpath, nn, w, final_ti)
        data = a.data
        data[(data > 0)] = 1
        data = a.spatial_sum(data)
        data = a.time_sum(data)
        data = a.reduce_timeseries(data)
        data /= (len(nn) * w / 100)
        data = a.format_out_arr(data)
        return data

    @classmethod
    def cloud_type(cls, var, data_fpath, nn, w, final_ti):
        """Run cloud type aggregation, returning the most common cloud type.

        Parameters
        ----------
        var : str
            Variable (dataset) name being aggregated (cloud_type).
        data_fpath : str
            Filepath to h5 file containing source var data.
        nn : np.ndarray
            1D array of site (column) indices in data_fpath to aggregate.
        w : int
            Window size for temporal aggregation.
        final_ti : pd.DateTimeIndex
            Final datetime index (used to ensure the aggregated profile has
            correct length).

        Returns
        -------
        data : np.ndarray
            (n, ) array unscaled and rounded data from the nn with time
            series matching final_ti.
        """
        a = cls(var, data_fpath, nn, w, final_ti)
        data = a.data
        data[(data == 1)] = 0
        data = a.cloud_type_mode(data, w)
        data = a.reduce_timeseries(data)
        data = a.format_out_arr(data)
        return data

    @staticmethod
    def cloud_property_avg(cprop_source, ctype_source, ctype_out_full, w):
        """Run cloud property aggregation based on output cloud type.

        Parameters
        ----------
        cprop_source : np.ndarray
            Source (full resolution) cloud property data.
        ctype_source : np.ndarray
            Source (full resolution) cloud type data.
        ctype_out_full : np.ndarray
            Output (reduced resolution) cloud type data, interpolated to the
            same length as the source resolution.
        w : int
            Window size.

        Returns
        -------
        cprop_out : np.ndarray
            Average cloud property data in the window surrounding each
            timestep masked by cloud type output == cloud type source.
            Shape is same as ctype_out_full.
        """

        iarr = Aggregation._get_rolling_window_index(len(ctype_source), w)

        cprop_out = np.ndarray((len(cprop_source), ),
                               dtype=cprop_source.dtype)

        for i, j in enumerate(iarr):
            mask = (ctype_source[j] == ctype_out_full[i])
            if mask.any():
                cprop_out[i] = np.mean(cprop_source[j][mask])
            else:
                cprop_out[i] = np.nan

        cprop_out[(ctype_out_full[:, 0] == 0)] = 0

        return cprop_out

    @classmethod
    def cloud_property(cls, var, data_fpath, nn, w, final_ti, gid, fout):
        """Run cloud type aggregation, returning the most common cloud type.

        Parameters
        ----------
        var : str
            Variable (dataset) name being aggregated (cloud_type).
        data_fpath : str
            Filepath to h5 file containing source var data.
        nn : np.ndarray
            1D array of site (column) indices in data_fpath to aggregate.
        w : int
            Window size for temporal aggregation.
        final_ti : pd.DateTimeIndex
            Final datetime index (used to ensure the aggregated profile has
            correct length).
        gid : int
            Site index in fout.
        fout : str
            Filepath to the output file containing aggregated cloud type.

        Returns
        -------
        data : np.ndarray
            (n, ) array unscaled and rounded data from the nn with time
            series matching final_ti.
        """

        with Outputs(fout) as out:
            ctype_out_final = out['cloud_type', :, gid]
            sza = out['solar_zenith_angle', :, gid]

        a = cls('cloud_type', data_fpath, nn, w, final_ti)
        ctype_source = a.data
        ctype_source[(ctype_source == 1)] = 0

        ctype_out_full = temporal_step(ctype_out_final, final_ti,
                                       a.source_time_index)

        a = cls(var, data_fpath, nn, w, final_ti)
        cprop_source = a.data

        cprop_out = a.cloud_property_avg(cprop_source, ctype_source,
                                         ctype_out_full, w)

        cprop_out = a.reduce_timeseries(cprop_out)
        cprop_out = a.format_out_arr(cprop_out)

        if np.isnan(cprop_out).sum():
            emsg = ('Aggregation of cloud property failed for site '
                    '{}, {} NaN values persisted.'
                    .format(gid, np.isnan(cprop_out).sum()))
            bad_locs = np.where(np.isnan(cprop_out))[0]
            logger.info('Bad locs: {}'.format(bad_locs))
            logger.info('Bad time index: {}'.format(final_ti[bad_locs]))
            logger.info('Ctypes: {}'.format(ctype_out_final[bad_locs]))
            logger.info('SZA: {}'.format(sza[bad_locs]))
            i0 = np.max(bad_locs[0], 0) - 5
            i1 = i0 + 10
            logger.info('Cprop from {} to {}: {}'
                        .format(i0, i1, cprop_out[i0:i1]))
            logger.exception(emsg)
            raise ValueError(emsg)

        return cprop_out

    @classmethod
    def mean(cls, var, data_fpath, nn, w, final_ti):
        """Run agg using a spatial average and temporal moving window average.

        Parameters
        ----------
        var : str
            Variable (dataset) name being aggregated.
        data_fpath : str
            Filepath to h5 file containing source var data.
        nn : np.ndarray
            1D array of site (column) indices in data_fpath to aggregate.
        w : int
            Window size for temporal aggregation.
        final_ti : pd.DateTimeIndex
            Final datetime index (used to ensure the aggregated profile has
            correct length).

        Returns
        -------
        data : np.ndarray
            (n, ) array unscaled and rounded data from the nn with time
            series matching final_ti.
        """

        a = cls(var, data_fpath, nn, w, final_ti)
        data = a.spatial_avg(a.data)
        data = a.time_avg(data)
        data = a.reduce_timeseries(data)
        data = a.format_out_arr(data)
        return data


class Manager:
    """Framework for aggregation to a final NSRDB spatiotemporal resolution."""

    DEFAULT_METHOD = Aggregation.mean

    AGG_METHODS = {'alpha': Aggregation.point,
                   'aod': Aggregation.point,
                   'asymmetry': Aggregation.point,
                   'ozone': Aggregation.point,
                   'ssa': Aggregation.point,
                   'surface_albedo': Aggregation.point,
                   'surface_pressure': Aggregation.point,
                   'total_precipitable_water': Aggregation.point,
                   'dew_point': Aggregation.point,
                   'relative_humidity': Aggregation.point,
                   'air_temperature': Aggregation.point,
                   'wind_direction': Aggregation.point,
                   'wind_speed': Aggregation.point,
                   'cloud_type': Aggregation.cloud_type,
                   'cld_opd_dcomp': Aggregation.cloud_property,
                   'cld_reff_dcomp': Aggregation.cloud_property,
                   'cld_press_acha': Aggregation.cloud_property,
                   'solar_zenith_angle': Aggregation.point,
                   'dhi': Aggregation.dhi,
                   'dni': Aggregation.mean,
                   'ghi': Aggregation.mean,
                   'clearsky_dhi': Aggregation.dhi,
                   'clearsky_dni': Aggregation.mean,
                   'clearsky_ghi': Aggregation.mean,
                   'fill_flag': Aggregation.fill_flag,
                   }

    def __init__(self, data, data_dir, meta_dir, year=2018,
                 i_chunk=0, n_chunks=1):
        """
        Parameters
        ----------
        data : dict
            Nested dictionary containing data on all NSRDB data sources
            (east, west, conus) and the final aggregated output.
        data_dir : str
            Root directory containing sub dirs with all data sources.
        meta_dir : str
            Directory containing meta and ckdtree files for each data source
            and the final aggregated output.
        year : int
            Year being analyzed.
        i_chunk : int
            Meta data chunk index currently being processed (zero indexed).
        n_chunks : int
            Number of chunks to process the meta data in.
        """

        self.data = data
        self.data_dir = data_dir
        self.meta_dir = meta_dir
        self.year = year
        self.n_chunks = n_chunks
        self.i_chunk = i_chunk
        self._meta = None
        self._meta_chunk = None

        self.parse_data()
        self.preflight()
        self.run_nn()
        self.add_temporal()
        self._init_fout()

    def parse_data(self):
        """Parse the data input for several useful attributes."""
        self.final_sres = self.data['final']['spatial']
        self.final_tres = self.data['final']['temporal']
        self.fout = os.path.join(self.data_dir,
                                 self.data['final']['data_sub_dir'] + '/',
                                 self.data['final']['fout'])
        self.fout = self.fout.replace('.h5', '_{}.h5'.format(self.i_chunk))
        self.data_sources = self.meta.source.unique()

        logger.info('Data sources: {}'.format(self.data_sources))

        out_dir = os.path.join(self.data_dir,
                               self.data['final']['data_sub_dir'] + '/')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def preflight(self, reqs=('data_sub_dir', 'tree_file', 'meta_file',
                              'spatial', 'temporal')):
        """Run validity checks on input data.

        Parameters
        ----------
        reqs : list | tuple
            Required fields for each source dataset.
        """

        sources = self.data_sources
        for source in sources:
            if source not in self.data:
                warn('Need "{}" in the data input!'.format(source))
            for r in reqs:
                if r not in self.data[source]:
                    warn('Data input source "{}" needs field "{}"!'
                         .format(source, r))

    @property
    def time_index(self):
        """Get the final time index.

        Returns
        -------
        ti : pd.DatetimeIndex
            Time index for the intended year at the final (aggregated) time
            resolution.
        """

        ti = pd.date_range('1-1-{}'.format(self.year),
                           '1-1-{}'.format(self.year + 1),
                           freq=self.final_tres)[:-1]
        return ti

    @property
    def meta(self):
        """Get the final meta data with sources.

        Returns
        -------
        meta : pd.DataFrame
            Meta data for the final (aggregated) datasets with data source col.
        """

        if self._meta is None:
            meta_path = os.path.join(self.meta_dir,
                                     self.data['final']['meta_file'])
            self._meta = MetaManager.meta_sources_2018(meta_path)
        return self._meta

    @property
    def meta_chunk(self):
        """Get the meta data for just this chunk of sites based on
        n_chunks and i_chunk.

        Returns
        -------
        meta_chunk : pd.DataFrame
            Meta data reduced to a chunk of sites based on n_chunks and i_chunk
        """

        if self._meta_chunk is None:
            gids_full = np.arange(len(self.meta))
            gid_chunk = np.array_split(gids_full, self.n_chunks)[self.i_chunk]
            self._meta_chunk = self._meta.iloc[gid_chunk, :]
            self._meta_chunk['gid'] = self._meta_chunk.index.values
            logger.info('Working on meta chunk {} out of {} '
                        'with GIDs {} through {}'
                        .format(self.i_chunk + 1, self.n_chunks,
                                np.min(gid_chunk), np.max(gid_chunk)))
        return self._meta_chunk

    @staticmethod
    def get_dset_attrs(h5dir):
        """Get output file dataset attributes for a set of datasets.

        Parameters
        ----------
        h5dir : str
            Path to directory containing multiple h5 files with all available
            dsets.

        Returns
        -------
        dsets : list
            List of datasets.
        attrs : dict
            Dictionary of dataset attributes keyed by dset name.
        chunks : dict
            Dictionary of chunk tuples keyed by dset name.
        dtypes : dict
            dictionary of numpy datatypes keyed by dset name.
        """

        dsets = []
        attrs = {}
        chunks = {}
        dtypes = {}

        h5_files = [fn for fn in os.listdir(h5dir) if fn.endswith('.h5')]

        for fn in h5_files:
            with Outputs(os.path.join(h5dir, fn)) as out:
                for d in out.dsets:
                    if d not in ['time_index', 'meta'] and d not in attrs:
                        attrs[d] = out.get_attrs(dset=d)
                        _, dtypes[d], chunks[d] = out.get_dset_properties(d)

                        if 'dhi' in d:
                            attrs[d]['units'] = 'percent of filled timesteps'

        dsets = list(attrs.keys())

        return dsets, attrs, chunks, dtypes

    def _init_fout(self):
        """Initialize the output file with all datasets and final
        time index and meta"""

        data_sub_dir = self.data[self.data_sources[0]]['data_sub_dir'] + '/'

        self.dsets, self.attrs, self.chunks, self.dtypes = \
            self.get_dset_attrs(os.path.join(self.data_dir, data_sub_dir))

        if not os.path.exists(self.fout):
            logger.info('Initializing output file: {}'.format(self.fout))
            Outputs.init_h5(self.fout, self.dsets, self.attrs, self.chunks,
                            self.dtypes, self.time_index, self.meta_chunk)
        else:
            logger.info('Output file exists: {}'.format(self.fout))

    def _init_arr(self, var):
        """Initialize a numpy array for a given var for current chunk of sites.

        Returns
        -------
        arr : np.ndarray
            Numpy array for var with final disk dtype and shape (t, n) where
            t is the final time index and n is the number of sites in this
            meta chunk.
        """
        arr = np.zeros((len(self.time_index), len(self.meta_chunk)),
                       self.dtypes[var])
        logger.debug('Initializing array for "{}" with shape {}'
                     .format(var, arr.shape))
        return arr

    @staticmethod
    def _get_spatial_k(sres, final_sres):
        """Get the required number of nearest neighbors based on spatial res.

        Parameters
        ----------
        sres : str
            Spatial resolution of the source data (4km, 2km).
        final_sres : str
            Spatial resolution of the final dataset (4km).

        Returns
        -------
        k : int
            Number of neighbors in the source data to aggregate to a final
            dataset site.
        """

        if final_sres == '4km':
            if sres == '4km':
                k = 1
            elif sres == '2km':
                k = 4
            else:
                raise ValueError('Did not recognize spatial resolution {}'
                                 .format(sres))
        else:
            raise ValueError('Did not recognize final spatial resolution: '
                             '{}'.format(final_sres))
        return k

    @staticmethod
    def _get_temporal_w(tres, final_tres):
        """Get the required moving window size for temporal agg

        Parameters
        ----------
        tres : str
            Temporal resolution of the source data (30min, 15min, 5min).
        final_tres : str
            Temporal resolution of the final dataset (30min).

        Returns
        -------
        w : int
            Window size to consider for the temporal aggregation to the
            final_tres.
        """

        if final_tres == '30min':
            if tres == '30min':
                w = 1
            elif tres == '15min':
                w = 3
            elif tres == '5min':
                w = 7
            else:
                raise ValueError('Did not recognize temporal resolution {}'
                                 .format(tres))
        else:
            raise ValueError('Did not recognize final temporal resolution: '
                             '{}'.format(final_tres))

        return w

    def _get_fpath(self, var, data_dir, data_sub_dir, source):
        """Get the h5 filepath in data_dir/data_sub_dir/ containing var.

        Parameters
        ----------
        var : str
            Variable name (h5 dataset) being searched for.
        data_dir : str
            Root data directory.
        data_sub_dir : str
            Sub directory in data_dir containing h5 files.
        source : str
            Data source (conus, east, west).

        Returns
        -------
        fpath : str
            File path to h5 file in data_sub_dir containing var dataset.
        """

        if var in self.data[source]:
            fpath = self.data[source][var]
        else:
            if not data_sub_dir.endswith('/'):
                data_sub_dir += '/'
            for fn in os.listdir(os.path.join(data_dir, data_sub_dir)):
                if fn.endswith('.h5'):
                    fpath = os.path.join(data_dir, data_sub_dir, fn)
                    with Outputs(fpath) as out:

                        if (var == 'fill_flag' and var in out.dsets
                                and 'irradiance' in fn):
                            break
                        elif var != 'fill_flag' and var in out.dsets:
                            break
            self.data[source][var] = fpath
        return fpath

    def add_temporal(self):
        """Get the temporal window sizes for all data sources."""
        for source in self.data_sources:
            w = self._get_temporal_w(self.data[source]['temporal'],
                                     self.final_tres)
            self.data[source]['window'] = w
            logger.info('Data source {} has a window size of {}'
                        .format(source, w))

    def run_nn(self):
        """Run nearest neighbor for all data sources against the final meta."""
        for source in self.data_sources:

            k = self._get_spatial_k(self.data[source]['spatial'],
                                    self.final_sres)
            logger.info('Data source {} will use {} neighbors'
                        .format(source, k))
            tree_fpath = os.path.join(self.meta_dir,
                                      self.data[source]['tree_file'])
            _, i = self.knn(self.meta, tree_fpath, k=k)

            self.data[source]['nn'] = i

    @staticmethod
    def knn(meta, tree_fpath, k=1):
        """Run KNN between the final meta data and the pickled ckdtree.

        Parameters
        ----------
        meta : pd.DataFrame
            Final meta data.
        tree_fpath : str
            Filepath to a pickled ckdtree.
        k : int
            Number of neighbors to query.

        Returns
        -------
        d : np.ndarray
            Distance results. Shape is (len(meta), k)
        i : np.ndarray
            Index results. Shape is (len(meta), k)
        """
        with open(tree_fpath, 'rb') as pkl:
            tree = pickle.load(pkl)
        d, i = tree.query(meta[['latitude', 'longitude']], k=k)
        if len(i.shape) == 1:
            d = d.reshape((len(i), 1))
            i = i.reshape((len(i), 1))
        return d, i

    def _get_agg_method(self, var):
        """Get the aggregation method for a given variable.

        Parameters
        ----------
        var : str
            Variable name

        Returns
        -------
        method : function
            Aggregation method for the input var from the Aggregation class
            above.
        """

        if var in self.AGG_METHODS:
            method = self.AGG_METHODS[var]
        else:
            method = self.DEFAULT_METHOD
        return method

    def _get_args(self, var, i):
        """Get an argument list for a given variable and site.

        Parameters
        ----------
        var : str
            Variable name.
        i : int
            Site index number in current meta chunk.

        Returns
        -------
        args : list
            List of arguments
        """

        if 'dhi' in var:
            args = [var, i, self.fout]

        else:
            gid = self.meta_chunk.index.values[i]
            source = self.meta_chunk.iloc[i, :]['source']
            data_sub_dir = self.data[source]['data_sub_dir']
            nn = self.data[source]['nn'][gid, :]
            w = self.data[source]['window']
            data_fpath = self._get_fpath(var, self.data_dir, data_sub_dir,
                                         source)

            args = [var, data_fpath, nn, w, self.time_index]

        if 'cld_' in var:
            args.append(gid)
            args.append(self.fout)

        return args

    def _agg_var_serial(self, var, method):
        """Aggregate one var for all sites in this chunk in parallel.

        Parameters
        ----------
        var : str
            Variable name being aggregated.
        method : function
            Aggregation method.

        Returns
        -------
        arr : np.ndarray
            Aggregated data with shape (t, n) where t is the final time index
            length and n is the number of sites in the current meta chunk.
        """

        arr = self._init_arr(var)

        for i in range(len(self.meta_chunk)):
            logger.debug('Kicking off site index #{}'.format(i))
            args = self._get_args(var, i)
            arr[:, i] = method(*args)

        return arr

    def _agg_var_parallel(self, var, method):
        """Aggregate one var for all sites in this chunk in parallel.

        Parameters
        ----------
        var : str
            Variable name being aggregated.
        method : function
            Aggregation method.

        Returns
        -------
        arr : np.ndarray
            Aggregated data with shape (t, n) where t is the final time index
            length and n is the number of sites in the current meta chunk.
        """

        futures = {}
        arr = self._init_arr(var)

        with ProcessPoolExecutor() as exe:
            logger.debug('Submitting futures...')
            for i in range(len(self.meta_chunk)):
                args = self._get_args(var, i)
                f = exe.submit(method, *args)
                futures[f] = i
            logger.debug('Finished submitting futures')

            for j, f in enumerate(as_completed(futures)):
                if (j + 1) % 1000 == 0:
                    logger.info('Futures completed: {} out of {}.'
                                .format(j + 1, len(futures)))
                i = futures[f]
                arr[:, i] = f.result()

        return arr

    def write_output(self, arr, var):
        """Write aggregated output data to the final output file.

        Parameters
        ----------
        arr : np.ndarray
            Aggregated data with shape (t, n) where t is the final time index
            length and n is the number of sites in the current meta chunk.
        var : str
            Variable (dataset) name to write to.
        """
        logger.debug('Writing data for "{}".'.format(var))
        with Outputs(self.fout, mode='a') as out:
            out[var] = arr

    @classmethod
    def run_chunk(cls, data, data_dir, meta_dir, i_chunk, n_chunks,
                  year=2018, parallel=True, chunk_log=False):
        """
        Parameters
        ----------
        data : dict
            Nested dictionary containing data on all NSRDB data sources
            (east, west, conus) and the final aggregated output.
        data_dir : str
            Root directory containing sub dirs with all data sources.
        meta_dir : str
            Directory containing meta and ckdtree files for each data source
            and the final aggregated output.
        i_chunk : int
            Single chunk index to process.
        n_chunks : int
            Number of chunks to process the meta data in.
        year : int
            Year being analyzed.
        parallel : bool
            Flag to use parallel compute.
        chunk_log : bool
            Flag to start a logger for this agg chunk.
        """

        if chunk_log:
            log_file = os.path.join(data_dir,
                                    data['final']['data_sub_dir'] + '/',
                                    'agg_{}.log'.format(i_chunk))
            init_logger(__name__, log_level='INFO', log_file=log_file)

        logger.info('Working on site chunk {} out of {}'
                    .format(i_chunk + 1, n_chunks))

        m = cls(data, data_dir, meta_dir, year=year, i_chunk=i_chunk,
                n_chunks=n_chunks)

        datasets = [d for d in m.dsets if 'dhi' not in d
                    and 'cld_' not in d]
        delayed_datasets = [d for d in m.dsets if 'dhi' in d
                            or 'cld_' in d]
        n_var = len(datasets) + len(delayed_datasets)
        i_var = 0
        for dsets in [datasets, delayed_datasets]:
            for var in dsets:
                method = m._get_agg_method(var)
                i_var += 1
                logger.info('Aggregating variable "{}" '
                            '({} out of {}). Using method: {}'
                            .format(var, i_var, n_var, method))
                if parallel:
                    arr = m._agg_var_parallel(var, method)
                else:
                    arr = m._agg_var_serial(var, method)

                m.write_output(arr, var)

        logger.info('NSRDB aggregation complete!')

    @classmethod
    def eagle(cls, data, data_dir, meta_dir, year, n_chunks, alloc='pxs',
              memory=90, walltime=4, feature='--qos=normal', node_name='agg',
              stdout_path=None):
        """Run NSRDB aggregation on Eagle with each agg chunk on a node.

        Parameters
        ----------
        data : dict
            Nested dictionary containing data on all NSRDB data sources
            (east, west, conus) and the final aggregated output.
        data_dir : str
            Root directory containing sub dirs with all data sources.
        meta_dir : str
            Directory containing meta and ckdtree files for each data source
            and the final aggregated output.
        year : int
            Year being analyzed.
        n_chunks : int
            Number of chunks to process the meta data in (each chunk will be
            a node).
        alloc : str
            SLURM project allocation.
        memory : int
            Node memory request in GB.
        walltime : int
            Node walltime request in hours.
        feature : str
            Additional flags for SLURM job. Format is "--qos=high"
            or "--depend=[state:job_id]". Default is None.
        node_name : str
            Name for the SLURM job.
        stdout_path : str
            Path to dump the stdout/stderr files.
        """

        if stdout_path is None:
            stdout_path = os.getcwd()

        cmd = ("python -c \'from nsrdb.aggregation.aggregation import Manager;"
               "Manager.run_chunk({})\'")

        for i_chunk in range(n_chunks):
            i_node_name = node_name + '_{}'.format(i_chunk)
            a = ('{}, "{}", "{}", i_chunk={}, n_chunks={}, year={}, '
                 'parallel=True, chunk_log=True'
                 .format(json.dumps(data), data_dir, meta_dir,
                         i_chunk, n_chunks, year))
            icmd = cmd.format(a)

            slurm = SLURM(icmd, alloc=alloc, memory=memory, walltime=walltime,
                          feature=feature, name=i_node_name,
                          stdout_path=stdout_path)

            print('\ncmd:\n{}\n'.format(icmd))

            if slurm.id:
                msg = ('Kicked off job "{}" (SLURM jobid #{}) on '
                       'Eagle.'.format(i_node_name, slurm.id))
            else:
                msg = ('Was unable to kick off job "{}". '
                       'Please see the stdout error messages'
                       .format(i_node_name))
            print(msg)


if __name__ == '__main__':
    data_dir = '/projects/pxs/processing/2018/nsrdb_output_final/'
    meta_dir = '/projects/pxs/reference_grids/'
    n_chunks = 100
    year = 2018
    Manager.eagle(NSRDB_4km_30min, data_dir, meta_dir, year, n_chunks,
                  alloc='pxs', memory=90, walltime=4, feature='--qos=high',
                  node_name='agg',
                  stdout_path=os.path.join(data_dir, 'stdout/'))
