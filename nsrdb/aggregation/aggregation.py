# -*- coding: utf-8 -*-
"""NSRDB aggregation methods
 - 2km 5min CONUS -> 4km 30min NSRDB PSM v3 Meta
 - 2km 15min East -> 4km 30min NSRDB PSM v3 Meta
 - 4km 30min West -> 4km 30min NSRDB PSM v3 Meta
"""
import os
import pickle
import numpy as np
import pandas as pd
from warnings import warn

from nsrdb.file_handlers.resource import Resource
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.utilities.plots import Spatial


def meta_source_2018(fpath_4km):
    """Make 2018 4km meta data with data source column (west/east/conus).

    WARNING: This is a script very specific to the 2018 GOES data arrangement,
    with the 4km 30min GOES West satellite, the 2km 15min GOES East satellite,
    and the 2km 5min CONUS data from GOES East. This only works with the psm v3
    4km meta data (accessed on 8/28/2019).

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
    # boundary, above which no cloud properties are returned for the East data.
    lat_boundary = 0.6 * (meta.longitude.values + 125) + 42.7
    angle_mask = ((meta.latitude > lat_boundary)
                  & (meta.source != 'west')
                  & (meta.longitude < -104.5))
    meta.loc[angle_mask, 'source'] = 'west'

    return meta


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

    meta = meta_source_2018(fpath_4km)
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
            Filepath to h5 file containing var data.
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
        self.nn = nn
        self.w = w
        self.final_ti = final_ti

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

        with Resource(self.data_fpath) as res:
            _data = res[self.var, :, self.nn]
        return _data

    def spatial_avg(self):
        """Average the source data across the spatial extent.

        Returns
        -------
        data : np.ndarray
            Unscaled float data array with shape (ti, 1) where ti is the
            native time index length the data was averaged accross all
            nn neighbors.
        """

        return self.data.mean(axis=1)

    @staticmethod
    def time_avg(inp, window=7):
        """Calculate the rolling time average for an input array or df.

        Parameters
        ----------
        inp : np.ndarray | pd.DataFrame
            Input array/df with data to average.
        window : int
            Window over which to calculate the average.

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

        out = inp.rolling(window, center=True, min_periods=1).mean()

        if array:
            out = out.values

        return out


class Manager:
    """Framework for aggregation to a final NSRDB spatiotemporal resolution."""

    def __init__(self, data, data_dir, meta_dir):
        """
        Parameters
        ----------
        data : dict
        data_dir : str
        meta_dir : str
        """
        self.data = data
        self.data_dir = data_dir
        self.meta_dir = meta_dir

        meta_path = os.path.join(meta_dir, self.data['final']['meta_file'])
        self.meta = meta_source_2018(meta_path)

        self.parse_data()
        self.preflight()
        self.run_nn()

    def parse_data(self):
        """Parse the data input for several useful attributes."""
        self.final_sres = self.data['final']['spatial']
        self.final_tres = self.data['final']['temporal']
        self.data_sources = self.meta.source.unique()

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

    def _get_fpath(var, data_dir, data_sub_dir):
        """Get the h5 filepath in data_dir/data_sub_dir/ containing var.

        Parameters
        ----------
        var : str
            Variable name (h5 dataset) being searched for.
        data_dir : str
            Root data directory.
        data_sub_dir : str
            Sub directory in data_dir containing h5 files.

        Returns
        -------
        fpath : str
            File path to h5 file in data_sub_dir containing var dataset.
        """

        if not data_sub_dir.endswith('/'):
            data_sub_dir += '/'

        for fn in os.path.join(data_dir, data_sub_dir):
            if fn.endswith('.h5'):
                fpath = os.path.join(data_dir, data_sub_dir, fn)
                with Outputs(fpath) as out:
                    if var in out.dsets:
                        break
        return fpath

    def run_nn(self):
        """Run nearest neighbor for all data sources against the final meta."""
        for source in self.data_sources:

            k = self._get_spatial_k(self.data[source]['spatial'],
                                    self.final_sres)

            d, i = self.knn(self.meta, self.data[source]['tree_file'], k=k)

            if d.max() > 1:
                warn('Max distance between "final" and "{}" is {}'
                     .format(source, d.max()))

            self.data[source]['nn'] = i

    def add_temporal(self):
        """Get the temporal window sizes for all data sources."""
        for source in self.data_sources:
            w = self._get_temporal_w(self.data[source]['temporal'],
                                     self.final_tres)
            self.data[source]['window'] = w

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


if __name__ == '__main__':
    data_dir = '/projects/pxs/processing/2018/nsrdb_output_final/'
    meta_dir = '/projects/pxs/reference_grids/'
    data = {'east': {'data_sub_dir': 'east',
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
                      'tree_file': 'kdtree_nsrdb_meta_4km.pkl',
                      'meta_file': 'nsrdb_meta_4km.csv',
                      'spatial': '4km',
                      'temporal': '30min'},
            }
