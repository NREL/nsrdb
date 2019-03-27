# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 20:35:24 2014

@author: alopez

"""

import h5py
import numpy as np
import os
import logging
from scipy.spatial import cKDTree
import pandas as pd
import re
import psutil
import gc
import calendar
from warnings import warn

from nsrdb import __version__
from nsrdb.utilities.loggers import init_logger
from nsrdb.utilities.execution import PBS, SLURM
from nsrdb.utilities.qa_qc import plot_geo_df


logger = logging.getLogger(__name__)


def log_mem():
    """Print memory status to debug logger."""
    mem = psutil.virtual_memory()
    logger.debug('{0:.3f} GB used of {1:.3f} GB total ({2:.1f}% used) '
                 '({3:.3f} GB free) ({4:.3f} GB available).'
                 ''.format(mem.used / 1e9,
                           mem.total / 1e9,
                           100 * mem.used / mem.total,
                           mem.free / 1e9,
                           mem.available / 1e9))


class Blender:
    """East-west NSRDB data blending utility."""

    def __init__(self, source_dir, out_dir, fout, year,
                 chunk_shape=[None, 100], var=None, plot_verify=False,
                 f_loc=('/lustre/eaglefs/projects/pxs/reference_grids/'
                        'location_info.h5')):
        """Initialize a blender utility object.

        Parameters
        ----------
        source_dir : str
            Directory containing east/west source h5 files to be blended.
        out_dir : str
            Target output directory to save blended data.
        fout : str
            Target output file (without path) to save blended data.
        year : int | str
            Year to be blended. Only files with this year in the filename are
            blended.
        chunk_shape : list | tuple
            Two-entry chunk shape (y chunk, x chunk). If an entry is None,
            this value will be replaced by the corresponding dimension in the
            final dataset.
        var : str | NoneType
            Optional variable flag to only blend a certain variable. If this
            is set, only files in source_dir with var in the name will be
            blended. This is the NSRDB variable name, not the MERRA variable.
        f_loc : str
            h5 file (with path) containing reference grid location info.
        """

        log_mem()
        self.year = int(year)
        self.i_start = 0
        self.i_end = 0
        self.lat = []
        self.lon = []
        self.elev = []
        self.chunk_shape = chunk_shape
        self.source_dir = source_dir
        self.out_dir = out_dir
        self.fout = fout
        self.f_loc = f_loc
        self.irradiance_dsets = ['ghi', 'clearsky_ghi', 'dni',
                                 'clearsky_dni', 'dhi', 'clearsky_dhi']
        self.daysinyear = 366 if calendar.isleap(self.year) else 365

        self._get_files(var=var)
        self._define_indices_meta()
        self._data_shape()
        self._set_chunk_shape()

        if plot_verify:
            self.plot_verify_source()

        self._create_HDF()
        gc.collect()
        log_mem()

    def _get_files(self, var=None):
        """Get the target file list in dataframe format."""
        logger.debug('Getting files...')

        if var is not None:
            logger.debug('Only blending files containing "{}"'.format(var))

        file_list = os.listdir(self.source_dir)
        east_files = [f for f in file_list if
                      self.check_blend(f, 'east', self.year, flag=var)]
        west_files = [f for f in file_list if
                      self.check_blend(f, 'west', self.year, flag=var)]

        logger.debug('Blending the following east files:\n{}'
                     .format(east_files))
        logger.debug('Blending the following west files:\n{}'
                     .format(west_files))

        # Sort all of the HDFs so that we get them in ascending index order
        if len(east_files) > 1 and len(west_files) > 1:
            try:
                east_files = sorted(east_files,
                                    key=lambda x: int(x.split('_')[2]))
                west_files = sorted(west_files,
                                    key=lambda x: int(x.split('_')[2]))
            except Exception as e:
                warn('Could not sort east/west files by lambda: {}'.format(e))
                east_files = sorted(east_files)
                west_files = sorted(west_files)

        self._define_files_df(west_files + east_files)

    def _define_files_df(self, flist):
        # intialize the dataframe
        self.files = pd.DataFrame({'files': flist,
                                   'region': None,
                                   'year': None,
                                   })

        for index, row in self.files.iterrows():
            if 'west' in row.files:
                self.files.loc[index, 'region'] = 'west'
            elif 'east' in row.files:
                self.files.loc[index, 'region'] = 'east'

            match = re.match(r'.*([1-2][0-9]{3})', row.files)
            if match:
                self.files.loc[index, 'year'] = str(int(match.group(1)))

        logger.debug('Files df:\n{}'.format(self.files.to_string()))

    def _define_indices_meta(self):
        """Construct meta and determine indicies given overlap"""
        logger.debug('Defining Indices...')

        all_indices = []

        for _, row in self.files.iterrows():

            # get the relevant meta data from the file
            lat, lon, elev = self.get_lat_lon_elev(
                os.path.join(self.source_dir, row.files))

            if row.region == 'west' and np.any(lon > -105):
                indices = np.where(lon <= -105)[0]
                all_indices.append(indices)
            else:
                indices = np.arange(len(lon))
                all_indices.append(indices)

            self.lat.append(lat[indices])
            self.lon.append(lon[indices])
            if elev is not None:
                self.elev.append(elev[indices])

        self.lat = np.concatenate(self.lat)
        self.lon = np.concatenate(self.lon)
        if elev is not None:
            self.elev = np.concatenate(self.elev)

        self.files['indices'] = np.array(all_indices)

    def _data_shape(self):
        """Define the shape of datasets to be created in final output"""
        logger.debug('determining shape...')

        self.y = 17568 if calendar.isleap(self.year) else 17520
        self.x = len(self.lon)
        self.shape = (self.y, self.x)
        logger.debug('Output data shape is: {}'.format(self.shape))

    def _set_chunk_shape(self):
        """Set chunk shape values that are "None" based on data shape."""
        if hasattr(self, 'shape') and hasattr(self, 'chunk_shape'):
            new_shape = [100, 100]
            for i in range(2):
                if self.chunk_shape[i] is None:
                    new_shape[i] = self.shape[i]
                else:
                    new_shape[i] = self.chunk_shape[i]
            self.chunk_shape = new_shape
        if isinstance(self.chunk_shape, list):
            self.chunk_shape = tuple(self.chunk_shape)
        logger.debug('Output data chunk shape is: {}'.format(self.chunk_shape))

    def _create_HDF(self):
        """Create the output HDF with datasets and attrs"""

        logger.debug('creating output HDF...')

        self.hfile = h5py.File(os.path.join(self.out_dir, self.fout), 'w')

        # create summary stats group/datasets
        self.hfile.create_group('stats')

        fname0 = os.path.join(self.source_dir, self.files.files[0])

        datasets, dtypes = self.get_dsets(fname0)

        with h5py.File(fname0, 'r') as hdf:

            logger.debug('Found the following datasets: {}'.format(datasets))

            for i, dataset in enumerate(datasets):
                logger.debug('Creating dataset "{}" with shape {}, dtype {}, '
                             'and chunk shape {}.'
                             .format(dataset, self.shape, dtypes[i],
                                     self.chunk_shape))
                self.hfile.create_dataset(dataset, shape=self.shape,
                                          dtype=dtypes[i],
                                          chunks=self.chunk_shape)

                for stat in ['min', 'max', 'avg', 'std']:
                    self.hfile.create_dataset(
                        'stats/{d}_{s}'.format(d=dataset, s=stat),
                        shape=(self.x, ), dtype=np.float)

                for attr in hdf[dataset].attrs.keys():
                    self.hfile[dataset].attrs[attr] = hdf[dataset].attrs[attr]

        # add meta
        logger.debug('Creating meta data...')
        meta = np.dtype([('latitude', np.float), ('longitude', np.float),
                         ('elevation', np.float), ('timezone', np.int),
                         ('country', 'S30'), ('state', 'S30'),
                         ('county', 'S30'), ('urban', 'S30'),
                         ('population', np.int), ('landcover', np.int)])

        self.hfile.create_dataset('meta', shape=(self.x,), dtype=meta)

        self.hfile['meta']['latitude'] = self.lat
        self.hfile['meta']['longitude'] = self.lon

        if self.elev:
            self.hfile['meta']['elevation'] = self.elev

        self.hfile['meta'].attrs['lat_units'] = 'decimal degrees'
        self.hfile['meta'].attrs['lng_units'] = 'decimal degrees'
        self.hfile['meta'].attrs['elevation_units'] = 'meters'
        self.hfile['meta'].attrs['timezone_units'] = 'UTC Offset'

        # add time stamp
        logger.debug('Creating time_index data...')
        time_index = pd.date_range('1-1-{y}'.format(y=self.year),
                                   '1-1-{y}'.format(y=self.year + 1),
                                   freq='30Min')[:-1]
        self.hfile.create_dataset('time_index',
                                  data=time_index.values.astype('S30'),
                                  dtype='S30')

        # # add location info
        logger.debug('Creating location info...')
        with h5py.File(self.f_loc, 'r') as htz:
            target_ll = np.dstack((self.lon.ravel(), self.lat.ravel()))[0]
            # Build lat/lng pairs by unravel
            logger.debug('Extracting lat/lon data from {}'
                         .format(self.f_loc))
            ref_ll = np.dstack((np.ravel(htz['meta']['longitude'][...]),
                                np.ravel(htz['meta']['latitude'][...])))[0]

            # get the nearest neighbor indices
            index = self.get_nn_ind(ref_ll, target_ll)

            # iterate each dataset
            for key in htz.keys():
                if key != 'meta':
                    logger.debug('Processing "{}" from {}'
                                 .format(key, self.f_loc))
                    data = htz[key][...]
                    self.hfile['meta'][key] = data[index]
                    attr = '{key}_fill_value'.format(key=key)

                    self.hfile['meta'].attrs[attr] = \
                        htz[key].attrs['fill_value']

                    if key == 'landcover':
                        self.hfile['meta'].attrs['landcover_types'] = \
                            htz[key].attrs['landcover_types']

        # Add NSRDB Version
        self.hfile.attrs['version'] = __version__

    @staticmethod
    def kdtree(ref_ll, target_ll):
        """Execute a cKDTree query.

        Parameters
        ----------
        ref_ll : np.ndarray
            Reference lat/lon array.
        target_ll : np.ndarray
            Final lat/lon array.

        Returns
        -------
        index : np.array
            cKDTree index results.
        """
        # Build NN tree
        logger.debug('Building KDTree...')
        tree = cKDTree(ref_ll)
        # Get the distance and index of NN
        logger.debug('Querying KDTree...')
        _, index = tree.query(target_ll, k=1,
                              distance_upper_bound=0.05)
        return index

    def get_nn_ind(self, ref_ll, target_ll, cache=True):
        """Get nearest neighbor indices.

        Parameters
        ----------
        ref_ll : np.ndarray
            Reference lat/lon array.
        target_ll : np.ndarray
            Final lat/lon array.
        cache : bool
            Flag to save/import cached nearest neighbor indices from the
            out_dir. The NN operation is the most time intensive process so
            this can speed up the code substantially. However, if this requires
            that the cached nn results were generated with the same ref/target
            arrays.

        Returns
        -------
        index : np.array
            cKDTree index results.
        """

        if cache:
            # try to get cached kdtree results. fast for prototyping.
            cache = os.path.join(self.out_dir, 'ckdtree.csv')
            if os.path.exists(cache):
                logger.debug('Found cached nearest neighbor indices, '
                             'importing: {}'.format(cache))
                index = np.genfromtxt(cache, dtype=int, delimiter=',')
            else:
                index = self.kdtree(ref_ll, target_ll)
                logger.debug('Saving nearest neighbor indices to: {}'
                             .format(cache))
                np.savetxt(cache, index, delimiter=',')
        else:
            index = self.kdtree(ref_ll, target_ll)
        return index

    @staticmethod
    def search_hdf(hdf, var):
        """Look for a variable as a dataset or meta data column in hdf.

        Parameters
        ----------
        hdf : str
            HDF filename (with path).
        var : str
            Variable to look for as a dataset or as a column in meta.

        Returns
        -------
        arr : np.array | None
            Var array if found, None if not found.
        """

        arr = None
        with h5py.File(hdf, 'r') as hdfobj:
            if var in hdfobj:
                arr = hdfobj[var][:]
            elif 'meta' in hdfobj:
                meta = pd.DataFrame(hdfobj['meta'][...])
                if var in meta:
                    arr = meta[var].values
        return arr

    @staticmethod
    def get_lat_lon_elev(fname):
        """Get file lat/lon/elev meta data.

        Parameters
        ----------
        fname : str
            Target filename (with path) containing latitude/longitude/elevation
            as datasets or containing meta dataset with these.

        Returns
        -------
        lat / lon / elev : np.array | None
            latitude/longitude/elevation array if found. None if not found.
        """

        lat = Blender.search_hdf(fname, 'latitude')
        lon = Blender.search_hdf(fname, 'longitude')
        elev = Blender.search_hdf(fname, 'elevation')

        if elev is None:
            warn('Elevation dataset could not be found in {}'.format(fname))

        return lat, lon, elev

    @staticmethod
    def check_blend(fname, region, year, flag=None, ext='.h5'):
        """Check whether to blend a file based on various flags.

        Parameters
        ----------
        fname : str
            Target filename that may have to be blended.
        region : str
            Region being blended (east or west).
        year : int | str
            Year being blended.
        flag : str | NoneType
            Optional additional flag to look for in fname. Could be an NSRDB
            variable.
        ext : str
            Requested file extension.

        Returns
        -------
        blend : bool
            Whether or not to blend the target file.
        """

        blend = False
        if region in fname and str(year) in fname and fname.endswith(ext):
            blend = True
        if flag is not None:
            if flag not in fname:
                blend = False
        return blend

    @staticmethod
    def get_dsets(h5, ignore=('latitude', 'longitude', 'elevation', 'meta',
                              'time_index')):
        """Get a list of datasets and dtypes to operate on from the file.

        Parameters
        ----------
        h5 : str
            Target h5 file to retrieve datasets from. Must include path.
        ignore : tuple | list
            Datasets to ignore.

        Returns
        -------
        datasets : list
            List of datasets.
        dtypes : list
            List of dataset dtypes.
        """

        with h5py.File(h5, 'r') as hdf:
            datasets = [i for i in hdf if i not in ignore]
            dtypes = [hdf[i].dtype for i in hdf if i not in ignore]
        return datasets, dtypes

    def blend_file(self, source_h5, indices, out_i_start, out_i_end,
                   stats=False):
        """Blend a single source h5 file to the final output blended h5.

        Parameters
        ----------
        source_h5 : str
            Source h5 file (either east or west) found in self.source_dir to
            be blended.
        indices : np.array
            Indices in source_h5 to retrieve and pass to the final output h5.
        out_i_start : int
            INCLUSIVE starting site (column) index in the output h5file to
            write the data to.
        out_i_end : int
            EXCLUSIVE final site (column) index in the output h5file to write
            the data to.
        stats : bool
            Flag to run stats on datasets. Off by default because this can
            sometimes crash a node on memory errors.
        """

        with h5py.File(os.path.join(self.source_dir, source_h5), 'r') as hdf:
            logger.debug('Blending file: {}'.format(source_h5))
            log_mem()

            datasets, _ = self.get_dsets(
                os.path.join(self.source_dir, source_h5))

            for dataset in datasets:

                logger.debug('Performing blend on dset: {}'.format(dataset))

                data = hdf[dataset][...]
                data = data[:, indices]
                self.hfile[dataset][:, out_i_start:out_i_end] = data
                logger.debug('Blended dset: {}'.format(dataset))
                log_mem()

                if stats:
                    try:
                        self.write_stats(data, dataset, out_i_start, out_i_end)
                    except Exception as e:
                        logger.exception('Could not write stats for {}. '
                                         'Received the following error: {}'
                                         .format(dataset, e))

        logger.debug('Finished blending file: {}'.format(source_h5))

    def write_stats(self, data, dataset, out_i_start, out_i_end):
        """Calculate statistics on data and write summary to output h5.

        Parameters
        ----------
        data : np.ndarray
            Data array to calculate stats on
        dataset : str
            Dataset name.
        out_i_start : int
            INCLUSIVE starting site (column) index in the output h5file to
            write the data to.
        out_i_end : int
            EXCLUSIVE final site (column) index in the output h5file to write
            the data to.
        """
        logger.debug('Writing stats for dset "{}"...'.format(dataset))
        log_mem()

        # extract scalar but only use it to scale final stat values
        if 'psm_scale_factor' in self.hfile[dataset].attrs:
            scalar = self.hfile[dataset].attrs['psm_scale_factor']
        else:
            scalar = 1

        # create and save stats
        if dataset in self.irradiance_dsets:
            davg = np.sum(data, axis=0)
            davg /= (self.daysinyear * 1000. * 2.)
        else:
            davg = np.mean(data, axis=0)

        logger.debug('Stats: calculated mean for dset "{}".'.format(dataset))
        log_mem()

        dmin = np.min(data, axis=0)
        dmax = np.max(data, axis=0)
        dstd = np.std(data, axis=0)

        logger.debug('Stats: calculated min/max/std for "{}".'.format(dataset))
        log_mem()

        k1 = 'stats/{d}_{s}'.format(d=dataset, s='avg')
        k2 = 'stats/{d}_{s}'.format(d=dataset, s='min')
        k3 = 'stats/{d}_{s}'.format(d=dataset, s='max')
        k4 = 'stats/{d}_{s}'.format(d=dataset, s='std')

        logger.debug('Stats: writing stats for "{}".'.format(dataset))
        log_mem()

        self.hfile[k1][out_i_start:out_i_end] = davg / scalar
        self.hfile[k2][out_i_start:out_i_end] = dmin / scalar
        self.hfile[k3][out_i_start:out_i_end] = dmax / scalar
        self.hfile[k4][out_i_start:out_i_end] = dstd / scalar

    def process_all(self):
        """Process and blend all files in the file list."""

        logger.debug('Processing {} files...'.format(len(self.files)))
        log_mem()

        for _, row in self.files.iterrows():

            self.i_start = self.i_end
            self.i_end = self.i_start + len(row.indices)

            self.blend_file(row.files, row.indices, self.i_start, self.i_end)
            gc.collect()

        logger.debug('Blender.process_all() complete. Closing {}'
                     .format(self.fout))
        self.hfile.close()
        logger.info('Blending operation finished.')

    def plot_verify_source(self):
        """Plot source data to verify mapping."""
        for _, row in self.files.iterrows():
            fname = os.path.join(self.source_dir, row.files)
            datasets, _ = self.get_dsets(fname)
            for dset in datasets:
                with h5py.File(fname, 'r') as hdf:
                    data = (hdf[dset][0, :] /
                            hdf[dset].attrs['psm_scale_factor'])
                    lat, lon, _ = self.get_lat_lon_elev(fname)
                    df = pd.DataFrame({'latitude': lat, 'longitude': lon,
                                       dset: data})
                    plot_name = '{}_{}'.format(row.files.replace('.h5', ''),
                                               dset)
                    plot_geo_df(df, plot_name, self.out_dir)

    @staticmethod
    def summarize(out_dir, fout, dset, save_meta=False, plot=True):
        """Log a summary of the output file and extract meta and/or a plot.

        Parameters
        ----------
        out_dir : str
            Target location of the file to summarize.
        fout : str
            Target h5 file to summarize (must be in out_dir).
        dset : str
            Dataset name of interest. (meta will be extracted in addition).
        save_meta : bool
            Flag to export the meta data to a csv in out_dir.
        plot : bool
            Flag to plot 1st timestep of the dset on a lat/lon plot.
        """
        logger.info('Summarizing "{}" located in {}'.format(fout, out_dir))
        try:
            with h5py.File(os.path.join(out_dir, fout), 'r') as f:
                logger.info('"{}" contains the following datasets: {}'
                            .format(fout, list(f.keys())))
                for d in list(f.keys()):
                    if d not in ['stats']:
                        logger.info('Dataset "{}" has dtype: {}'
                                    .format(d, f[d].dtype))

                meta = pd.DataFrame(f['meta'][...])
                logger.info('"{}" meta data head/tail are as follows:\n{}\n{}'
                            .format(fout, meta.head(), meta.tail()))
                logger.info('"{}" meta data shape is: {}'
                            .format(fout, meta.shape))
                base_name = fout.replace('.h5', '')

                if save_meta:
                    meta.to_csv(
                        os.path.join(out_dir, base_name + '_meta.csv'))

                logger.info('"{}" dataset "{}" has shape {} and chunk shape {}'
                            .format(fout, dset, f[dset].shape, f[dset].chunks))

                if plot:
                    df = pd.DataFrame({'latitude': meta['latitude'],
                                       'longitude': meta['longitude'],
                                       dset: f[dset][0, :]})
                    plot_geo_df(df, 'blended_{}'.format(base_name), out_dir)
        except Exception as e:
            logger.exception('Could not summarize {}. Received the following '
                             'exception: \n{}'
                             .format(os.path.join(out_dir, fout), e))

    @classmethod
    def blend_var(cls, var, year, out_dir, fout, source_dir,
                  f_loc=('/lustre/eaglefs/projects/pxs/reference_grids/'
                         'location_info.h5'),
                  chunk_shape=[None, 100], plot_verify=True):
        """Blend a single variable.

        Parameters
        ----------
        var : str
            Target variable to blend. This string will be searched for in the
            file names in the source_dir. This is the NSRDB variable name, not
            the MERRA variable.
        year : int | str
            Year to be blended. Only files with this year in the filename are
            blended.
        out_dir : str
            Target output directory to save blended data.
        fout : str
            Target output file (without path) to save blended data.
        source_dir : str
            Directory containing east/west source h5 files to be blended.
        chunk_shape : list | tuple
            Two-entry chunk shape (y chunk, x chunk). If an entry is None,
            this value will be replaced by the corresponding dimension in the
            final dataset.
        plot_verify : bool
            Flag for whether to plot geo maps of the source data and blended
            output.
        """

        init_logger(__name__, log_level='DEBUG', log_file=None)

        blend = Blender(source_dir, out_dir, fout, year, var=var,
                        chunk_shape=chunk_shape, plot_verify=plot_verify,
                        f_loc=f_loc)
        blend.process_all()
        blend.summarize(out_dir, fout, var, save_meta=True, plot=plot_verify)

    @classmethod
    def peregrine_blend(cls, var, year_range, out_dir, source_dir):
        """Blend a single variable over a range of years on Peregrine bigmem.

        Parameters
        ----------
        var : str
            Target variable to blend. This string will be searched for in the
            file names in the source_dir. This is the NSRDB variable name, not
            the MERRA variable.
        year_range : iterable
            Year range to be blended. Each year will be a seperate job on
            peregrine in the BIGMEM queue.
        out_dir : str
            Target output directory to save blended data.
        source_dir : str
            Directory containing east/west source h5 files to be blended.
        """

        for year in year_range:
            node_name = '{}_{}'.format(str(year)[-2:], var)
            fout = '{}_{}.h5'.format(str(year), var)

            cmd = ('python -c '
                   '\'from nsrdb.blend_e_w import Blender; '
                   'Blender.blend_var("{var}", {year}, fout="{fout}", '
                   'out_dir="{out_dir}", source_dir="{source_dir}", '
                   'f_loc="/projects/PXS/reference_grids/location_info.h5")\''
                   )

            cmd = cmd.format(var=var, year=year, fout=fout, out_dir=out_dir,
                             source_dir=source_dir)

            pbs = PBS(cmd, alloc='pxs', queue='bigmem', name=node_name,
                      stdout_path=os.path.join(out_dir, 'stdout/'),
                      feature=None)

            print('\ncmd:\n{}\n'.format(cmd))

            if pbs.id:
                msg = ('Kicked off job "{}" (PBS jobid #{}) on '
                       'Peregrine.'.format(fout.replace('.h5', ''), pbs.id))
            else:
                msg = ('Was unable to kick off job "{}". '
                       'Please see the stdout error messages'
                       .format(fout.replace('.h5', '')))
            print(msg)

    @classmethod
    def eagle_blend(cls, var, year_range, out_dir, source_dir):
        """Blend a single variable over a range of years on EAGLE.

        Parameters
        ----------
        var : str
            Target variable to blend. This string will be searched for in the
            file names in the source_dir. This is the NSRDB variable name, not
            the MERRA variable.
        year_range : iterable
            Year range to be blended. Each year will be a seperate job on
            Eagle.
        out_dir : str
            Target output directory to save blended data.
        source_dir : str
            Directory containing east/west source h5 files to be blended.
        """

        for year in year_range:
            node_name = '{}_{}'.format(str(year)[-2:], var)
            fout = '{}_{}.h5'.format(str(year), var)

            cmd = ('python -c '
                   '\'from nsrdb.blend_e_w import Blender; '
                   'Blender.blend_var("{var}", {year}, fout="{fout}", '
                   'out_dir="{out_dir}", source_dir="{source_dir}", '
                   'f_loc="/lustre/eaglefs/projects/pxs/reference_grids/'
                   'location_info.h5")\''
                   )

            cmd = cmd.format(var=var, year=year, fout=fout, out_dir=out_dir,
                             source_dir=source_dir)

            slurm = SLURM(cmd, alloc='pxs', memory=768, walltime=5,
                          name=node_name,
                          stdout_path=os.path.join(out_dir, 'stdout/'))

            print('\ncmd:\n{}\n'.format(cmd))

            if slurm.id:
                msg = ('Kicked off job "{}" (SLURM jobid #{}) on '
                       'Eagle.'.format(fout.replace('.h5', ''), slurm.id))
            else:
                msg = ('Was unable to kick off job "{}". '
                       'Please see the stdout error messages'
                       .format(fout.replace('.h5', '')))
            print(msg)
