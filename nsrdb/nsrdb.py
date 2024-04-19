# -*- coding: utf-8 -*-
"""Entry point for NSRDB data pipeline execution.


Created on Thu Apr 25 15:47:53 2019

@author: gbuster
"""

import calendar
import copy
import datetime
import json
import logging
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import psutil
from rex import MultiFileResource, init_logger
from rex.utilities.loggers import create_dirs

from nsrdb import CONFIGDIR, __version__
from nsrdb.aggregation.aggregation import Manager
from nsrdb.all_sky.all_sky import (
    ALL_SKY_ARGS,
    all_sky,
    all_sky_h5,
    all_sky_h5_parallel,
)
from nsrdb.data_model import DataModel, VarFactory
from nsrdb.file_handlers.collection import Collector
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.gap_fill.cloud_fill import CloudGapFill
from nsrdb.pipeline import Status
from nsrdb.utilities.file_utils import clean_meta, pd_date_range, ts_freq_check

logger = logging.getLogger(__name__)


class NSRDB:
    """Entry point for NSRDB data pipeline execution."""

    OUTS = {'nsrdb_ancillary_a_{y}.h5': ('alpha',
                                         'aod',
                                         'asymmetry',
                                         'ssa'),
            'nsrdb_ancillary_b_{y}.h5': ('ozone',
                                         'solar_zenith_angle',
                                         'surface_albedo',
                                         'total_precipitable_water'),
            'nsrdb_clearsky_{y}.h5': ('clearsky_dhi',
                                      'clearsky_dni',
                                      'clearsky_ghi'),
            'nsrdb_clouds_{y}.h5': ('cloud_type',
                                    'cld_opd_dcomp',
                                    'cld_reff_dcomp',
                                    'cld_press_acha',
                                    'cloud_fill_flag'),
            'nsrdb_csp_{y}.h5': ('dew_point',
                                 'relative_humidity',
                                 'surface_pressure'),
            'nsrdb_irradiance_{y}.h5': ('dhi',
                                        'dni',
                                        'ghi',
                                        'fill_flag'),
            'nsrdb_pv_{y}.h5': ('air_temperature',
                                'wind_direction',
                                'wind_speed')}

    def __init__(self, out_dir, year, grid, freq='5min', var_meta=None,
                 make_out_dirs=True):
        """
        Parameters
        ----------
        out_dir : str
            Project directory.
        year : int | str
            Processing year.
        grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | None
            File path to NSRDB variables meta data. None will use the default
            file from the github repo.
        make_out_dirs : bool
            Flag to make output directories for logs, daily, collect, and final
        """

        self._out_dir = out_dir
        self._log_dir = os.path.join(out_dir, 'logs/')
        self._daily_dir = os.path.join(out_dir, 'daily/')
        self._collect_dir = os.path.join(out_dir, 'collect/')
        self._final_dir = os.path.join(out_dir, 'final/')
        self._year = int(year)
        self._grid = grid
        self._freq = freq
        self._var_meta = var_meta
        self._ti = None

        ts_freq_check(freq)

        if make_out_dirs:
            self.make_out_dirs()

    @staticmethod
    def collect_blended(kwargs):
        """Collect blended data into single file

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying
            the case for blend collection
        """

        default_kwargs = {
            "basename": "nsrdb",
            "metadir": "/projects/pxs/reference_grids",
            "spatial": "4km",
            "outdir": "./",
            "freq": "30min",
            "extent": "full"
        }

        user_input = copy.deepcopy(default_kwargs)
        user_input.update(kwargs)

        meta_file = f'nsrdb_meta_{user_input["spatial"]}.csv'
        meta_file = os.path.join(user_input['metadir'], meta_file)
        collect_dir = f'nsrdb_{user_input["year"]}'
        collect_dir += f'_{user_input["extent"]}_blend'
        collect_tag = f'{user_input["basename"]}_'
        collect_tag += f'{user_input["extent"]}_{user_input["year"]}_'
        fout = os.path.join(
            f'{user_input["outdir"]}',
            f'{user_input["basename"]}_{user_input["year"]}.h5')

        user_input['log_file'] = f'{user_input["basename"]}_'
        user_input['log_file'] += f'{user_input["year"]}_collect_blend.log'

        log_file = os.path.join(user_input['outdir'], user_input['log_file'])
        logger = init_logger(
            __name__, log_file=log_file, log_level='DEBUG')
        logger.info('Running collect_blended with '
                    f'meta_file={meta_file}, collect_dir={collect_dir}, '
                    f'collect_tag={collect_tag}, fout={fout}')

        meta_file = pd.read_csv(meta_file, index_col=0)

        fns = os.listdir(collect_dir)
        flist = [fn for fn in fns if fn.endswith('.h5')
                 and collect_tag in fn
                 and os.path.join(collect_dir, fn) != fout]
        flist = sorted(
            flist, key=lambda x: int(x.replace('.h5', '').split('_')[-1]))

        temp = Manager.get_dset_attrs(collect_dir)
        dsets_all, attrs, chunks, dtypes, ti = temp
        Outputs.init_h5(fout, dsets_all, attrs, chunks, dtypes, ti, meta_file)
        _, _, shape, _ = Collector._get_collection_attrs(
            [flist[0]], collect_dir, dsets_all[0])

        for fname in flist:
            fpath = os.path.join(collect_dir, fname)
            f = Outputs(fpath, unscale=False, mode='r')
            dsets = [d for d in f.dsets if d in dsets_all]

            logger.info(f'Collecting {dsets} from {fname}')
            for dset in dsets:

                attrs, _, final_dtype = VarFactory.get_dset_attrs(dset)

                mem = psutil.virtual_memory()
                logger.debug(
                    'Initializing output dataset "{}" in-memory with shape '
                    '{} and dtype {}. Current memory usage is '
                    '{:.3f} GB out of {:.3f} GB total.'
                    .format(dset, shape, final_dtype, mem.used / 1e9,
                            mem.total / 1e9))

                logger.info(f'Writing {dset} to {fout} from {fpath}')

                Collector._ensure_dset_in_output(fout, dset)
                with Outputs(fout, mode='a') as f_combined:
                    f_combined[dset, :, :] = f[dset][...]
                logger.debug(
                    'Finished writing "{}" to: {}'
                    .format(dset, os.path.basename(fout)))
        logger.info(f'Finished blend collection: {fout}')

    @staticmethod
    def collect_aggregation(kwargs):
        """Collect aggregation chunks

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying
            the case for aggregation collection
        """
        default_kwargs = {
            "basename": "nsrdb",
            "metadir": "/projects/pxs/reference_grids",
            "final_spatial": "4km",
            "final_freq": "30min",
            "outdir": "./",
        }

        user_input = copy.deepcopy(default_kwargs)
        user_input.update(kwargs)

        meta_file = f'nsrdb_meta_{user_input["final_spatial"]}.csv'
        meta_file = os.path.join(user_input['metadir'], meta_file)
        collect_dir = f'nsrdb_{user_input["final_spatial"]}'
        collect_dir += f'_{user_input["final_freq"]}'
        collect_tag = f'{user_input["basename"]}_'
        fout = os.path.join(
            f'{user_input["outdir"]}',
            f'{user_input["basename"]}_{user_input["year"]}.h5')

        user_input['log_file'] = f'{user_input["basename"]}_'
        user_input['log_file'] += f'{user_input["year"]}_collect_agg.log'

        log_file = os.path.join(user_input['outdir'], user_input['log_file'])
        logger = init_logger(__name__, log_file=log_file, log_level='DEBUG')
        logger.info('Running collect_aggregation with '
                    f'meta_file={meta_file}, collect_dir={collect_dir}, '
                    f'collect_tag={collect_tag}, fout={fout}')

        Manager.collect(
            meta_file, collect_dir, collect_tag, fout, max_workers=1)

        logger.info(f'Finished aggregation collection: {fout}')

    @staticmethod
    def aggregate_files(kwargs):
        """Aggregate conus and full disk blends

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the
            case for which to aggregate files
        """
        default_kwargs = {
            "basename": "nsrdb",
            "metadir": "/projects/pxs/reference_grids",
            "full_spatial": "2km",
            "conus_spatial": "2km",
            "final_spatial": "4km",
            "outdir": "./",
            "full_freq": "10min",
            "conus_freq": "5min",
            "final_freq": "30min",
            "n_chunks": 32,
            "alloc": "pxs",
            "memory": 90,
            "walltime": 40,
            "source_priority": ['conus', 'full_disk']
        }
        user_input = copy.deepcopy(default_kwargs)
        user_input.update(kwargs)

        full_sub_dir = f'{user_input["basename"]}_{user_input["year"]}'
        full_sub_dir += '_full_blend'
        conus_sub_dir = f'{user_input["basename"]}_{user_input["year"]}'
        conus_sub_dir += '_conus_blend'
        final_sub_dir = f'nsrdb_{user_input["final_spatial"]}'
        final_sub_dir += f'_{user_input["final_freq"]}'

        meta_file = 'nsrdb_meta_{res}.csv'
        tree_file = 'kdtree_nsrdb_meta_{res}.pkl'

        conus_meta_file = f'nsrdb_meta_{user_input["conus_spatial"]}_conus.csv'
        conus_tree_file = 'kdtree_nsrdb_meta_'
        conus_tree_file += f'{user_input["conus_spatial"]}_conus.pkl'

        full_meta_file = f'nsrdb_meta_{user_input["full_spatial"]}_full.csv'
        full_tree_file = 'kdtree_nsrdb_meta_'
        full_tree_file += f'{user_input["full_spatial"]}_full.pkl'

        source_priority = user_input['source_priority']

        NSRDB = {
            'full_disk':
            {'data_sub_dir': full_sub_dir,
             'tree_file': full_tree_file,
             'meta_file': full_meta_file,
             'spatial': f'{user_input["full_spatial"]}',
             'temporal': f'{user_input["full_freq"]}'},
            'conus':
            {'data_sub_dir': conus_sub_dir,
             'tree_file': conus_tree_file,
             'meta_file': conus_meta_file,
             'spatial': f'{user_input["conus_spatial"]}',
             'temporal': f'{user_input["conus_freq"]}'},
            'final':
            {'data_sub_dir': final_sub_dir,
             'fout': 'nsrdb.h5',
             'tree_file': tree_file.format(res=user_input["final_spatial"]),
             'meta_file': meta_file.format(res=user_input["final_spatial"]),
             'spatial': f'{user_input["final_spatial"]}',
             'temporal': f'{user_input["final_freq"]}',
             'source_priority': source_priority}}

        run_name = f'{user_input["basename"]}_{user_input["year"]}_agg'
        Manager.hpc(NSRDB, user_input['outdir'], user_input['metadir'],
                    user_input['year'], user_input['n_chunks'],
                    alloc=user_input['alloc'], memory=user_input['memory'],
                    walltime=user_input['walltime'], feature='--qos=normal',
                    node_name=run_name, stdout_path=os.path.join(
                        user_input['outdir'], f'{final_sub_dir}/stdout/'))

    @staticmethod
    def blend_files(kwargs):
        """Blend all data files

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the
            case for which to blend data files
        """
        default_kwargs = {
            "file_tag": "all",
            "basename": "nsrdb",
            "metadir": "/projects/pxs/reference_grids",
            "spatial": "2km",
            "extent": "conus",
            "outdir": "./",
            "alloc": "pxs",
            "walltime": 48,
            "chunk_size": 100000,
            "memory": 83,
            "meta_file": None,
            "east_dir": None,
            "west_dir": None
        }
        user_input = copy.deepcopy(default_kwargs)
        user_input.update(kwargs)

        if user_input['year'] < 2018:
            user_input['extent'] = 'full'
            user_input['spatial'] = '4km'

        map_col_map = {'full': 'gid_full', 'conus': 'gid_full_conus'}
        map_col = (user_input.get('map_col', None)
                   or map_col_map[user_input['extent']])

        meta_lon_map = {'full': -105, 'conus': -113}
        meta_lon = (user_input.get('lon_seam', None)
                    or meta_lon_map[user_input['extent']])

        if user_input['meta_file'] is None:
            meta_file = f'nsrdb_meta_{user_input["spatial"]}'

            if user_input['year'] > 2017:
                meta_file += f'_{user_input["extent"]}'

            meta_file += '.csv'
            user_input['meta_file'] = os.path.join(
                user_input['metadir'], meta_file)

        src_dir = f"{user_input['basename']}"
        src_dir += "_{satellite}"
        src_dir += f"_{user_input['extent']}_{user_input['year']}"
        src_dir += f"_{user_input['spatial']}/final"
        src_dir = os.path.join(user_input['outdir'], src_dir)

        if user_input['east_dir'] is None:
            user_input['east_dir'] = src_dir.format(satellite="east")
        if user_input['west_dir'] is None:
            user_input['west_dir'] = src_dir.format(satellite="west")

        west_dir = user_input['west_dir']
        east_dir = user_input['east_dir']

        user_input['name'] = f'{user_input["basename"]}_{user_input["year"]}'
        user_input['name'] += f'_{user_input["extent"]}_blend'
        name = user_input['name']
        out_dir = user_input['outdir'] = os.path.join(
            user_input['outdir'], name)
        log_dir = os.path.join(out_dir, 'logs/')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger = init_logger('nsrdb.cli', stream=True)
        logger.info(f'Blending NSRDB data files with {user_input}')

        all_tags = ['ancillary_a', 'ancillary_b', 'clearsky',
                    'clouds', 'csp', 'irradiance', 'pv']

        cmd = f'python -m nsrdb.blend.cli -n {name}'
        cmd += '_{tag}'
        cmd += f' -m {user_input["meta_file"]} -od {out_dir}'
        cmd += f' -ed {east_dir} -wd {west_dir}'
        cmd += ' -t "{tag}"'
        cmd += f' -mc {map_col} -ls {meta_lon}'
        cmd += f' -cs {user_input["chunk_size"]}'
        cmd += f' -ld "{log_dir}"'
        cmd += f' slurm -a {user_input["alloc"]}'
        cmd += f' -wt {user_input["walltime"]}'
        cmd += f' -mem {user_input["memory"]}'
        cmd += f' -sout "{out_dir}/stdout"'
        cmd += ' -l "--qos=normal"'

        if user_input['file_tag'] == 'all':
            for tag in all_tags:
                logger.debug(f'Running command: {cmd.format(tag=tag)}')
                os.system(cmd.format(tag=tag))
        else:
            tag = user_input['file_tag']
            logger.debug(f'Running command: {cmd.format(tag=tag)}')
            os.system(cmd.format(tag=tag))

    @staticmethod
    def create_configs_all_domains(kwargs):
        """Modify config files for all domains with
        specified parameters

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameters
            including year, basename,
            satellite, extent, freq,
            spatial, meta_file, doy_range
        """
        if kwargs['year'] < 2018:
            kwargs.update({'spatial': '4km', 'extent': 'full', 'freq': '30min',
                           'satellite': 'east'})
            NSRDB.create_config_files(kwargs)
            kwargs.update({'satellite': 'west'})
            NSRDB.create_config_files(kwargs)
        elif kwargs['year'] == 2018:
            kwargs.update({'spatial': '2km', 'extent': 'full', 'freq': '10min',
                           'satellite': 'east'})
            NSRDB.create_config_files(kwargs)
            kwargs.update({'extent': 'conus', 'freq': '5min'})
            NSRDB.create_config_files(kwargs)
            kwargs.update({'spatial': '4km', 'extent': 'full', 'freq': '30min',
                           'satellite': 'west'})
            NSRDB.create_config_files(kwargs)
        else:
            kwargs.update({'spatial': '2km', 'extent': 'full', 'freq': '10min',
                           'satellite': 'east'})
            NSRDB.create_config_files(kwargs)
            kwargs.update({'satellite': 'west'})
            NSRDB.create_config_files(kwargs)
            kwargs.update({'extent': 'conus', 'freq': '5min',
                           'satellite': 'east'})
            NSRDB.create_config_files(kwargs)
            kwargs.update({'satellite': 'west'})
            NSRDB.create_config_files(kwargs)

    @staticmethod
    def create_config_files(kwargs):
        """Modify config files with
        specified parameters

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameters
            including year, basename,
            satellite, extent, freq,
            spatial, meta_file, doy_range
        """

        default_kwargs = {
            "basename": "nsrdb",
            "freq": "5min",
            "spatial": "4km",
            "satellite": "east",
            "extent": "conus",
            "outdir": "./",
            "meta_file": None,
            "doy_range": None
        }
        user_input = copy.deepcopy(default_kwargs)
        user_input.update(kwargs)

        extent_tag_map = {'full': 'RadF', 'conus': 'RadC'}
        meta_lon_map = {'full': -105, 'conus': -113}
        user_input['extent_tag'] = extent_tag_map[user_input['extent']]
        meta_lon = meta_lon_map[user_input['extent']]

        if user_input['meta_file'] is None:
            meta_file = f'nsrdb_meta_{user_input["spatial"]}'

            if user_input['year'] > 2017:
                meta_file += f'_{user_input["extent"]}'

            meta_file += f'_{user_input["satellite"]}_{meta_lon}.csv'
            user_input['meta_file'] = meta_file

        if user_input['doy_range'] is None:
            if calendar.isleap(user_input["year"]):
                user_input['doy_range'] = [1, 367]
            else:
                user_input['doy_range'] = [1, 366]

        user_input['start_doy'] = user_input['doy_range'][0]
        user_input['end_doy'] = user_input['doy_range'][1]

        PRE2018_CONFIG_TEMPLATE = os.path.join(
            CONFIGDIR, 'templates/config_nsrdb_pre2018.json')
        POST2017_CONFIG_TEMPLATE = os.path.join(
            CONFIGDIR, 'templates/config_nsrdb_post2017.json')
        PIPELINE_CONFIG_TEMPLATE = os.path.join(
            CONFIGDIR, 'templates/config_pipeline.json')

        run_name = f"{user_input['basename']}_{user_input['satellite']}"
        run_name += f"_{user_input['extent']}_{user_input['year']}"
        run_name += f"_{user_input['spatial']}"

        user_input['outdir'] = os.path.join(user_input['outdir'], run_name)

        logger = init_logger('nsrdb.cli', stream=True)
        logger.info(f'Creating NSRDB config files with {user_input}')

        if int(user_input['year']) < 2018:
            with open(PRE2018_CONFIG_TEMPLATE, encoding='utf-8') as s:
                s = s.read()
        else:
            with open(POST2017_CONFIG_TEMPLATE, encoding='utf-8') as s:
                s = s.read()

        for k, v in user_input.items():
            if isinstance(v, int):
                s = s.replace(f'"%{k}%"', str(v))
            s = s.replace(f'%{k}%', str(v))

        if not os.path.exists(user_input['outdir']):
            os.makedirs(user_input['outdir'])

        outfile = os.path.join(user_input['outdir'], 'config_nsrdb.json')
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(s)

        logger.info(f'Created file: {outfile}')

        with open(PIPELINE_CONFIG_TEMPLATE, encoding='utf-8') as s:
            s = s.read()

        for k, v in user_input.items():
            if isinstance(v, int):
                s = s.replace(f'"%{k}%"', str(v))
            s = s.replace(f'%{k}%', str(v))

        outfile = os.path.join(user_input['outdir'], 'config_pipeline.json')
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(s)

        logger.info(f'Created file: {outfile}')

        outfile = os.path.join(user_input['outdir'], 'run.sh')
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write('python -m nsrdb.cli pipeline -c config_pipeline.json')

        logger.info(f'Created file: {outfile}')

    def make_out_dirs(self):
        """Ensure that all output directories exist"""
        all_dirs = [self._out_dir, self._log_dir, self._daily_dir,
                    self._collect_dir, self._final_dir]
        for d in all_dirs:
            create_dirs(d)

    @property
    def time_index_year(self):
        """Get the NSRDB full-year time index.

        Returns
        -------
        nsrdb_ti : pd.DatetimeIndex
            Pandas datetime index for the current year at the NSRDB resolution.
        """
        if self._ti is None:
            self._ti = pd.date_range('1-1-{y}'.format(y=self._year),
                                     '1-1-{y}'.format(y=self._year + 1),
                                     freq=self._freq)[:-1]
        return self._ti

    @property
    def meta(self):
        """Get the NSRDB meta dataframe from the grid file.

        Returns
        -------
        meta : pd.DataFrame
            DataFrame of meta data from grid file csv.
            The first column must be the NSRDB site gid's.
        """

        if isinstance(self._grid, str):
            self._grid = clean_meta(pd.read_csv(self._grid, index_col=0))
        return self._grid

    @staticmethod
    def _log_version():
        """Check NSRDB and python version and 64-bit and print to logger."""

        logger.info('Running NSRDB version: {}'.format(__version__))
        logger.info('Running python version: {}'.format(sys.version_info))

        is_64bits = sys.maxsize > 2 ** 32
        if is_64bits:
            logger.info('Running on 64-bit python, sys.maxsize: {}'
                        .format(sys.maxsize))
        else:
            logger.warning('Running 32-bit python, sys.maxsize: {}'
                           .format(sys.maxsize))

    def _exe_daily_data_model(self, month, day, dist_lim=1.0, var_list=None,
                              factory_kwargs=None, fpath_out=None,
                              max_workers=None, max_workers_regrid=None,
                              mlclouds=False):
        """Execute the data model for a single day.

        Parameters
        ----------
        month : int | str
            Month to run data model for.
        day : int | str
            Day to run data model for.
        dist_lim : float
            Return only neighbors within this distance during cloud regrid.
            The distance is in decimal degrees (more efficient than real
            distance). NSRDB sites further than this value from GOES data
            pixels will be warned and given missing cloud types and properties
            resulting in a full clearsky timeseries.
        var_list : list | tuple | None
            Variables to process with the data model. None will default to all
            NSRDB variables.
        factory_kwargs : dict | None
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs
        fpath_out : str | None
            File path to dump results. If no file path is given, results will
            be returned as an object.
        max_workers : int | None
            Number of workers to run in parallel. 1 runs serial,
            None will use all available workers.
        max_workers_regrid : None | int
            Max parallel workers allowed for cloud regrid processing. None uses
            all available workers. 1 runs regrid in serial.
        mlclouds : bool
            Flag to add extra variables to the variable processing list of
            mlclouds gap fill is expected to be run as the next pipeline step.

        Returns
        -------
        data_model : nsrdb.data_model.DataModel
            Daily data model object.
        """

        date = datetime.date(year=self._year, month=int(month), day=int(day))

        if var_list is None and mlclouds:
            var_list = DataModel.ALL_VARS_ML
        if var_list is None and not mlclouds:
            var_list = DataModel.ALL_VARS

        logger.info('Starting daily data model execution for {}-{}-{}'
                    .format(month, day, self._year))

        # run data model
        data_model = DataModel.run_multiple(
            var_list, date, self._grid,
            nsrdb_freq=self._freq,
            dist_lim=dist_lim,
            var_meta=self._var_meta,
            max_workers=max_workers,
            max_workers_regrid=max_workers_regrid,
            return_obj=True,
            fpath_out=fpath_out,
            factory_kwargs=factory_kwargs)

        logger.info('Finished daily data model execution for {}-{}-{}'
                    .format(month, day, self._year))

        return data_model

    def _exe_fout(self, data_model):
        """Send the single-day data model results to output files.

        Parameters
        ----------
        data_model : nsrdb.data_model.DataModel
            Daily data model object.
        """

        logger.info('Starting file export of daily data model results to: {}'
                    .format(self._out_dir))

        # output handling for each entry in data model
        for var, arr in data_model._processed.items():
            if var not in ['time_index', 'meta']:

                fpath_out = self._get_daily_fpath_out(data_model.date)
                fpath_out = fpath_out.format(var=var, i=self.meta.index[0])

                logger.debug('\tWriting file: {}'
                             .format(os.path.basename(fpath_out)))

                # make file for each var
                with Outputs(fpath_out, mode='w') as fout:
                    fout.time_index = data_model.nsrdb_ti
                    fout.meta = data_model.nsrdb_grid

                    var_obj = VarFactory.get_base_handler(
                        var, var_meta=self._var_meta, date=data_model.date)
                    attrs = var_obj.attrs

                    fout._add_dset(dset_name=var, data=arr,
                                   dtype=var_obj.final_dtype,
                                   chunks=var_obj.chunks, attrs=attrs)

        logger.info('Finished file export of daily data model results to: {}'
                    .format(self._out_dir))

    def _get_daily_fpath_out(self, date):
        """Get the data model file output path based on a date.
        Will have {var} and {i}.

        Parameters
        ----------
        date : datetime.date
            Single day for the output file.

        Returns
        -------
        fpath_out : str
            Full file path with directory. format is /dir/YYYYMMDD_{var}_{i}.h5
        """

        fname = ('{}{}{}'.format(date.year,
                                 str(date.month).zfill(2),
                                 str(date.day).zfill(2)))
        fname += '_{var}_{i}.h5'
        fpath_out = os.path.join(self._daily_dir, fname)
        return fpath_out

    def _init_loggers(self, loggers=None, log_file='nsrdb.log',
                      log_level='DEBUG', date=None, log_version=True,
                      use_log_dir=True):
        """Initialize nsrdb loggers.

        Parameters
        ----------
        loggers : None | list | tuple
            List of logger names to initialize. None defaults to all NSRDB
            loggers.
        log_file : str
            Log file name. Will be placed in the nsrdb out dir.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        date : None | datetime
            Optional date to put in the log file name.
        use_log_dir : bool
            Flag to use the class log directory (self._log_dir = ./logs/)
        """

        if log_level in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):

            if loggers is None:
                loggers = ('nsrdb.nsrdb', 'nsrdb.data_model',
                           'nsrdb.file_handlers', 'nsrdb.all_sky',
                           'nsrdb.gap_fill')

            if log_file is not None and use_log_dir:
                log_file = os.path.join(self._log_dir, log_file)
                create_dirs(os.path.dirname(log_file))

            if isinstance(date, datetime.date) and log_file is not None:
                doy = str(date.timetuple().tm_yday).zfill(3)
                date_str = ('{}{}{}'.format(date.year,
                                            str(date.month).zfill(2),
                                            str(date.day).zfill(2)))

                log_file = log_file.replace('.log',
                                            '_{}_{}.log'.format(doy, date_str))

            for name in loggers:
                init_logger(name, log_level=log_level, log_file=log_file)

        if log_version:
            self._log_version()

    @staticmethod
    def init_output_h5(f_out, dsets, time_index, meta, force=False,
                       var_meta=None):
        """Initialize a target output h5 file if it does not already exist
        or if Force is True.

        Parameters
        ----------
        f_out : str
            File path to final .h5 file.
        dsets : list
            List of dataset / variable names that are to be contained in f_out.
        time_index : pd.datetimeindex
            Time index to init to file.
        meta : pd.DataFrame
            Meta data to init to file.
        force : bool
            Flag to overwrite / force the creation of the f_out even if a
            previous file exists.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        """

        if force or not os.path.isfile(f_out):
            logger.info('Initializing {} for the following datasets: {}'
                        .format(f_out, dsets))

            attrs, chunks, dtypes = VarFactory.get_dsets_attrs(
                dsets, var_meta=var_meta)

            mode = 'w' if force else 'w-'
            Outputs.init_h5(f_out, dsets, attrs, chunks, dtypes,
                            time_index, meta, mode=mode)

    @classmethod
    def _parse_data_model_output_ti(cls, data_model_dir, nsrdb_freq):
        """Parse a directory of data model outputs for the resulting time index

        Parameters
        ----------
        data_model_dir : str
            Directory of data model daily output files. This method looks for
            the "YYYYMMDD_" file prefix.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.

        Returns
        -------
        ti : pd.Datetimeindex
            Datetime index encompassing the data model output files.
        """

        date_list = [int(f.split('_')[0]) for f in os.listdir(data_model_dir)
                     if len(f.split('_')[0]) == 8 and f.endswith('.h5')]
        date_list = sorted(date_list)

        start = cls.to_datetime(date_list[0])
        end = cls.to_datetime(date_list[-1]) + datetime.timedelta(days=1)

        ti = pd_date_range(start=start, end=end, freq=nsrdb_freq,
                           closed='left')
        return ti

    @staticmethod
    def doy_to_datestr(year, doy):
        """Convert day of year to YYYYMMDD string format

        Parameters
        ----------
        year : int
            Year of interest
        doy : int
            Enumerated day of year.

        Returns
        -------
        date : str
            Single day to extract ancillary data for.
            str in YYYYMMDD format.
        """
        date = (datetime.datetime(int(year), 1, 1)
                + datetime.timedelta(int(doy) - 1))
        datestr = '{}{}{}'.format(date.year,
                                  str(date.month).zfill(2),
                                  str(date.day).zfill(2))
        return datestr

    @classmethod
    def date_to_doy(cls, date):
        """Convert a date to a day of year integer.

        Parameters
        ----------
        date : datetime.date | str | int
            Single day to extract ancillary data for.
            Can be str or int in YYYYMMDD format.

        Returns
        -------
        doy : int
            Day of year.
        """
        return cls.to_datetime(date).timetuple().tm_yday

    @staticmethod
    def to_datetime(date):
        """Convert a date string or integer to datetime object.

        Parameters
        ----------
        date : datetime.date | str | int
            Single day to extract ancillary data for.
            Can be str or int in YYYYMMDD format.

        Returns
        -------
        date : datetime.date
            Date object.
        """

        if isinstance(date, (int, float)):
            date = str(int(date))
        if isinstance(date, str):
            if len(date) == 8:
                date = datetime.date(year=int(date[0:4]),
                                     month=int(date[4:6]),
                                     day=int(date[6:]))
            else:
                raise ValueError('Could not parse date: {}'.format(date))

        return date

    @classmethod
    def run_data_model(cls, out_dir, date, grid, dist_lim=1.0, var_list=None,
                       freq='5min', var_meta=None, factory_kwargs=None,
                       mlclouds=False, max_workers=None,
                       max_workers_regrid=None,
                       log_level='DEBUG', log_file='data_model.log',
                       job_name=None):
        """Run daily data model, and save output files.

        Parameters
        ----------
        out_dir : str
            Project directory.
        date : datetime.date | str | int
            Single day to extract ancillary data for.
            Can be str or int in YYYYMMDD format.
        grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        dist_lim : float
            Return only neighbors within this distance during cloud regrid.
            The distance is in decimal degrees (more efficient than real
            distance). NSRDB sites further than this value from GOES data
            pixels will be warned and given missing cloud types and properties
            resulting in a full clearsky timeseries.
        var_list : list | tuple | None
            Variables to process with the data model. None will default to all
            NSRDB variables.
        freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        factory_kwargs : dict | None
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs
        mlclouds : bool
            Flag to add extra variables to the variable processing list of
            mlclouds gap fill is expected to be run as the next pipeline step.
        max_workers : int | None
            Number of workers to run in parallel. 1 runs serial,
            None uses all available workers.
        max_workers_regrid : None | int
            Max parallel workers allowed for cloud regrid processing. None uses
            all available workers. 1 runs regrid in serial.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        job_name : str
            Optional name for pipeline and status identification.
        """

        t0 = time.time()
        date = cls.to_datetime(date)

        nsrdb = cls(out_dir, date.year, grid, freq=freq, var_meta=var_meta)
        nsrdb._init_loggers(date=date, log_file=log_file, log_level=log_level)

        fpath_out = nsrdb._get_daily_fpath_out(date)

        if isinstance(factory_kwargs, str):
            factory_kwargs = factory_kwargs.replace('True', 'true')
            factory_kwargs = factory_kwargs.replace('False', 'false')
            factory_kwargs = factory_kwargs.replace('None', 'null')
            factory_kwargs = json.loads(factory_kwargs)
        if isinstance(var_list, str):
            var_list = json.loads(var_list)

        data_model = nsrdb._exe_daily_data_model(
            date.month, date.day,
            dist_lim=dist_lim,
            var_list=var_list,
            factory_kwargs=factory_kwargs,
            max_workers=max_workers,
            max_workers_regrid=max_workers_regrid,
            fpath_out=fpath_out,
            mlclouds=mlclouds)

        if fpath_out is None:
            nsrdb._exe_fout(data_model)

        if job_name is not None:
            runtime = (time.time() - t0) / 60
            year = date.year
            date_str = nsrdb.doy_to_datestr(year, nsrdb.date_to_doy(date))
            status = {'out_dir': nsrdb._daily_dir,
                      'fout': fpath_out,
                      'job_status': 'successful',
                      'runtime': runtime,
                      'grid': grid,
                      'freq': freq,
                      'var_meta': nsrdb._var_meta,
                      'data_model_date': date_str}
            Status.make_job_file(nsrdb._out_dir, 'data-model',
                                 job_name, status)

    @classmethod
    def collect_data_model(cls, out_dir, year, grid, n_chunks, i_chunk,
                           i_fname, n_writes=1, freq='5min', var_meta=None,
                           log_level='DEBUG', log_file='collect_dm.log',
                           max_workers=None, job_name=None, final=False,
                           final_file_name=None):
        """Collect daily data model files to a single site-chunked output file.

        Parameters
        ----------
        out_dir : str
            Project directory.
        year : int | str
            Year of analysis
        grid : str
            Final/full NSRDB grid file. The first column must be the NSRDB
            site gid's.
        n_chunks : int
            Number of chunks (site-wise) to collect to.
        i_chunks : int
            Chunk index (site-wise) (indexing n_chunks) to run.
        i_fname : int
            File name index from sorted NSRDB.OUTS keys to run collection for.
        n_writes : None | int
            Number of file list divisions to write per dataset. For example,
            if ghi and dni are being collected and n_writes is set to 2,
            half of the source ghi files will be collected at once and then
            written, then the second half of ghi files, then dni.
        freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        max_workers : int | None
            Number of workers to run in parallel. 1 runs serial,
            None uses all available workers.
        job_name : str
            Optional name for pipeline and status identification.
        final : bool
            Flag signifying that this is the last step in the NSRDB pipeline.
            this will collect the data to the out_dir/final/ directory instead
            of the out_dir/collect Directory.
        final_file_name : str | None
            Final file name for filename outputs if this is the
            terminal job.
        """

        t0 = time.time()
        nsrdb = cls(out_dir, year, grid, freq=freq, var_meta=var_meta)
        nsrdb._init_loggers(log_file=log_file, log_level=log_level)
        ti = nsrdb._parse_data_model_output_ti(nsrdb._daily_dir, freq)

        chunks = np.array_split(range(len(nsrdb.meta)), n_chunks)

        fnames = sorted(list(cls.OUTS.keys()))

        chunk = chunks[i_chunk]
        fname = fnames[i_fname]
        dsets = cls.OUTS[fname]

        if '{y}' in fname:
            fname = fname.format(y=year)

        if final:
            if final_file_name is not None:
                fname = fname.replace('nsrdb_', '{}_'.format(final_file_name))
            f_out = os.path.join(nsrdb._final_dir, fname)
        else:
            f_out = os.path.join(nsrdb._collect_dir, fname)
            f_out = f_out.replace('.h5', '_{}.h5'.format(i_chunk))

        meta_chunk = nsrdb.meta.iloc[chunk, :]
        if n_chunks > 1 and 'gid' not in meta_chunk:
            # make sure to track gids if collecting in spatial chunks
            meta_chunk['gid'] = meta_chunk.index
            if not final:
                # if not the final collection, minimize file size
                meta_chunk = meta_chunk[['gid']]

        logger.info('Running data model collection for chunk {} out of {} '
                    'with meta gid {} to {} and target file: {}'
                    .format(i_chunk, n_chunks, meta_chunk.index.values[0],
                            meta_chunk.index.values[-1], f_out))

        nsrdb.init_output_h5(f_out, dsets, ti, meta_chunk, force=True,
                             var_meta=nsrdb._var_meta)
        Collector.collect_daily(nsrdb._daily_dir, f_out, dsets,
                                sites=chunk, n_writes=n_writes,
                                var_meta=nsrdb._var_meta,
                                max_workers=max_workers)
        logger.info('Finished file collection to: {}'.format(f_out))

        if job_name is not None:
            runtime = (time.time() - t0) / 60
            status = {'out_dir': nsrdb._collect_dir,
                      'fout': f_out,
                      'job_status': 'successful',
                      'runtime': runtime,
                      'grid': grid,
                      'freq': freq,
                      'n_chunks': n_chunks,
                      'i_chunk': i_chunk,
                      'i_fname': i_fname,
                      }
            Status.make_job_file(nsrdb._out_dir, 'collect-data-model',
                                 job_name, status)

    @classmethod
    def collect_final(cls, collect_dir, out_dir, year, grid, freq='5min',
                      var_meta=None, i_fname=None, tmp=False,
                      log_level='DEBUG', log_file='final_collection.log',
                      job_name=None):
        """Collect chunked files to single final output files.

        Parameters
        ----------
        collect_dir : str
            Directory with chunked files to be collected.
        out_dir : str
            Project directory.
        year : int | str
            Year of analysis
        grid : str
            Final/full NSRDB grid file. The first column must be the NSRDB
            site gid's.
        freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        i_fname : int | None
            Optional index to collect just a single output file. Indexes the
            sorted OUTS class attribute keys.
        tmp : bool
            Flag to use temporary scratch storage, then move to out_dir when
            finished. Doesn't seem to be faster than collecting to normal
            scratch on hpc.
        job_name : str
            Optional name for pipeline and status identification.
        """

        t0 = time.time()
        nsrdb = cls(out_dir, year, grid, freq=freq, var_meta=var_meta)
        nsrdb._init_loggers(log_file=log_file, log_level=log_level)
        ti = nsrdb._parse_data_model_output_ti(nsrdb._daily_dir, freq)

        fnames = sorted(list(cls.OUTS.keys()))
        if i_fname is not None:
            fnames = [fnames[i_fname]]

        for fname in fnames:
            dsets = cls.OUTS[fname]
            fname = fname.format(y=year)

            if tmp:
                dir_out = '/tmp/scratch/'
            else:
                dir_out = nsrdb._final_dir

            flist = [fn for fn in os.listdir(collect_dir)
                     if fn.endswith('.h5')
                     and fname.replace('.h5', '') in fn]
            flist = sorted(flist, key=lambda x: float(
                x.replace('.h5', '').split('_')[-1]))
            fids = [int(fn.replace('.h5', '').split('_')[-1]) for fn in flist]
            if fids != list(range(np.min(fids), 1 + np.max(fids))):
                emsg = ('File list appears to be missing files. '
                        '{} files from {} to {}.'
                        .format(len(flist), np.min(fids), np.max(fids)))
                raise FileNotFoundError(emsg)

            if job_name is not None:
                if job_name.endswith('_{}'.format(i_fname)):
                    jns = job_name[:-2]
                    fname = fname.replace('nsrdb_', '{}_'.format(jns))
                else:
                    fname = fname.replace('nsrdb_', '{}_'.format(job_name))

                f_out = os.path.join(dir_out, fname)

            if any(flist):
                nsrdb.init_output_h5(f_out, dsets, ti, nsrdb.meta,
                                     var_meta=nsrdb._var_meta)
                logger.info('Collecting {} files in list: {}'
                            .format(len(flist), flist))

                for dset in dsets:
                    logger.info('Collecting dataset "{}".'.format(dset))
                    Collector.collect_flist_lowmem(flist, collect_dir, f_out,
                                                   dset,
                                                   var_meta=nsrdb._var_meta)

            else:
                emsg = ('Could not find files to collect for {} in the '
                        'collect dir: {}'
                        .format(fname, collect_dir))
                raise FileNotFoundError(emsg)

            if tmp:
                logger.info('Moving temp file to final output directory.')
                shutil.move(f_out, os.path.join(out_dir, fname))

        logger.info('Finished final file collection to: {}'.format(out_dir))

        if job_name is not None:
            runtime = (time.time() - t0) / 60
            status = {'out_dir': nsrdb._final_dir,
                      'fout': f_out,
                      'job_status': 'successful',
                      'runtime': runtime,
                      'grid': grid,
                      'freq': freq,
                      }
            Status.make_job_file(nsrdb._out_dir, 'collect-final',
                                 job_name, status)

    @classmethod
    def gap_fill_clouds(cls, out_dir, year, i_chunk,
                        rows=slice(None), cols=slice(None),
                        col_chunk=None, var_meta=None, log_level='DEBUG',
                        log_file='cloud_fill.log', job_name=None):
        """Gap fill cloud properties in a collected data model output file.

        Parameters
        ----------
        out_dir : str
            Project directory.
        year : int | str
            Year of analysis
        i_chunks : int
            Chunk index (indexing n_chunks) to run.
        rows : slice
            Subset of rows to gap fill.
        cols : slice
            Subset of columns to gap fill.
        col_chunk : None | int
            Optional chunking method to gap fill one column chunk at a time
            to reduce memory requirements. If provided, this should be an
            integer specifying how many columns to work on at one time.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        job_name : str
            Optional name for pipeline and status identification.
        """
        t0 = time.time()
        nsrdb = cls(out_dir, year, None, var_meta=var_meta)

        fname_clouds = [fn for fn in nsrdb.OUTS
                        if 'cloud' in fn][0].format(y=year)
        fname_ancillary = [fn for fn in nsrdb.OUTS
                           if 'ancillary_b' in fn][0].format(y=year)
        f_cloud = os.path.join(nsrdb._collect_dir, fname_clouds)
        f_cloud = f_cloud.replace('.h5', '_{}.h5'.format(i_chunk))
        f_ancillary = os.path.join(nsrdb._collect_dir, fname_ancillary)
        f_ancillary = f_ancillary.replace('.h5', '_{}.h5'.format(i_chunk))

        nsrdb._init_loggers(log_file=log_file, log_level=log_level)
        CloudGapFill.fill_file(f_cloud, f_ancillary, rows=rows, cols=cols,
                               col_chunk=col_chunk)
        logger.info('Finished cloud gap fill.')

        if job_name is not None:
            runtime = (time.time() - t0) / 60
            status = {'out_dir': nsrdb._collect_dir,
                      'fout': f_cloud,
                      'gap_fill_method': 'legacy_nn',
                      'job_status': 'successful',
                      'runtime': runtime,
                      }
            Status.make_job_file(nsrdb._out_dir, 'cloud-fill',
                                 job_name, status)

    @classmethod
    def ml_cloud_fill(cls, out_dir, date, fill_all=False,
                      model_path=None, var_meta=None,
                      log_level='DEBUG', log_file='cloud_fill.log',
                      job_name=None, col_chunk=None, max_workers=None):
        """Gap fill cloud properties using a physics-guided neural
        network (phygnn).

        Parameters
        ----------
        out_dir : str
            Project directory.
        date : datetime.date | str | int
            Single day data model output to run cloud fill on.
            Can be str or int in YYYYMMDD format.
        fill_all : bool
            Flag to fill all cloud properties for all timesteps where
            cloud_type is cloudy.
        model_path : str | None
            Directory to load phygnn model from. This is typically a fpath to
            a .pkl file with an accompanying .json file in the same directory.
            None will try to use the default model path from the mlclouds
            project directory.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        job_name : str
            Optional name for pipeline and status identification.
        col_chunk : None | int
            Optional chunking method to gap fill one column chunk at a time
            to reduce memory requirements. If provided, this should be an
            integer specifying how many columns to work on at one time.
        max_workers : None | int
            Maximum workers to clean data in parallel. 1 is serial and None
            uses all available workers.
        """
        from nsrdb.gap_fill.mlclouds_fill import MLCloudsFill
        t0 = time.time()
        assert len(str(date)) == 8
        nsrdb = cls(out_dir, str(date)[0:4], None, var_meta=var_meta)
        h5_source = os.path.join(nsrdb._daily_dir, str(date) + '_*.h5')
        nsrdb._init_loggers(log_file=log_file, log_level=log_level)

        is_merra = MLCloudsFill.merra_clouds(h5_source, var_meta=var_meta)

        if not is_merra:
            MLCloudsFill.run(h5_source,
                             fill_all=fill_all,
                             model_path=model_path,
                             var_meta=var_meta,
                             col_chunk=col_chunk,
                             max_workers=max_workers)

        logger.info('Finished mlclouds gap fill.')

        if job_name is not None:
            runtime = (time.time() - t0) / 60
            status = {'out_dir': nsrdb._daily_dir,
                      'h5_source': h5_source,
                      'gap_fill_method': 'mlclouds',
                      'job_status': 'successful',
                      'runtime': runtime,
                      }
            Status.make_job_file(nsrdb._out_dir, 'ml-cloud-fill',
                                 job_name, status)

    @classmethod
    def run_all_sky(cls, out_dir, year, grid, freq='5min', var_meta=None,
                    col_chunk=10, rows=slice(None), cols=slice(None),
                    max_workers=None, log_level='DEBUG',
                    log_file='all_sky.log', i_chunk=None, job_name=None,
                    disc_on=False):
        """Run the all-sky physics model from collected .h5 files

        Parameters
        ----------
        out_dir : str
            Project directory.
        year : int | str
            Year of analysis
        grid : str
            Final/full NSRDB grid file. The first column must be the NSRDB
            site gid's.
        freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        col_chunk :  int
            Chunking method to run all sky one column chunk at a time
            to reduce memory requirements. This is an integer specifying
            how many columns to work on at one time.
        rows : slice
            Subset of rows to run.
        cols : slice
            Subset of columns to run.
        max_workers : int | None
            Number of workers to run in parallel. 1 will run serial,
            None will use all available.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        i_chunk : None | int
            Enumerated file index if running on site chunk.
        job_name : str
            Optional name for pipeline and status identification.
        disc_on : bool
            Compute cloudy sky dni with the disc model (True) or the farms-dni
            model (False)
        """
        t0 = time.time()
        nsrdb = cls(out_dir, year, grid, freq=freq, var_meta=var_meta)
        nsrdb._init_loggers(log_file=log_file, log_level=log_level)

        for fname, dsets in cls.OUTS.items():
            if 'irradiance' in fname:
                f_out = os.path.join(nsrdb._collect_dir,
                                     fname.format(y=year))
                irrad_dsets = dsets
            elif 'clear' in fname:
                f_out_cs = os.path.join(nsrdb._collect_dir,
                                        fname.format(y=year))
                cs_irrad_dsets = dsets

        f_source = os.path.join(nsrdb._collect_dir, 'nsrdb*{}.h5'.format(year))

        if i_chunk is not None:
            f_out = f_out.replace('.h5', '_{}.h5'.format(i_chunk))
            f_out_cs = f_out_cs.replace('.h5', '_{}.h5'.format(i_chunk))
            f_source = f_source.replace('.h5', '_{}.h5'.format(i_chunk))

        with MultiFileResource(f_source) as source:
            meta = source.meta
            time_index = source.time_index.tz_convert(None)

        nsrdb.init_output_h5(f_out, irrad_dsets, time_index, meta,
                             var_meta=nsrdb._var_meta)
        nsrdb.init_output_h5(f_out_cs, cs_irrad_dsets, time_index, meta,
                             var_meta=nsrdb._var_meta)

        if max_workers != 1:
            out = all_sky_h5_parallel(f_source, rows=rows, cols=cols,
                                      max_workers=max_workers,
                                      col_chunk=col_chunk, disc_on=disc_on)
        else:
            out = all_sky_h5(f_source, rows=rows, cols=cols,
                             col_chunk=col_chunk, disc_on=disc_on)

        logger.info('Finished all-sky. Writing to: {}'.format(f_out))
        with Outputs(f_out, mode='a') as f:
            for dset, arr in out.items():
                if dset in f.dsets:
                    f[dset, rows, cols] = arr

        logger.info('Finished all-sky. Writing to: {}'.format(f_out_cs))
        with Outputs(f_out_cs, mode='a') as f:
            for dset, arr in out.items():
                if dset in f.dsets:
                    f[dset, rows, cols] = arr

        logger.info('Finished writing all-sky results.')

        if job_name is not None:
            runtime = (time.time() - t0) / 60
            status = {'out_dir': nsrdb._collect_dir,
                      'fout': f_out,
                      'job_status': 'successful',
                      'runtime': runtime,
                      'grid': grid,
                      'freq': freq,
                      }
            Status.make_job_file(nsrdb._out_dir, 'all-sky', job_name, status)

    @classmethod
    def run_daily_all_sky(cls, out_dir, year, grid, date, freq='5min',
                          var_meta=None, col_chunk=500,
                          rows=slice(None), cols=slice(None),
                          max_workers=None, log_level='DEBUG',
                          log_file='all_sky.log', job_name=None,
                          disc_on=False):
        """Run the all-sky physics model from daily data model output files.

        Parameters
        ----------
        out_dir : str
            Project directory.
        year : int | str
            Year of analysis
        grid : str
            Final/full NSRDB grid file. The first column must be the NSRDB
            site gid's.
        date : datetime.date | str | int
            Single day data model output to run cloud fill on.
            Can be str or int in YYYYMMDD format.
        freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        col_chunk :  int
            Chunking method to run all sky one column chunk at a time
            to reduce memory requirements. This is an integer specifying
            how many columns to work on at one time.
        rows : slice
            Subset of rows to run.
        cols : slice
            Subset of columns to run.
        max_workers : int | None
            Number of workers to run in parallel. 1 will run serial,
            None will use all available.
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        job_name : str
            Optional name for pipeline and status identification.
        disc_on : bool
            Compute cloudy sky dni with the disc model (True) or the farms-dni
            model (False)
        """
        t0 = time.time()
        assert len(str(date)) == 8
        nsrdb = cls(out_dir, year, grid, freq=freq, var_meta=var_meta)
        nsrdb._init_loggers(log_file=log_file, log_level=log_level)

        f_source = os.path.join(nsrdb._daily_dir, '{}*.h5'.format(date))

        with MultiFileResource(f_source) as source:
            meta = source.meta
            time_index = source.time_index.tz_convert(None)

        if max_workers != 1:
            out = all_sky_h5_parallel(f_source, rows=rows, cols=cols,
                                      max_workers=max_workers,
                                      col_chunk=col_chunk, disc_on=disc_on)
        else:
            out = all_sky_h5(f_source, rows=rows, cols=cols, disc_on=disc_on)

        logger.info('Finished all-sky compute.')
        for dset, arr in out.items():
            fn = '{}_{}_0.h5'.format(date, dset)
            f_out = os.path.join(nsrdb._daily_dir, fn)
            logger.info('Writing {} to: {}'.format(dset, f_out))
            nsrdb.init_output_h5(f_out, [dset], time_index, meta,
                                 var_meta=nsrdb._var_meta)
            with Outputs(f_out, mode='a') as f:
                f[dset, rows, cols] = arr

        logger.info('Finished writing all-sky results.')

        if job_name is not None:
            runtime = (time.time() - t0) / 60
            status = {'out_dir': nsrdb._daily_dir,
                      'f_source': f_source,
                      'f_out': '{}_*_0.h5'.format(date),
                      'job_status': 'successful',
                      'runtime': runtime,
                      'grid': grid,
                      'freq': freq,
                      }
            Status.make_job_file(nsrdb._out_dir, 'daily-all-sky',
                                 job_name, status)

    @classmethod
    def run_full(cls, date, grid, freq, var_meta=None, factory_kwargs=None,
                 fill_all=False, model_path=None, dist_lim=1.0,
                 max_workers=None, low_mem=False,
                 log_file=None, log_level='INFO', disc_on=False):
        """Run the full nsrdb pipeline in-memory using serial compute.

        Parameters
        ----------
        date : datetime.date | str | int
            Single day to extract ancillary data for.
            Can be str or int in YYYYMMDD format.
        grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe. The first csv column
            must be the NSRDB site gid's.
        freq : str
            Final desired NSRDB temporal frequency.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        factory_kwargs : dict | None
            Optional namespace of kwargs to use to initialize variable data
            handlers from the data model's variable factory. Keyed by
            variable name. Values can be "source_dir", "handler", etc...
            source_dir for cloud variables can be a normal directory
            path or /directory/prefix*suffix where /directory/ can have
            more sub dirs
        fill_all : bool
            Flag to fill all cloud properties for all timesteps where
            cloud_type is cloudy.
        model_path : str | None
            Directory to load phygnn model from. This is typically a fpath to
            a .pkl file with an accompanying .json file in the same directory.
            None will try to use the default model path from the mlclouds
            project directory.
        dist_lim : float
            Return only neighbors within this distance during cloud regrid.
            The distance is in decimal degrees (more efficient than real
            distance). NSRDB sites further than this value from GOES data
            pixels will be warned and given missing cloud types and properties
            resulting in a full clearsky timeseries.
        max_workers : int, optional
            Number of workers to use for NSRDB computation. If 1 run in serial,
            else in parallel. If None use all available cores. by default None
        low_mem : bool
            Option to run predictions in low memory mode. Typically the
            memory bloat during prediction is:
            (n_time x n_sites x n_nodes_per_layer). low_mem=True will
            reduce this to (1000 x n_nodes_per_layer)
        log_level : str | None
            Logging level (DEBUG, INFO). If None, no logging will be
            initialized.
        log_file : str
            File to log to. Will be put in output directory.
        disc_on : bool
            Compute cloudy sky dni with the disc model (True) or the farms-dni
            model (False)

        Returns
        -------
        data_model : nsrdb.data_model.DataModel
            DataModel instance with all processed nsrdb variables available in
            the DataModel.processed_data attribute.
        """

        t0 = time.time()
        from nsrdb.gap_fill.mlclouds_fill import MLCloudsFill
        date = cls.to_datetime(date)

        nsrdb = cls('./', date.year, grid, freq=freq, var_meta=var_meta,
                    make_out_dirs=False)
        nsrdb._init_loggers(date=date, log_file=log_file, log_level=log_level,
                            use_log_dir=False)

        logger.info('Starting daily data model execution for {}-{}-{}'
                    .format(date.month, date.day, date.year))

        data_model = DataModel.run_multiple(
            DataModel.ALL_VARS_ML, date, grid,
            nsrdb_freq=freq,
            var_meta=var_meta,
            max_workers=max_workers,
            max_workers_regrid=max_workers,
            return_obj=True,
            fpath_out=None,
            scale=False,
            dist_lim=dist_lim,
            factory_kwargs=factory_kwargs)

        logger.info('Finished daily data model execution for {}-{}-{}'
                    .format(date.month, date.day, date.year))

        data_model = MLCloudsFill.clean_data_model(data_model,
                                                   fill_all=fill_all,
                                                   model_path=model_path,
                                                   var_meta=var_meta,
                                                   low_mem=low_mem)

        all_sky_inputs = {k: v for k, v in data_model.processed_data.items()
                          if k in ALL_SKY_ARGS}
        all_sky_inputs['time_index'] = data_model.nsrdb_ti
        all_sky_inputs['scale_outputs'] = False
        all_sky_inputs['disc_on'] = disc_on
        logger.info('Running NSRDB All-Sky.')
        all_sky_out = all_sky(**all_sky_inputs)
        for k, v in all_sky_out.items():
            logger.debug('Sending all sky output "{}" to data model.'
                         .format(k))
            data_model[k] = v

        logger.info('Finished running NSRDB All-Sky.')
        runtime = (time.time() - t0) / 60
        logger.info('NSRDB full processing job is complete in {:.2f} minutes.'
                    .format(runtime))

        return data_model
