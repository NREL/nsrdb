# -*- coding: utf-8 -*-
"""Entry point for NSRDB data pipeline execution.

Created on Thu Apr 25 15:47:53 2019

@author: gbuster

TODO: Clean up create_config_files, blend_files, aggregate_files
"""

import calendar
import copy
import json
import logging
import os
import pprint

from nsrdb import CONFIGDIR
from nsrdb.utilities.file_utils import (
    str_replace_dict,
)

PRE2018_CONFIG_TEMPLATE = os.path.join(
    CONFIGDIR, 'templates/config_nsrdb_pre2018.json'
)
POST2017_CONFIG_TEMPLATE = os.path.join(
    CONFIGDIR, 'templates/config_nsrdb_post2017.json'
)
PIPELINE_CONFIG_TEMPLATE = os.path.join(
    CONFIGDIR, 'templates/config_pipeline.json'
)

DEFAULT_EXEC_CONFIG = {
    'option': 'kestrel',
    'memory': 173,
    'walltime': 10,
    'alloc': 'pxs',
}

logger = logging.getLogger(__name__)


class CreateConfigs:
    """Collection of methods to create config files for NSRDB module inputs for
    standard CONUS / Full Disc runs."""

    @staticmethod
    def aggregate(kwargs):
        """Get config for conus and full disk high-resolution to low-resolution
        aggregation.  This is then used as the input to `nsrdb.cli.aggregate`

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the
            case for which to aggregate files
        """
        default_kwargs = {
            'basename': 'nsrdb',
            'metadir': '/projects/pxs/reference_grids',
            'full_spatial': '2km',
            'conus_spatial': '2km',
            'final_spatial': '4km',
            'out_dir': './',
            'full_freq': '10min',
            'conus_freq': '5min',
            'final_freq': '30min',
            'n_chunks': 32,
            'source_priority': ['conus', 'full_disk'],
            'execution_control': DEFAULT_EXEC_CONFIG,
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
            'full_disk': {
                'data_sub_dir': full_sub_dir,
                'tree_file': full_tree_file,
                'meta_file': full_meta_file,
                'spatial': f'{user_input["full_spatial"]}',
                'temporal': f'{user_input["full_freq"]}',
            },
            'conus': {
                'data_sub_dir': conus_sub_dir,
                'tree_file': conus_tree_file,
                'meta_file': conus_meta_file,
                'spatial': f'{user_input["conus_spatial"]}',
                'temporal': f'{user_input["conus_freq"]}',
            },
            'final': {
                'data_sub_dir': final_sub_dir,
                'fout': 'nsrdb.h5',
                'tree_file': tree_file.format(res=user_input['final_spatial']),
                'meta_file': meta_file.format(res=user_input['final_spatial']),
                'spatial': f'{user_input["final_spatial"]}',
                'temporal': f'{user_input["final_freq"]}',
                'source_priority': source_priority,
            },
        }

        user_input['data'] = NSRDB

        out_dir = user_input['out_dir']
        os.makedirs(out_dir, exist_ok=True)
        config_file = os.path.join(out_dir, 'config_aggregate.json')
        with open(config_file, 'w') as f:
            f.write(json.dumps(user_input, indent=2))

        logger.info(f'Created config file: {config_file}.')

    @classmethod
    def blend(cls, kwargs):
        """Get config dictionary for nsrdb.cli.blend for standard NSRDB runs.

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the
            case for which to blend data files
        """
        default_kwargs = {
            'file_tag': 'all',
            'basename': 'nsrdb',
            'metadir': '/projects/pxs/reference_grids',
            'spatial': '2km',
            'extent': 'conus',
            'out_dir': './',
            'chunk_size': 100000,
            'meta_file': None,
            'east_dir': None,
            'west_dir': None,
            'execution_control': DEFAULT_EXEC_CONFIG,
        }
        user_input = copy.deepcopy(default_kwargs)
        user_input.update(kwargs)

        if user_input['year'] < 2018:
            user_input['extent'] = 'full'
            user_input['spatial'] = '4km'

        map_col_map = {'full': 'gid_full', 'conus': 'gid_full_conus'}
        user_input['map_col'] = (
            user_input.get('map_col', None)
            or map_col_map[user_input['extent']]
        )

        meta_lon_map = {'full': -105, 'conus': -113}
        user_input['meta_lon'] = (
            user_input.get('lon_seam', None)
            or meta_lon_map[user_input['extent']]
        )

        if user_input['meta_file'] is None:
            meta_file = f'nsrdb_meta_{user_input["spatial"]}'

            if user_input['year'] > 2017:
                meta_file += f'_{user_input["extent"]}'

            meta_file += '.csv'
            user_input['meta_file'] = os.path.join(
                user_input['metadir'], meta_file
            )

        src_dir = f"{user_input['basename']}"
        src_dir += '_{satellite}'
        src_dir += f"_{user_input['extent']}_{user_input['year']}"
        src_dir += f"_{user_input['spatial']}/final"
        src_dir = os.path.join(user_input['out_dir'], src_dir)

        if user_input['east_dir'] is None:
            user_input['east_dir'] = src_dir.format(satellite='east')
        if user_input['west_dir'] is None:
            user_input['west_dir'] = src_dir.format(satellite='west')

        out_dir = user_input['out_dir']
        os.makedirs(out_dir, exist_ok=True)
        config_file = os.path.join(out_dir, 'config_blend.json')
        with open(config_file, 'w') as f:
            f.write(json.dumps(user_input, indent=2))

        logger.info(f'Created config file: {config_file}.')

    @classmethod
    def main_all_domains(cls, kwargs):
        """Modify config files for all domains with specified parameters

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameters including year, basename, satellite,
            extent, freq, spatial, meta_file, doy_range
        """
        if kwargs['year'] < 2018:
            kwargs.update(
                {
                    'spatial': '4km',
                    'extent': 'full',
                    'freq': '30min',
                    'satellite': 'east',
                }
            )
            cls.create_config_files(kwargs)
            kwargs.update({'satellite': 'west'})
            cls.create_config_files(kwargs)
        elif kwargs['year'] == 2018:
            kwargs.update(
                {
                    'spatial': '2km',
                    'extent': 'full',
                    'freq': '10min',
                    'satellite': 'east',
                }
            )
            cls.create_config_files(kwargs)
            kwargs.update({'extent': 'conus', 'freq': '5min'})
            cls.create_config_files(kwargs)
            kwargs.update(
                {
                    'spatial': '4km',
                    'extent': 'full',
                    'freq': '30min',
                    'satellite': 'west',
                }
            )
            cls.create_config_files(kwargs)
        else:
            kwargs.update(
                {
                    'spatial': '2km',
                    'extent': 'full',
                    'freq': '10min',
                    'satellite': 'east',
                }
            )
            cls.create_config_files(kwargs)
            kwargs.update({'satellite': 'west'})
            cls.create_config_files(kwargs)
            kwargs.update(
                {'extent': 'conus', 'freq': '5min', 'satellite': 'east'}
            )
            cls.create_config_files(kwargs)
            kwargs.update({'satellite': 'west'})
            cls.create_config_files(kwargs)

    @staticmethod
    def _update_run_templates(user_input):
        """Replace format keys and dictionary keys in config templates with
        user input values."""

        logger.info(
            'Updating NSRDB run templates with user_input:\n'
            f'{pprint.pformat(user_input, indent=2)}'
        )

        template = (
            PRE2018_CONFIG_TEMPLATE
            if int(user_input['year']) < 2018
            else POST2017_CONFIG_TEMPLATE
        )
        with open(template, encoding='utf-8') as s:
            s = s.read()

        s = str_replace_dict(s, user_input)

        if not os.path.exists(user_input['out_dir']):
            os.makedirs(user_input['out_dir'])

        config_dict = json.loads(s)
        config_dict.update(
            {k: v for k, v in user_input.items() if k in config_dict}
        )
        outfile = os.path.join(user_input['out_dir'], 'config_nsrdb.json')
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(json.dumps(config_dict, indent=2))

        logger.info(f'Created file: {outfile}')

        with open(PIPELINE_CONFIG_TEMPLATE, encoding='utf-8') as s:
            s = s.read()

        s = str_replace_dict(s, user_input)

        outfile = os.path.join(user_input['out_dir'], 'config_pipeline.json')
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(s)

        logger.info(f'Created file: {outfile}')

    @classmethod
    def main(cls, kwargs):
        """Modify config files with specified parameters

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameters including year, basename, satellite,
            extent, freq, spatial, meta_file, doy_range
        """

        default_kwargs = {
            'basename': 'nsrdb',
            'freq': '5min',
            'spatial': '4km',
            'satellite': 'east',
            'extent': 'conus',
            'out_dir': './',
            'meta_file': None,
            'doy_range': None,
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
            if calendar.isleap(user_input['year']):
                user_input['doy_range'] = [1, 367]
            else:
                user_input['doy_range'] = [1, 366]

        user_input['start_doy'] = user_input['doy_range'][0]
        user_input['end_doy'] = user_input['doy_range'][1]

        run_name = '_'.join(
            str(user_input[k])
            for k in [
                'basename',
                'satellite',
                'extent',
                'year',
                'spatial',
                'freq',
            ]
        )

        user_input['out_dir'] = os.path.join(user_input['out_dir'], run_name)

        cls._update_run_templates(user_input)

    @classmethod
    def collect_blend(cls, kwargs):
        """Create blend collect config files.

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for blend collection
        """

        default_kwargs = {
            'basename': 'nsrdb',
            'metadir': '/projects/pxs/reference_grids',
            'spatial': '4km',
            'out_dir': './',
            'freq': '30min',
            'extent': 'full',
            'execution_control': DEFAULT_EXEC_CONFIG,
        }

        user_input = copy.deepcopy(default_kwargs)
        user_input.update(kwargs)

        meta_file = f'nsrdb_meta_{user_input["spatial"]}.csv'
        meta_file = os.path.join(user_input['metadir'], meta_file)
        user_input['meta'] = meta_file
        collect_dir = f'nsrdb_{user_input["year"]}'
        collect_dir += f'_{user_input["extent"]}_blend'
        collect_tag = f'{user_input["basename"]}_'
        collect_tag += f'{user_input["extent"]}_{user_input["year"]}_'
        user_input['collect_dir'] = collect_dir
        user_input['collect_tag'] = collect_tag
        user_input['fout'] = os.path.join(
            f'{user_input["out_dir"]}',
            f'{user_input["basename"]}_{user_input["year"]}.h5',
        )

        out_dir = user_input['out_dir']
        os.makedirs(out_dir, exist_ok=True)
        config_file = os.path.join(out_dir, 'config_collect_blend.json')
        with open(config_file, 'w') as f:
            f.write(json.dumps(user_input, indent=2))

        logger.info(f'Created file: {config_file}')

    @classmethod
    def collect_aggregate(cls, kwargs):
        """Create config for aggregation collection

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for aggregation collection
        """
        default_kwargs = {
            'basename': 'nsrdb',
            'metadir': '/projects/pxs/reference_grids',
            'final_spatial': '4km',
            'final_freq': '30min',
            'out_dir': './',
            'execution_control': DEFAULT_EXEC_CONFIG,
        }

        user_input = copy.deepcopy(default_kwargs)
        user_input.update(kwargs)

        meta_file = f'nsrdb_meta_{user_input["final_spatial"]}.csv'
        meta_file = os.path.join(user_input['metadir'], meta_file)
        collect_dir = f'nsrdb_{user_input["final_spatial"]}'
        collect_dir += f'_{user_input["final_freq"]}'
        collect_tag = f'{user_input["basename"]}_'
        user_input['collect_dir'] = collect_dir
        user_input['collect_tag'] = collect_tag
        user_input['fout'] = os.path.join(
            f'{user_input["out_dir"]}',
            f'{user_input["basename"]}_{user_input["year"]}.h5',
        )

        out_dir = user_input['out_dir']
        os.makedirs(out_dir, exist_ok=True)
        config_file = os.path.join(out_dir, 'config_collect_aggregate.json')
        with open(config_file, 'w') as f:
            f.write(json.dumps(user_input, indent=2))

        logger.info(f'Created file: {config_file}')
