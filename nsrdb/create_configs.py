"""Config creation class for CONUS / Full Disc NSRDB runs."""

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

    RUN_NAME = '{basename}_{satellite}_{extent}_{year}_{spatial}_{freq}'

    @classmethod
    def main_all(cls, kwargs):
        """Modify config files for all domains with specified parameters.

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
            cls.main(kwargs)
            kwargs.update({'satellite': 'west'})
            cls.main(kwargs)
        elif kwargs['year'] == 2018:
            kwargs.update(
                {
                    'spatial': '2km',
                    'extent': 'full',
                    'freq': '10min',
                    'satellite': 'east',
                }
            )
            cls.main(kwargs)
            kwargs.update({'extent': 'conus', 'freq': '5min'})
            cls.main(kwargs)
            kwargs.update(
                {
                    'spatial': '4km',
                    'extent': 'full',
                    'freq': '30min',
                    'satellite': 'west',
                }
            )
            cls.main(kwargs)
        else:
            kwargs.update(
                {
                    'spatial': '2km',
                    'extent': 'full',
                    'freq': '10min',
                    'satellite': 'east',
                }
            )
            cls.main(kwargs)
            kwargs.update({'satellite': 'west'})
            cls.main(kwargs)
            kwargs.update(
                {'extent': 'conus', 'freq': '5min', 'satellite': 'east'}
            )
            cls.main(kwargs)
            kwargs.update({'satellite': 'west'})
            cls.main(kwargs)

    @classmethod
    def full(cls, kwargs):
        """Modify config files for all domains with specified parameters. Write
        all post processing config files and post processing pipeline config
        file for the given year

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameters including year, basename, satellite,
            extent, freq, spatial, meta_file, doy_range
        """
        cls.main_all(kwargs)
        cls.post(kwargs)

    @classmethod
    def _update_run_templates(cls, user_input):
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

        config_dict = json.loads(str_replace_dict(s, user_input))
        config_dict.update(
            {k: v for k, v in user_input.items() if k in config_dict}
        )
        cls._write_config(
            config_dict,
            os.path.join(user_input['out_dir'], 'config_nsrdb.json'),
        )

        with open(PIPELINE_CONFIG_TEMPLATE, encoding='utf-8') as s:
            s = s.read()

        config_dict = json.loads(str_replace_dict(s, user_input))

        cls._write_config(
            config_dict,
            os.path.join(user_input['out_dir'], 'config_pipeline.json'),
        )

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
        os.makedirs(user_input['out_dir'], exist_ok=True)

        extent_tag_map = {'full': 'RadF', 'conus': 'RadC'}
        lon_seam_map = {'full': -105, 'conus': -113}
        user_input['extent_tag'] = extent_tag_map[user_input['extent']]
        lon_seam = lon_seam_map[user_input['extent']]

        if user_input['meta_file'] is None:
            meta_file = f'nsrdb_meta_{user_input["spatial"]}'

            if user_input['year'] > 2017:
                meta_file += f'_{user_input["extent"]}'

            meta_file += f'_{user_input["satellite"]}_{lon_seam}.csv'
            user_input['meta_file'] = meta_file

        if user_input['doy_range'] is None:
            if calendar.isleap(user_input['year']):
                user_input['doy_range'] = [1, 367]
            else:
                user_input['doy_range'] = [1, 366]

        user_input['start_doy'] = user_input['doy_range'][0]
        user_input['end_doy'] = user_input['doy_range'][1]

        run_name = cls.RUN_NAME.format(
            basename=user_input['basename'],
            satellite=user_input['satellite'],
            extent=user_input['extent'],
            year=user_input['year'],
            spatial=user_input['spatial'],
            freq=user_input['freq'],
        )

        user_input['out_dir'] = os.path.join(user_input['out_dir'], run_name)

        cls._update_run_templates(user_input)

        run_file = os.path.join(user_input['out_dir'], 'run.sh')
        with open(run_file, 'w') as f:
            f.write('python -m nsrdb.cli pipeline -c config_pipeline.json')

        logger.info(f'Saved run script: {run_file}.')

    @classmethod
    def post(cls, kwargs):
        """Create all post processing config files for blending / aggregation /
        collection."""

        pipeline_config = {'pipeline': []}
        out_dir = kwargs.get('out_dir', './')

        if kwargs['year'] > 2017:
            kwargs.update({'extent': 'conus'})
            config = cls._blend(kwargs)
            cls._write_config(
                config, 'config_blend_conus.json', module_name='blend'
            )
            pipeline_config['pipeline'].append(
                {
                    'blend-conus': './config_blend_conus.json',
                    'command': 'blend',
                }
            )
            kwargs.update({'extent': 'full'})
            config = cls._blend(kwargs)
            cls._write_config(
                config, 'config_blend_full.json', module_name='blend'
            )
            pipeline_config['pipeline'].append(
                {'blend-full': './config_blend_full.json', 'command': 'blend'}
            )
            config = cls._aggregate(kwargs)
            cls._write_config(
                config, 'config_aggregate.json', module_name='aggregate'
            )
            pipeline_config['pipeline'].append(
                {'aggregate': './config_aggregate.json'}
            )
            config = cls._collect_aggregate(kwargs)
            cls._write_config(
                config,
                'config_collect_aggregate.json',
                module_name='collect-aggregate',
            )
            pipeline_config['pipeline'].append(
                {'collect-aggregate': './config_collect_aggregate.json'}
            )
        else:
            config = cls._blend(kwargs)
            cls._write_config(
                config, 'config_blend_full.json', module_name='blend'
            )
            pipeline_config['pipeline'].append(
                {'blend': './config_blend_full.json'}
            )
            config = cls._collect_blend(kwargs)
            cls._write_config(
                config,
                'config_collect_blend.json',
                module_name='collect-blend',
            )
            pipeline_config['pipeline'].append(
                {'collect-blend': './config_collect_blend.json'}
            )
        cls._write_config(
            pipeline_config, os.path.join(out_dir, 'config_pipeline_post.json')
        )

    @classmethod
    def _aggregate(cls, kwargs):
        """Get config for conus and full disk high-resolution to low-resolution
        aggregation.  This is then used as the input to `nsrdb.cli.aggregate`

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for which to aggregate
            files
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
        run_name = user_input.get('run_name', None)
        user_input['run_name'] = (
            run_name
            if run_name is not None
            else f'aggregate_{user_input["year"]}'
        )
        return user_input

    @classmethod
    def aggregate(cls, kwargs):
        """Get config for conus and full disk high-resolution to low-resolution
        aggregation.  This is then used as the input to `nsrdb.cli.aggregate`

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for which to aggregate
            files
        """

        config = cls._aggregate(kwargs)
        cls._write_config(
            config, 'config_aggregate.json', module_name='aggregate'
        )

    @classmethod
    def _blend(cls, kwargs):
        """Get config dictionary for nsrdb.cli.blend for standard NSRDB runs
        for a given extent.

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for which to blend data
            files
        """
        default_kwargs = {
            'file_tag': 'all',
            'basename': 'nsrdb',
            'metadir': '/projects/pxs/reference_grids',
            'spatial': '4km',
            'extent': 'full',
            'out_dir': './',
            'chunk_size': 100000,
            'meta_file': None,
            'east_dir': None,
            'west_dir': None,
            'freq': '30min',
            'execution_control': DEFAULT_EXEC_CONFIG,
        }
        user_input = copy.deepcopy(default_kwargs)
        user_input.update(kwargs)

        if user_input['year'] > 2017:
            user_input['spatial'] = '2km'
            if user_input['extent'] == 'full':
                user_input['freq'] = '10min'
            else:
                user_input['freq'] = '5min'

        map_col_map = {'full': 'gid_full', 'conus': 'gid_full_conus'}
        user_input['map_col'] = (
            user_input.get('map_col', None)
            or map_col_map[user_input['extent']]
        )

        lon_seam_map = {'full': -105, 'conus': -113}
        user_input['lon_seam'] = (
            user_input.get('lon_seam', None)
            or lon_seam_map[user_input['extent']]
        )

        if user_input['meta_file'] is None:
            meta_file = f'nsrdb_meta_{user_input["spatial"]}'

            if user_input['year'] > 2017:
                meta_file += f'_{user_input["extent"]}'

            meta_file += '.csv'
            user_input['meta_file'] = os.path.join(
                user_input['metadir'], meta_file
            )

        if user_input['east_dir'] is None:
            user_input['east_dir'] = os.path.join(
                user_input['out_dir'],
                cls.RUN_NAME.format(
                    basename=user_input['basename'],
                    satellite='east',
                    extent=user_input['extent'],
                    year=user_input['year'],
                    spatial=user_input['spatial'],
                    freq=user_input['freq'],
                ),
                'final',
            )
        if user_input['west_dir'] is None:
            user_input['west_dir'] = os.path.join(
                user_input['out_dir'],
                cls.RUN_NAME.format(
                    basename=user_input['basename'],
                    satellite='west',
                    extent=user_input['extent'],
                    year=user_input['year'],
                    spatial=user_input['spatial'],
                    freq=user_input['freq'],
                ),
                'final',
            )
        run_name = user_input.get('run_name', None)
        user_input['run_name'] = (
            run_name
            if run_name is not None
            else f'blend_{user_input["extent"]}_{user_input["year"]}'
        )

        return user_input

    @classmethod
    def _write_config(cls, config, config_file, module_name=None):
        """Write config to .json file."""

        config_dir = os.path.dirname(config_file)
        out_dir = config.get('out_dir', config_dir or './')
        os.makedirs(out_dir, exist_ok=True)
        config_file = os.path.join(out_dir, os.path.basename(config_file))

        if module_name is not None:
            exec_kwargs = config.pop('execution_control')
            config = {
                module_name: config,
                'execution_control': exec_kwargs,
            }
        with open(config_file, 'w') as f:
            f.write(json.dumps(config, indent=2))

        logger.info(
            f'Created config file: {config_file}:'
            f'\n{pprint.pformat(config, indent=2)}'
        )

    @classmethod
    def blend(cls, kwargs):
        """Get config dictionary for nsrdb.cli.blend for standard NSRDB runs.

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for which to blend data
            files
        """
        if 'extent' in kwargs:
            config = cls._blend(kwargs)
            cls._write_config(
                config,
                f'config_blend_{config["extent"]}.json',
                module_name='blend',
            )

        elif kwargs['year'] > 2017 and 'extent' not in kwargs:
            kwargs.update({'extent': 'conus'})
            config = cls._blend(kwargs)
            cls._write_config(
                config, 'config_blend_conus.json', module_name='blend'
            )
            kwargs.update({'extent': 'full'})
            config = cls._blend(kwargs)
            cls._write_config(
                config, 'config_blend_full.json', module_name='blend'
            )
        else:
            config = cls._blend(kwargs)
            cls._write_config(
                config, 'config_blend_full.json', module_name='blend'
            )

    @classmethod
    def _collect_blend(cls, kwargs):
        """Create config for collecting blended files into a single output
        file.

        Note
        ----
        This is used to combine year < 2018 dset files into single files with
        all dsets. e.g. clearsky, irradiance, clouds, etc -> nsrdb_{year}.h5

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

        run_name = user_input.get('run_name', None)
        user_input['run_name'] = (
            run_name
            if run_name is not None
            else f'collect_blend_{user_input["extent"]}_{user_input["year"]}'
        )
        return user_input

    @classmethod
    def collect_blend(cls, kwargs):
        """Collect blended files into a single output file.

        Note
        ----
        This is used to combine year < 2018 dset files into single files with
        all dsets. e.g. clearsky, irradiance, clouds, etc -> nsrdb_{year}.h5

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for blend collection
        """

        config = cls._collect_blend(kwargs)
        cls._write_config(
            config, 'config_collect_blend.json', module_name='collect-blend'
        )

    @classmethod
    def _collect_aggregate(cls, kwargs):
        """Create config for aggregation collection

        Note
        ----
        This is used to collect single dset aggregated files into a single file
        with all dsets. e.g. e.g. clearsky, irradiance, clouds, etc ->
        nsrdb_{year}.h5

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
        run_name = user_input.get('run_name', None)
        user_input['run_name'] = (
            run_name
            if run_name is not None
            else f'collect_aggregate_{user_input["year"]}'
        )
        return user_input

    @classmethod
    def collect_aggregate(cls, kwargs):
        """Create config for aggregation collection

        Note
        ----
        This is used to collect single dset aggregated files into a single file
        with all dsets. e.g. e.g. clearsky, irradiance, clouds, etc ->
        nsrdb_{year}.h5

        Parameters
        ----------
        kwargs : dict
            Dictionary with keys specifying the case for aggregation collection
        """
        config = cls._collect_aggregate(kwargs)

        cls._write_config(
            config,
            'config_collect_aggregate.json',
            module_name='collect-aggregate',
        )
