"""Config creation class for CONUS / Full Disc NSRDB runs."""

import calendar
import copy
import json
import logging
import os
import pprint

from nsrdb import CONFIGDIR
from nsrdb.aggregation.aggregation import NSRDB_2018
from nsrdb.utilities.file_utils import get_format_keys, str_replace_dict

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

DEFAULT_META_DIR = '/projects/pxs/reference_grids/'


BASE_KWARGS = {
    'basename': 'nsrdb',
    'out_dir': './',
    'execution_control': DEFAULT_EXEC_CONFIG,
    'meta_dir': DEFAULT_META_DIR,
}

MAIN_KWARGS = {
    **BASE_KWARGS,
    'freq': '30min',
    'spatial': '4km',
    'satellite': 'east',
    'extent': 'full',
}

BLEND_KWARGS = {
    **BASE_KWARGS,
    'file_tag': 'all',
    'extent': 'full',
    'main_dir': '../',
}

AGG_KWARGS = {
    **BASE_KWARGS,
    'full_spatial': '2km',
    'conus_spatial': '2km',
    'final_spatial': '4km',
    'data_dir': './',
    'full_freq': '10min',
    'conus_freq': '5min',
    'final_freq': '30min',
    'n_chunks': 32,
    'source_priority': ['conus', 'full_disk'],
}

COLLECT_AGG_KWARGS = {
    **BASE_KWARGS,
    'final_spatial': '4km',
    'final_freq': '30min',
    'max_workers': 10,
}


logger = logging.getLogger(__name__)


class CreateConfigs:
    """Collection of methods to create config files for NSRDB module inputs for
    standard CONUS / Full Disc runs."""

    MAIN_RUN_NAME = '{basename}_{satellite}_{extent}_{year}_{spatial}_{freq}'
    BLEND_RUN_NAME = '{basename}_{extent}_{year}_blend'
    AGG_RUN_NAME = '{basename}_{year}_aggregate'
    COLLECT_AGG_RUN_NAME = '{basename}_{year}_collect_aggregate'
    COLLECT_BLEND_RUN_NAME = '{basename}_{extent}_{year}_collect_blend'

    @classmethod
    def main(cls, kwargs):
        """Modify config files with specified parameters

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameters including year, basename, satellite,
            extent, freq, spatial, meta_file, doy_range
        """
        config = copy.deepcopy(MAIN_KWARGS)
        config.update(kwargs)
        config['out_dir'] = os.path.abspath(config['out_dir'])
        os.makedirs(config['out_dir'], exist_ok=True)

        extent_tag_map = {'full': 'RadF', 'conus': 'RadC'}
        lon_seam_map = {'full': -105, 'conus': -113}
        config['extent_tag'] = extent_tag_map[config['extent']]
        lon_seam = lon_seam_map[config['extent']]

        if config['year'] != 2018:
            meta_file = f'nsrdb_meta_{config["spatial"]}'

            if config['year'] > 2018:
                meta_file += f'_{config["extent"]}'

            meta_file += f'_{config["satellite"]}_{lon_seam}.csv'
            config['meta_file'] = meta_file

        if config.get('doy_range', None) is None:
            if calendar.isleap(config['year']):
                config['doy_range'] = [1, 367]
            else:
                config['doy_range'] = [1, 366]

        config['start_doy'] = config['doy_range'][0]
        config['end_doy'] = config['doy_range'][1]

        config['out_dir'] = os.path.join(
            config['out_dir'], cls._get_run_name(config)
        )

        cls._update_run_templates(config)

        run_file = os.path.join(config['out_dir'], 'run.sh')
        with open(run_file, 'w') as f:
            f.write('python -m nsrdb.cli pipeline -c config_pipeline.json')

        logger.info(f'Saved run script: {run_file}.')

    @classmethod
    def main_all(cls, kwargs):
        """Modify config files for all domains with specified parameters.

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameters including year, basename, satellite,
            extent, freq, spatial, meta_file, doy_range
        """
        out_dir = os.path.abspath(kwargs.get('out_dir', './'))
        if kwargs['year'] < 2018:
            full_kws = {'spatial': '4km', 'extent': 'full', 'freq': '30min'}
            kwargs_list = [
                {**full_kws, 'satellite': 'east'},
                {**full_kws, 'satellite': 'west'},
            ]
        elif kwargs['year'] == 2018:
            kwargs_list = [
                {'extent': 'full', 'satellite': 'east', **NSRDB_2018['east']},
                {'extent': 'full', 'satellite': 'west', **NSRDB_2018['west']},
                {
                    'extent': 'conus',
                    'satellite': 'east',
                    **NSRDB_2018['conus'],
                },
            ]
        else:
            full_kws = {'spatial': '2km', 'extent': 'full', 'freq': '10min'}
            conus_kws = {'extent': 'conus', 'freq': '5min'}
            kwargs_list = [
                {**full_kws, 'satellite': 'east'},
                {**full_kws, 'satellite': 'west'},
                {**conus_kws, 'satellite': 'east'},
                {**conus_kws, 'satellite': 'west'},
            ]
        run_cmd = ''
        for kws in kwargs_list:
            kwargs.update(kws)
            run_cmd += f'cd {cls._get_run_name(kwargs)}; bash run.sh; cd ../; '
            cls.main(kwargs)

        run_file = os.path.join(out_dir, 'run.sh')
        with open(run_file, 'w') as f:
            f.write(run_cmd)

        logger.info(f'Saved run script: {run_file}.')

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
    def _update_run_templates(cls, config):
        """Replace format keys and dictionary keys in config templates with
        user input values."""

        logger.info(
            'Updating NSRDB run templates with config:\n'
            f'{pprint.pformat(config, indent=2)}'
        )

        template = (
            PRE2018_CONFIG_TEMPLATE
            if int(config['year']) < 2018
            else POST2017_CONFIG_TEMPLATE
        )
        with open(template, encoding='utf-8') as s:
            s = s.read()

        config_dict = json.loads(str_replace_dict(s, config))
        config_dict.update(
            {k: v for k, v in config.items() if k in config_dict}
        )
        cls._write_config(
            config_dict,
            os.path.join(config['out_dir'], 'config_nsrdb.json'),
        )

        with open(PIPELINE_CONFIG_TEMPLATE, encoding='utf-8') as s:
            s = s.read()

        config_dict = json.loads(str_replace_dict(s, config))

        cls._write_config(
            config_dict,
            os.path.join(config['out_dir'], 'config_pipeline.json'),
        )

    @classmethod
    def _get_run_name(cls, config, run_type='main'):
        """Get name of run for given main run input."""
        config.update(
            {k: v for k, v in MAIN_KWARGS.items() if k not in config}
        )
        pattern_dict = {
            'main': cls.MAIN_RUN_NAME,
            'blend': cls.BLEND_RUN_NAME,
            'aggregate': cls.AGG_RUN_NAME,
            'collect-aggregate': cls.COLLECT_AGG_RUN_NAME,
            'collect-blend': cls.COLLECT_BLEND_RUN_NAME,
        }
        pattern = pattern_dict[run_type]
        keys = get_format_keys(pattern)
        return pattern.format(**{k: v for k, v in config.items() if k in keys})

    @classmethod
    def post(cls, kwargs):
        """Create all post processing config files for blending / aggregation /
        collection."""

        pipeline_config = {'pipeline': []}
        out_dir = os.path.abspath(kwargs.get('out_dir', './'))
        post_proc_dir = os.path.join(out_dir, 'post_proc')
        kwargs['main_dir'] = out_dir

        if kwargs['year'] > 2018:
            kwargs_list = [{'extent': 'conus'}, {'extent': 'full'}, None, None]
            fnames = [
                'config_blend_conus.json',
                'config_blend_full.json',
                'config_aggregate.json',
                'config_collect_aggregate.json',
            ]
            step_names = [
                'blend-conus',
                'blend-full',
                'aggregate',
                'collect-aggregate',
            ]
            mod_names = ['blend', 'blend', *step_names[2:]]
        elif kwargs['year'] == 2018:
            kwargs_list = [None, None]
            fnames = [
                'config_aggregate.json',
                'config_collect_aggregate.json',
            ]
            step_names = ['aggregate', 'collect-aggregate']
            mod_names = step_names
        else:
            kwargs_list = [None, None]
            fnames = [
                'config_blend.json',
                'config_collect_blend.json',
            ]
            step_names = ['blend', 'collect-blend']
            mod_names = step_names

        for kws, fname, step_name, mod_name in zip(
            kwargs_list, fnames, step_names, mod_names
        ):
            func = getattr(cls, f'_{mod_name.replace("-", "_")}')
            config = func(kwargs if kws is None else {**kwargs, **kws})
            if mod_name == 'blend':
                config['out_dir'] = os.path.join(
                    post_proc_dir, config['run_name']
                )
            cls._write_config(
                config,
                os.path.join(post_proc_dir, fname),
                module_name=mod_name,
            )
            pipeline_config['pipeline'].append(
                {step_name: fname, 'command': mod_name}
            )
        cls._write_config(
            pipeline_config,
            os.path.join(post_proc_dir, 'config_pipeline_post.json'),
        )
        run_file = os.path.join(post_proc_dir, 'run.sh')
        with open(run_file, 'w') as f:
            f.write(
                'python -m nsrdb.cli pipeline -c config_pipeline_post.json'
            )

        logger.info(f'Saved run script: {run_file}.')

    @classmethod
    def _get_agg_entry(cls, config, extent):
        """Get entry in the aggregate data argument for a given extent."""
        meta_file = 'nsrdb_meta_{res}_{extent}.csv'
        tree_file = 'kdtree_nsrdb_meta_{res}_{extent}.pkl'

        if extent == 'final':
            source_priority = config.pop('source_priority')
            final_sub_dir = (
                f'{config.get("basename", "nsrdb")}_{config["final_spatial"]}'
                f'_{config["final_freq"]}'
            )
            return {
                'data_sub_dir': final_sub_dir,
                'fout': f'nsrdb_{config["year"]}.h5',
                'tree_file': tree_file.replace('_{extent}', '').format(
                    res=config['final_spatial']
                ),
                'meta_file': meta_file.replace('_{extent}', '').format(
                    res=config['final_spatial']
                ),
                'spatial': f'{config["final_spatial"]}',
                'temporal': f'{config["final_freq"]}',
                'source_priority': source_priority,
            }

        return {
            'data_sub_dir': cls._get_run_name(
                {**config, 'extent': extent}, run_type='blend'
            ),
            'tree_file': tree_file.format(
                res=config[f'{extent}_spatial'], extent=extent
            ),
            'meta_file': meta_file.format(
                res=config[f'{extent}_spatial'], extent=extent
            ),
            'spatial': config[f'{extent}_spatial'],
            'temporal': config[f'{extent}_freq'],
        }

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
        config = copy.deepcopy(AGG_KWARGS)
        config.update(kwargs)

        if config['year'] == 2018:
            data = NSRDB_2018

        else:
            data = {
                'full_disk': cls._get_agg_entry(config, extent='full'),
                'conus': cls._get_agg_entry(config, extent='conus'),
                'final': cls._get_agg_entry(config, extent='final'),
            }

        config['data'] = data
        config['run_name'] = config.get(
            'run_name', cls._get_run_name(config, run_type='aggregate')
        )
        return config

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
            config,
            os.path.join(
                kwargs.get('out_dir', './'),
                'config_aggregate.json',
            ),
            module_name='aggregate',
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
        config = copy.deepcopy(BLEND_KWARGS)
        config.update(kwargs)

        if config['year'] > 2017:
            config['spatial'] = '2km'
            if config['extent'] == 'full':
                config['freq'] = '10min'
            else:
                config['freq'] = '5min'

        map_col_map = {'full': 'gid_full', 'conus': 'gid_full_conus'}
        config['map_col'] = map_col_map[config['extent']]

        lon_seam_map = {'full': -105, 'conus': -113}
        config['lon_seam'] = lon_seam_map[config['extent']]

        meta_file = f'nsrdb_meta_{config["spatial"]}'

        if config['year'] > 2017:
            meta_file += f'_{config["extent"]}'

        meta_file += '.csv'
        config['meta_file'] = os.path.join(config['meta_dir'], meta_file)

        config['east_dir'] = os.path.join(
            config['main_dir'],
            cls._get_run_name({'satellite': 'east', **config}),
            'final',
        )
        config['west_dir'] = os.path.join(
            config['main_dir'],
            cls._get_run_name({'satellite': 'west', **config}),
            'final',
        )
        config['run_name'] = config.get(
            'run_name', cls._get_run_name(config, run_type='blend')
        )

        return config

    @classmethod
    def _write_config(cls, config, config_file, module_name=None):
        """Write config to .json file."""

        config_dir = os.path.dirname(config_file)
        config_dir = config_dir if config_dir else config.get('out_dir', './')
        for k, v in config.items():
            if '_dir' in k:
                config[k] = os.path.abspath(v)

        os.makedirs(config_dir, exist_ok=True)

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
        config = cls._blend(kwargs)
        cls._write_config(
            config,
            os.path.join(
                kwargs.get('out_dir', './'),
                f'config_blend_{config["extent"]}.json',
            ),
            module_name='blend',
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

        config = copy.deepcopy(BASE_KWARGS)
        config.update(kwargs)

        config['meta'] = os.path.join(
            config['meta_dir'], f'nsrdb_meta_{config["spatial"]}.csv'
        )
        config['collect_dir'] = cls._get_run_name(config, run_type='blend')
        config['collect_tag'] = config['collect_dir'].replace('_blend', '')
        config['fout'] = os.path.join(
            f'{config["out_dir"]}',
            f'{config["basename"]}_{config["year"]}.h5',
        )

        config['run_name'] = config.get(
            'run_name', cls._get_run_name(config, run_type='collect-blend')
        )
        return config

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
            config,
            os.path.join(
                kwargs.get('out_dir', './'), 'config_collect_blend.json'
            ),
            module_name='collect-blend',
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
        config = copy.deepcopy(COLLECT_AGG_KWARGS)
        config.update(kwargs)

        meta_file = f'nsrdb_meta_{config["final_spatial"]}.csv'
        config['meta_final'] = os.path.join(config['meta_dir'], meta_file)
        collect_dir = f'nsrdb_{config["final_spatial"]}_{config["final_freq"]}'
        collect_tag = f'{config["basename"]}_'
        config['collect_dir'] = collect_dir
        config['collect_tag'] = collect_tag
        config['fout'] = os.path.join(
            f'{config["out_dir"]}',
            f'{config["basename"]}_{config["year"]}.h5',
        )
        config['run_name'] = config.get(
            'run_name', cls._get_run_name(config, run_type='collect-aggregate')
        )
        return config

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
            os.path.join(
                kwargs.get('out_dir', './'), 'config_collect_aggregate.json'
            ),
            module_name='collect-aggregate',
        )
