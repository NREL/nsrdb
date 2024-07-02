"""nsrdb base CLI class."""

import json
import logging
import os

import click
from gaps import Status
from gaps.config import load_config
from rex import safe_json_load
from rex.utilities.execution import SubprocessManager
from rex.utilities.fun_utils import get_fun_call_str
from rex.utilities.hpc import SLURM
from rex.utilities.loggers import init_mult

from nsrdb import NSRDB
from nsrdb.utilities import ModuleName

logger = logging.getLogger(__name__)
AVAILABLE_HARDWARE_OPTIONS = ('kestrel', 'slurm', 'local')

IMPORT_STR = (
    'from nsrdb.nsrdb import NSRDB;\n'
    'from nsrdb.file_handlers.collection import Collector;\n'
    'from rex import init_logger;\n'
    'import time;\n'
    'from gaps import Status;\n'
)


def get_doys(cmd_args):
    """Get the doy iterable from either the "doy_list" (prioritized)
    or "doy_range" input

    Parameters
    ----------
    cmd_args : dict
        Dictionary of kwargs from the nsrdb config file specifically for
        this command block.

    Returns
    -------
    doys : list | None
        List of day-of-year integers to iterate through. None if neither
        doy_list nor doy_range are found.
    """
    doy_list = cmd_args.get('doy_list', None)
    doy_range = cmd_args.get('doy_range', None)
    if doy_list is None and doy_range is None:
        return None

    if doy_list is None and doy_range is not None:
        doy_list = list(range(doy_range[0], doy_range[1]))

    return doy_list


class SlurmManager(SLURM):
    """GAPs-compliant SLURM manager"""

    def check_status_using_job_id(self, job_id):
        """Check the status of a job using the HPC queue and job ID.

        Parameters
        ----------
        job_id : int
            Job integer ID number.

        Returns
        -------
        status : str | None
            Queue job status string or `None` if not found.
        """
        return self.check_status(job_id=job_id)


class BaseCLI:
    """Base CLI class used to create CLI for modules in ModuleName"""

    @classmethod
    def from_config_preflight(
        cls, ctx, module_name, config, verbose, pipeline_step=None
    ):
        """Parse conifg file prior to running nsrdb module.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        module_name : str
            Module name string from :class:`nsrdb.utilities.ModuleName`.
        config : str
            Path to config file or dictionary providing all needed inputs to
            module_class
        verbose : bool
            Whether to run in verbose mode.
        pipeline_step : str, optional
            Name of the pipeline step being run. If ``None``, the
            ``pipeline_step`` will be set to the ``module_name``,
            mimicking old reV behavior. By default, ``None``.

        Returns
        -------
        config : dict
            Dictionary with module specifc inputs only.
        """
        ctx.ensure_object(dict)
        cls.check_module_name(module_name)

        cmd_args = cls.get_cmd_args(module_name=module_name, config=config)
        ctx.obj['STATUS_DIR'] = cmd_args['status_dir']
        ctx.obj['VERBOSE'] = verbose
        ctx.obj['OUT_DIR'] = cmd_args.get('outdir', cmd_args['status_dir'])
        ctx.obj['PIPELINE_STEP'] = pipeline_step or module_name
        mod_name = module_name.replace('-', '_')
        ctx.obj['LOG_DIR'] = os.path.join(
            cmd_args['status_dir'], 'logs', mod_name
        )
        os.makedirs(ctx.obj['LOG_DIR'], exist_ok=True)
        job_name = cmd_args.get('run_name', None)
        job_name = (
            f'{job_name}_{mod_name}' if job_name is not None else mod_name
        )
        ctx.obj['RUN_NAME'] = job_name
        ctx.obj['LOG_FILE'] = cmd_args.get(
            'log_file', os.path.join(ctx.obj['LOG_DIR'], job_name + '.log')
        )
        log_level = cmd_args.get('log_level', 'INFO')
        ctx.obj['LOG_ARG_STR'] = f'"nsrdb", log_level="{log_level}"'
        log_level = log_level == 'DEBUG'
        verbose = any([verbose, log_level, ctx.obj['VERBOSE']])

        init_mult(
            f'nsrdb_{mod_name}',
            ctx.obj['LOG_DIR'],
            modules=[__name__, 'nsrdb'],
            verbose=verbose,
        )

        cmd_args['log_file'] = ctx.obj['LOG_FILE']
        cmd_args['job_name'] = job_name
        return cmd_args

    @staticmethod
    def get_cmd_args(module_name, config):
        """Get module specific kwargs."""

        if not isinstance(config, dict):
            status_dir = os.path.dirname(os.path.abspath(config))
            config = load_config(config)
        else:
            status_dir = config.get('status_dir', './')

        exec_kwargs = config.get('execution_control', {})
        direct_args = config.get('direct', {})
        cmd_args = config.get(module_name, {})

        # replace any args with higher priority entries in command dict
        exec_kwargs.update(
            {k: v for k, v in cmd_args.items() if k in exec_kwargs}
        )
        direct_args.update(
            {k: v for k, v in cmd_args.items() if k in direct_args}
        )
        exec_kwargs['stdout_path'] = os.path.join(status_dir, 'stdout/')
        logger.debug(
            f'Found execution kwargs {exec_kwargs} for {module_name} module'
        )
        cmd_args.update(direct_args)
        cmd_args['status_dir'] = status_dir
        cmd_args['execution_control'] = exec_kwargs
        return cmd_args

    @classmethod
    def check_module_name(cls, module_name):
        """Make sure module_name is a valid member of the ModuleName class"""
        msg = (
            f'Module name must be in ModuleName class. Received {module_name}.'
        )
        assert module_name in ModuleName, msg

    @classmethod
    def kickoff_slurm_job(
        cls,
        ctx,
        module_name,
        cmd,
        option='kestrel',
        alloc='nsrdb',
        memory=None,
        walltime=4,
        feature=None,
        stdout_path='./stdout/',
    ):
        """Run nsrdb module on HPC via SLURM job submission.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        module_name : str
            Module name string from :class:`nsrdb.utilities.ModuleName`.
        cmd : str
            Command to be submitted in SLURM shell script. Example:
                'python -m nsrdb.cli <module_name> -c <config>'
        option : str
            Hardware option. e.g. "kestrel" or "slurm".
        alloc : str
            HPC project (allocation) handle. Example: 'nsrdb'.
        memory : int
            Node memory request in GB.
        walltime : float
            Node walltime request in hours.
        feature : str
            Additional flags for SLURM job. Format is "--qos=high" or
            "--depend=[state:job_id]". Default is None.
        stdout_path : str
            Path to print .stdout and .stderr files.
        pipeline_step : str, optional
            Name of the pipeline step being run. If ``None``, the
            ``pipeline_step`` will be set to the ``module_name``, mimicking old
            reV behavior. By default, ``None``.
        """
        cls.check_module_name(module_name)
        pipeline_step = ctx.obj['PIPELINE_STEP']
        name = ctx.obj['JOB_NAME']
        status_dir = ctx.obj['STATUS_DIR']
        slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
        if slurm_manager is None:
            slurm_manager = SlurmManager()
            ctx.obj['SLURM_MANAGER'] = slurm_manager

        status = Status.retrieve_job_status(
            status_dir,
            pipeline_step=pipeline_step,
            job_name=name,
            subprocess_manager=slurm_manager,
        )
        job_failed = 'fail' in str(status).lower()
        job_submitted = status != 'not submitted'

        msg = f'nsrdb {module_name} CLI failed to submit jobs!'
        if status == 'successful':
            msg = (
                f'Job "{name}" is successful in status json found in '
                f'"{status_dir}", not re-running.'
            )
        elif not job_failed and job_submitted and status is not None:
            msg = (
                f'Job "{name}" was found with status "{status}", not '
                'resubmitting'
            )
        else:
            job_info = f'{module_name}'
            if pipeline_step != module_name:
                job_info = f'{job_info} (pipeline step {pipeline_step!r})'
            logger.info(
                f'Running nsrdb {job_info} on SLURM with node name "{name}".'
            )
            out = slurm_manager.sbatch(
                cmd,
                alloc=alloc,
                memory=memory,
                walltime=walltime,
                feature=feature,
                name=name,
                stdout_path=stdout_path,
            )[0]
            if out:
                msg = (
                    f'Kicked off nsrdb {job_info} job "{name}" (SLURM jobid '
                    f'#{out}).'
                )

            # add job to nsrdb status file.
            Status.mark_job_as_submitted(
                status_dir,
                pipeline_step=pipeline_step,
                job_name=name,
                replace=True,
                job_attrs={
                    'job_id': out,
                    'outdir': ctx.obj['OUT_DIR'],
                    'log_file': ctx.obj['LOG_FILE'],
                    'hardware': option,
                },
            )

        click.echo(msg)
        logger.info(msg)

    @classmethod
    def kickoff_local_job(cls, ctx, module_name, cmd):
        """Run nsrdb module locally.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        module_name : str
            Module name string from :class:`nsrdb.utilities.ModuleName`.
        cmd : str
            Command to be submitted in shell script. Example:
                'python -m nsrdb.cli <module_name> -c <config>'
        """
        cls.check_module_name(module_name)
        pipeline_step = ctx.obj['PIPELINE_STEP']
        name = ctx.obj['JOB_NAME']
        status_dir = ctx.obj['STATUS_DIR']
        subprocess_manager = SubprocessManager

        status = Status.retrieve_job_status(
            status_dir, pipeline_step=pipeline_step, job_name=name
        )
        job_failed = 'fail' in str(status).lower()
        job_submitted = status != 'not submitted'

        msg = f'nsrdb {module_name} CLI failed to submit jobs!'
        if status == 'successful':
            msg = (
                f'Job "{name}" is successful in status json found in '
                f'"{status_dir}", not re-running.'
            )
        elif not job_failed and job_submitted and status is not None:
            msg = (
                f'Job "{name}" was found with status "{status}", not '
                'resubmitting'
            )
        else:
            job_info = f'{module_name}'
            if pipeline_step != module_name:
                job_info = f'{job_info} (pipeline step {pipeline_step!r})'
            if job_failed:
                logger.info('Previous run failed.')
            logger.info(
                f'Running nsrdb {job_info} locally with job name "{name}".'
            )
            Status.mark_job_as_submitted(
                status_dir=status_dir,
                pipeline_step=pipeline_step,
                job_name=name,
                replace=True,
            )
            subprocess_manager.submit(cmd)
            msg = f'Completed nsrdb {job_info} job "{name}".'

        click.echo(msg)
        logger.info(msg)

    @classmethod
    def get_status_cmd(cls, config, pipeline_step):
        """Append status file command to command for executing given module

        Parameters
        ----------
        config : dict
            nsrdb config with all necessary args and kwargs to run given
            module.
        pipeline_step : str
            Name of the pipeline step being run.
        cmd : str
            String including command to execute given module.

        Returns
        -------
        cmd : str
            Command string with status file command included if job_name is
            not None
        """
        cmd = ''
        job_name = config.get('job_name', None)
        status_dir = config.get('status_dir', None)
        if job_name is not None and status_dir is not None:
            status_file_arg_str = f'"{status_dir}", '
            status_file_arg_str += f'pipeline_step="{pipeline_step}", '
            status_file_arg_str += f'job_name="{job_name}", '
            status_file_arg_str += 'attrs=job_attrs'

            cmd += 'job_attrs = {};\n'.format(
                json.dumps(config)
                .replace('null', 'None')
                .replace('false', 'False')
                .replace('true', 'True')
            )
            cmd += 'job_attrs.update({"job_status": "successful"});\n'
            cmd += 'job_attrs.update({"time": t_elap});\n'
            cmd += f'Status.make_single_job_file({status_file_arg_str})'

        cmd += ";'\n"
        return cmd.replace('\\', '/')

    @classmethod
    def kickoff_job(cls, ctx, module_name, func, config, log_id=None):
        """Run nsrdb module either locally or on HPC.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        module_name : str
            Module name string from :class:`nsrdb.utilities.ModuleName`.
        func : Callable
            Function used to run module. e.g. `NSRDB.run_data_model()`
        config : dict
            nsrdb config with all necessary args and kwargs to run given
            module.
        log_id : str | None
            String id to append to base log file if this job is part of a multi
            job kickoff. None is used is this is just a single job.
        """
        log_file = (
            ctx.obj['LOG_FILE']
            if log_id is None
            else ctx.obj['LOG_FILE'].replace('.log', f'_{log_id}.log')
        )
        log_arg_str = f'{ctx.obj["LOG_ARG_STR"]}, log_file="{log_file}"'
        pipeline_step = ctx.obj['PIPELINE_STEP']
        exec_kwargs = config.get('execution_control', {})
        hardware_option = exec_kwargs.get('option', 'local')
        config['log_file'] = log_file
        fun_str = get_fun_call_str(func, config)

        cmd = (
            f"python -c '{IMPORT_STR}\n"
            't0 = time.time();\n'
            f'logger = init_logger({log_arg_str});\n'
            f'{fun_str};\n'
            't_elap = time.time() - t0;\n'
        )

        cmd += cls.get_status_cmd(config, pipeline_step)

        ctx.obj['JOB_NAME'] = config['job_name']
        if hardware_option == 'local':
            cls.kickoff_local_job(ctx, module_name, cmd)
        else:
            cls.kickoff_slurm_job(ctx, module_name, cmd, **exec_kwargs)

    @classmethod
    def kickoff_single(
        cls, ctx, module_name, func, config, verbose, pipeline_step=None
    ):
        """Kick off single job.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        module_name : str
            Module name string from :class:`nsrdb.utilities.ModuleName`.
        func : Callable
            Function used to run module. e.g. `NSRDB.run_data_model()`
        config : str | dict
            Path to nsrdb config file or a dictionary with all necessary args
            and kwargs to run given module
        verbose : bool
            Flag to turn on debug logging
        pipeline_step : str, optional
            Name of the pipeline step being run. If ``None``, the
            ``pipeline_step`` will be set to the ``module_name``,
            mimicking old reV behavior. By default, ``None``.
        """

        config = cls.from_config_preflight(
            ctx=ctx,
            module_name=module_name,
            config=config,
            verbose=verbose,
            pipeline_step=pipeline_step,
        )
        cls.kickoff_job(
            ctx=ctx, module_name=module_name, func=func, config=config
        )

    @classmethod
    def kickoff_multiday(
        cls, ctx, module_name, func, config, verbose, pipeline_step=None
    ):
        """Kick off jobs for multiple days.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        module_name : str
            Module name string from :class:`nsrdb.utilities.ModuleName`.
        func : Callable
            Function used to run module. e.g. `NSRDB.run_data_model()`
        config : str | dict
            Path to nsrdb config file or a dictionary with all necessary args
            and kwargs to run given module
        verbose : bool
            Flag to turn on debug logging
        pipeline_step : str, optional
            Name of the pipeline step being run. If ``None``, the
            ``pipeline_step`` will be set to the ``module_name``,
            mimicking old reV behavior. By default, ``None``.
        """

        config_dict = cls.from_config_preflight(
            ctx=ctx,
            module_name=module_name,
            config=config,
            verbose=verbose,
            pipeline_step=pipeline_step,
        )
        doys = get_doys(config_dict)
        if doys is None:
            doys = get_doys(safe_json_load(config)[ModuleName.DATA_MODEL])
        if doys is None:
            msg = (
                'NSRDB data-model config needs either the "doy_list" or '
                '"doy_range" input.'
            )
            logger.error(msg)
            raise KeyError(msg)

        for doy in doys:
            date = NSRDB.doy_to_datestr(config_dict['year'], doy)
            log_id = f'{date}_{str(doy).zfill(3)}'
            config_dict['date'] = date
            config_dict['job_name'] = f'{ctx.obj["RUN_NAME"]}_{log_id}'
            config_dict['doy'] = doy

            cls.kickoff_job(
                ctx,
                module_name=module_name,
                func=func,
                config=config_dict,
                log_id=doy,
            )

    @classmethod
    def kickoff_multichunk(
        cls, ctx, module_name, func, config, verbose, pipeline_step=None
    ):
        """Kick off jobs for multiple chunks.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        module_name : str
            Module name string from :class:`nsrdb.utilities.ModuleName`.
        func : Callable
            Function used to run module. e.g. `NSRDB.gap_fill_clouds()`
        config : str | dict
            Path to nsrdb config file or a dictionary with all necessary args
            and kwargs to run given module
        verbose : bool
            Flag to turn on debug logging
        pipeline_step : str, optional
            Name of the pipeline step being run. If ``None``, the
            ``pipeline_step`` will be set to the ``module_name``,
            mimicking old reV behavior. By default, ``None``.
        """
        config = cls.from_config_preflight(
            ctx=ctx,
            module_name=module_name,
            config=config,
            verbose=verbose,
            pipeline_step=pipeline_step,
        )
        config['n_chunks'] = config.get('n_chunks', 1)

        for i_chunk in range(config['n_chunks']):
            config['i_chunk'] = i_chunk
            config['job_name'] = f'{ctx.obj["RUN_NAME"]}_{i_chunk}'

            cls.kickoff_job(
                ctx=ctx,
                module_name=module_name,
                func=func,
                config=config,
                log_id=i_chunk,
            )
