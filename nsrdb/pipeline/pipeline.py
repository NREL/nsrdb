# -*- coding: utf-8 -*-
"""
NSRDB data pipeline architecture.
"""
import logging

from reV.pipeline.pipeline import Pipeline
from rex.utilities.loggers import init_logger

from nsrdb.pipeline.config import NsrdbPipelineConfig

logger = logging.getLogger(__name__)


class NsrdbPipeline(Pipeline):
    """NSRDB pipeline execution framework."""

    CMD_BASE = 'python -m nsrdb.cli config -c {fp_config} -cmd {command}'

    COMMANDS = ('data-model',
                'cloud-fill',
                'ml-cloud-fill',
                'all-sky',
                'daily-all-sky',
                'collect-data-model',
                'collect-daily',
                'collect-flist',
                'collect-final'
                )

    def __init__(self, pipeline, monitor=True, verbose=False):
        """
        Parameters
        ----------
        pipeline : str | dict
            Pipeline config file path or dictionary.
        monitor : bool
            Flag to perform continuous monitoring of the pipeline.
        verbose : bool
            Flag to submit pipeline steps with -v flag for debug logging
        """
        self.monitor = monitor
        self.verbose = verbose
        self._config = NsrdbPipelineConfig(pipeline)
        self._run_list = self._config.pipeline
        self._init_status()

        # init logger for pipeline module if requested in input config
        if 'logging' in self._config:
            init_logger('nsrdb.pipeline', **self._config.logging)
            init_logger('reV.pipeline', **self._config.logging)
