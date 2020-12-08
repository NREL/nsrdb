# -*- coding: utf-8 -*-
"""
NSRDB data pipeline architecture.
"""
import logging
from reV.pipeline.pipeline import Pipeline


logger = logging.getLogger(__name__)


class NsrdbPipeline(Pipeline):
    """NSRDB pipeline execution framework."""

    COMMANDS = ('data-model',
                'cloud-fill',
                'ml-cloud-fill',
                'all-sky',
                'daily-all-sky',
                'collect-data-model',
                'collect-daily',
                'collect-flist',
                'collect-final',
                )
