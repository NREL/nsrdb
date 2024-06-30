"""NSRDB utilities."""

from enum import Enum


class ModuleName(str, Enum):
    """A collection of the module names available in nsrdb.  Each module name
    should match the name of the click command that will be used to invoke its
    respective cli. As of 6/29/2024, this means that all commands are lowercase
    with underscores replaced by dashes.
    """

    DATA_MODEL = 'data-model'
    CLOUD_FILL = 'cloud-fill'
    ML_CLOUD_FILL = 'ml-cloud-fill'
    ALL_SKY = 'all-sky'
    DAILY_ALL_SKY = 'daily-all-sky'
    BLEND = 'blend'
    AGGREGATE = 'aggregate'
    COLLECT_DATA_MODEL = 'collect-data-model'
    COLLECT_DAILY = 'collect-daily'
    COLLECT_FLIST = 'collect-flist'
    COLLECT_FINAL = 'collect-final'
    COLLECT_BLENDED = 'collect-blended'
    COLLECT_AGG = 'collect-agg'

    def __str__(self):
        return self.value

    def __format__(self, format_spec):
        return str.__format__(self.value, format_spec)

    @classmethod
    def all_names(cls):
        """All module names.

        Returns
        -------
        set
            The set of all module name strings.
        """
        # pylint: disable=no-member
        return {v.value for v in cls.__members__.values()}
