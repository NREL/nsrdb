# -*- coding: utf-8 -*-
"""Factory pattern for retrieving NSRDB data source handlers."""

import logging

from nsrdb.data_model.base_handler import AncillaryVarHandler
from nsrdb.data_model.albedo import AlbedoVar
from nsrdb.data_model.asymmetry import AsymVar
from nsrdb.data_model.merra import MerraVar, DewPoint, RelativeHumidity
from nsrdb.data_model.clouds import (CloudVar, CloudVarSingleH5,
                                     CloudVarSingleNC)
from nsrdb.data_model.solar_zenith_angle import SolarZenithAngle


logger = logging.getLogger(__name__)


class VarFactory:
    """Factory pattern to retrieve ancillary variable helper objects."""

    # mapping of NSRDB variable names to helper objects
    MAPPING = {'asymmetry': AsymVar,
               'air_temperature': MerraVar,
               'surface_albedo': AlbedoVar,
               'alpha': MerraVar,
               'aod': MerraVar,
               'cloud_type': CloudVar,
               'cld_opd_dcomp': CloudVar,
               'cld_reff_dcomp': CloudVar,
               'cld_press_acha': CloudVar,
               'dew_point': DewPoint,
               'ozone': MerraVar,
               'relative_humidity': RelativeHumidity,
               'solar_zenith_angle': SolarZenithAngle,
               'specific_humidity': MerraVar,
               'ssa': MerraVar,
               'surface_pressure': MerraVar,
               'total_precipitable_water': MerraVar,
               'wind_speed': MerraVar,
               }

    NO_ARGS = ('relative_humidity', 'dew_point', 'solar_zenith_angle')

    def get(self, var_name, *args, **kwargs):
        """Get a processing variable instance for the given var name.

        Parameters
        ----------
        var_name : str
            NSRDB variable name.
        *args : list
            List of positional args for instantiation of ancillary var.
        **kwargs : dict
            List of keyword args for instantiation of ancillary var.

        Returns
        -------
        instance : ancillary object
            Instantiated ancillary variable helper object (AsymVar, MerraVar).
        """

        # ensure var is in the available handlers
        if var_name in self.MAPPING:

            if var_name in self.NO_ARGS:
                kwargs = {}

            # kwarg reduction for non-cloud vars
            elif 'cld' not in var_name and 'cloud' not in var_name:
                del_list = ('extent', 'path', 'dsets')
                kwargs = {k: v for k, v in kwargs.items() if k not in del_list}

            # single creational statement to init handler
            return self.MAPPING[var_name](*args, **kwargs)

        else:
            raise KeyError('Did not recognize "{}" as an available NSRDB '
                           'variable. The following variables are available: '
                           '{}'.format(var_name, list(self.MAPPING.keys())))

    @staticmethod
    def get_base_handler(*args, **kwargs):
        """Get the base ancillary variable handler to parse var meta.

        Parameters
        ----------
        *args : list
            List of positional args for instantiation of ancillary var.
        **kwargs : dict
            List of keyword args for instantiation of ancillary var.

        Returns
        -------
        instance : AncillaryVarHandler
            Instantiated base ancillary variable helper object.
        """

        return AncillaryVarHandler(*args, **kwargs)

    @staticmethod
    def get_cloud_handler(fpath, dsets=None):
        """Get a cloud data file handler object.

        Parameters
        ----------
        fpath : str
            Full file path to cloud data file, must be either a .h5 or .nc
        dsets : NoneType | list | tuple
            Datasets of interest from cloud data file. None means only the
            grid data is of itnerest.

        Returns
        -------
        out : CloudVarSingleH5/NC
            Instantiated CloudVarSingle object appropraite for the .h5 or .nc
            fpath.
        """

        if fpath.endswith('.h5'):
            return CloudVarSingleH5(fpath, dsets=dsets)
        elif fpath.endswith('.nc'):
            return CloudVarSingleNC(fpath, dsets=dsets)
        else:
            raise TypeError('Did not recognize cloud file type as .nc or '
                            '.h5: {}'.format(fpath))
