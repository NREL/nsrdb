"""Common utilities for NSRDB all-sky module.
"""
from nsrdb.data_model import VarFactory


def scale_all_sky_outputs(all_sky_out, var_meta=None):
    """Perform safe scaling of all-sky outputs and change dtype.

    Parameters
    ----------
    all_sky_out : dict
        Namespace of all-sky irradiance output variables with the
        following keys:
            'clearsky_dhi'
            'clearsky_dni'
            'clearsky_ghi'
            'dhi'
            'dni'
            'ghi'
            'fill_flag'
    var_meta : str | pd.DataFrame | None
        CSV file or dataframe containing meta data for all NSRDB variables.
        Defaults to the NSRDB var meta csv in git repo.

    Returns
    -------
    all_sky_out : dict
        Same as input but all arrays are scaled and with final dtype.
    """
    for var, arr in all_sky_out.items():
        var_obj = VarFactory.get_base_handler(var, var_meta=var_meta)
        all_sky_out[var] = var_obj.scale_data(arr)

    return all_sky_out
