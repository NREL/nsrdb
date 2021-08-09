"""
Lambda function handler
"""
from cloud_fs import FileSystem
from datetime import date
import json
from nsrdb import NSRDB
from rex import init_logger
import os


def handler(event, context):
    """
    Wrapper for NSRDB to allow AWS Lambda invocation

    Parameters
    ----------
    event : dict
        The event dict that contains the parameters sent when the function
        is invoked.
    context : dict
        The context in which the function is called.
    """
    if event.get('verbose', False):
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    logger = init_logger('nsrdb', log_level=log_level)
    logger.debug(f'event: {event}')
    logger.debug(f'context: {context}')

    day = event.get('date', None)
    if not day:
        day = date.today().strftime("%Y%m%d")

    grid = event['grid']
    var_meta = event['var_meta']
    freq = event.get('freq', '5min')
    out_dir = event['out_dir']
    file_prefix = event['file_prefix']
    fpath = f'{file_prefix}-{day}.h5'

    factory_kwargs = {'air_temperature': {'handler': 'GfsVar'},
                      'surface_pressure': {'handler': 'GfsVar'},
                      'ozone': {'handler': 'GfsVar'},
                      'total_precipitable_water': {'handler': 'GfsVar'},
                      'alpha': {'handler': 'NrelVar'},
                      'aod': {'handler': 'NrelVar'},
                      'ssa': {'handler': 'NrelVar'},
                      }
    factory_kwargs = event.get("factory_kwargs", factory_kwargs)
    if isinstance(factory_kwargs, str):
        factory_kwargs = json.loads(factory_kwargs)

    logger.debug(f'NSRDB inputs:\nday = {day}\ngrid = {grid}\nfreq = {freq}'
                 f'\nvar_meta = {var_meta}'
                 f'\nfactory_kwargs = {factory_kwargs}')

    data_model = NSRDB.run_full(day, grid, freq,
                                var_meta=var_meta,
                                factory_kwargs=factory_kwargs,
                                log_level=log_level)

    temp_dir = event.get('temp_dir', "/tmp")
    if temp_dir is None:
        temp_dir = out_dir

    fpath_out = os.path.join(temp_dir, fpath)
    dump_vars = ['ghi', 'dni', 'dhi',
                 'clearsky_ghi', 'clearsky_dni', 'clearsky_dhi']
    for v in dump_vars:
        data_model.dump(v, fpath_out, None, mode='a')

    if temp_dir != out_dir:
        dst_path = os.path.join(out_dir, fpath)
        FileSystem.copy(fpath_out, dst_path)
