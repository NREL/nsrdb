"""
Lambda function handler
"""
from cloud_fs import FileSystem
from datetime import date
import json
from nsrdb import NSRDB
import os
from rex import init_logger, safe_json_load
import shutil
import tempfile


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
    if isinstance(event, str):
        event = safe_json_load(event)

    aws_access_key = event.get('AWS_ACCESS_KEY_ID')
    if aws_access_key:
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key

    aws_secret_key = event.get('AWS_SECRET_ACCESS_KEY')
    if aws_secret_key:
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key

    if event.get('verbose', False):
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

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
                      'alpha': {'handler': 'NrelVar'},
                      'aod': {'handler': 'NrelVar'},
                      'dew_point': {'handler': 'GfsDewPoint'},
                      'ozone': {'handler': 'GfsVar'},
                      'relative_humidity': {'handler': 'GfsVar'},
                      'ssa': {'handler': 'NrelVar'},
                      'surface_pressure': {'handler': 'GfsVar'},
                      'total_precipitable_water': {'handler': 'GfsVar'},
                      'wind_direction': {'handler': 'GfsVar'},
                      'wind_speed': {'handler': 'GfsVar'},
                      }
    factory_kwargs = event.get("factory_kwargs", factory_kwargs)
    if isinstance(factory_kwargs, str):
        factory_kwargs = json.loads(factory_kwargs)

    temp_dir = event.get('temp_dir', "/tmp")
    if temp_dir is None:
        temp_dir = out_dir
    else:
        temp_dir = tempfile.mkdtemp(prefix=f'NSRDB_{day}_', dir=temp_dir)

    logger = init_logger('nsrdb', log_level=log_level)
    aws_key = os.environ.get('AWS_ACCESS_KEY_ID')
    logger.debug(f'AWS_ACCESS_KEY_ID: {aws_key}')
    logger.debug(f'event: {event}')
    logger.debug(f'context: {context}')

    logger.debug(f'NSRDB inputs:\nday = {day}\ngrid = {grid}\nfreq = {freq}'
                 f'\nvar_meta = {var_meta}'
                 f'\nfactory_kwargs = {factory_kwargs}')

    try:
        data_model = NSRDB.run_full(day, grid, freq,
                                    var_meta=var_meta,
                                    factory_kwargs=factory_kwargs,
                                    low_mem=event.get('low_mem', False),
                                    max_workers=1,
                                    log_level=None)

        fpath_out = os.path.join(temp_dir, fpath)
        dump_vars = ['ghi', 'dni', 'dhi',
                     'clearsky_ghi', 'clearsky_dni', 'clearsky_dhi']
        for v in dump_vars:
            data_model.dump(v, fpath_out, None, mode='a')

        if temp_dir != out_dir:
            dst_path = os.path.join(out_dir, fpath)
            FileSystem.copy(fpath_out, dst_path)
    except Exception:
        logger.exception('Failed to run NSRDB!')
        raise
    finally:
        if temp_dir != out_dir:
            shutil.rmtree(temp_dir)
