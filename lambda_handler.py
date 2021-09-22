"""
Lambda function handler
"""
from cloud_fs import FileSystem
from datetime import date
import json
from nsrdb import NSRDB
import os
from rex import init_logger, safe_json_load
import sys
import tempfile
import time


class LambdaArgs(dict):
    """
    Class to handle Lambda function args either from event or env vars
    """
    def __init__(self, event):
        self.update({k.lower(): v for k, v in os.environ.items()})

        if isinstance(event, str):
            event = safe_json_load(event)

        self.update(event)


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
    args = LambdaArgs(event)

    if args.get('verbose', False):
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    day = args.get('date', None)
    if not day:
        day = date.today().strftime("%Y%m%d")

    grid = args['grid']
    var_meta = args['var_meta']
    freq = args.get('freq', '5min')
    out_dir = args['out_dir']
    file_prefix = args['file_prefix']
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
    factory_kwargs = args.get("factory_kwargs", factory_kwargs)
    if isinstance(factory_kwargs, str):
        factory_kwargs = json.loads(factory_kwargs)

    temp_dir = args.get('temp_dir', "/tmp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    with tempfile.TemporaryDirectory(prefix=f'NSRDB_{day}_',
                                     dir=temp_dir) as temp_dir:
        logger = init_logger('nsrdb', log_level=log_level)
        logger.debug(f'event: {event}')
        logger.debug(f'context: {context}')
        logger.debug(f'NSRDB inputs:'
                     f'\nday = {day}'
                     f'\ngrid = {grid}'
                     f'\nfreq = {freq}'
                     f'\nvar_meta = {var_meta}'
                     f'\nfactory_kwargs = {factory_kwargs}')
        try:
            data_model = NSRDB.run_full(day, grid, freq,
                                        var_meta=var_meta,
                                        factory_kwargs=factory_kwargs,
                                        low_mem=args.get('low_mem', False),
                                        max_workers=1,
                                        log_level=None)

            fpath_out = os.path.join(temp_dir, fpath)
            dump_vars = ['ghi', 'dni', 'dhi',
                         'clearsky_ghi', 'clearsky_dni', 'clearsky_dhi']
            for v in dump_vars:
                data_model.dump(v, fpath_out, None, mode='a')

            dst_path = os.path.join(out_dir, fpath)
            FileSystem.copy(fpath_out, dst_path)
        except Exception:
            logger.exception('Failed to run NSRDB!')
            raise

    success = f'NSRDB ran successfully for {day}'
    success = {'statusCode': 200, 'body': json.dumps(success)}

    return success


if __name__ == '__main__':
    if len(sys.argv) > 1:
        event = safe_json_load(sys.argv[1])
        ts = time.time()
        handler(event, None)
        print('NSRDB lambda runtime: {:.4f} minutes'
              .format((time.time() - ts) / 60))
