"""
Lambda function handler
"""
from cloud_fs import FileSystem
from datetime import date
from nsrdb import NSRDB
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
    print(context)
    day = event.get('date', None)
    if not day:
        day = date.today().strftime("%Y%m%d")

    grid = 's3://puerto-rico-nrel/puerto-rico/puerto_rico_2km_meta.csv'
    grid = event['grid']
    var_meta = 's3://puerto-rico-nrel/puerto-rico/puerto_rico_vars.csv'
    var_meta = event['var_meta']
    freq = event.get('freq', '5min')

    data_model = NSRDB.run_full(day, grid, freq, var_meta=var_meta,
                                log_level='DEBUG')

    temp_dir = event.get('temp_dir', None)
    out_dir = event['out_dir']
    if temp_dir is None:
        temp_dir = out_dir

    fpath = f'puerto-rico-{day}.h5'
    fpath_out = os.path.join(temp_dir, fpath)
    dump_vars = ['ghi', 'dni', 'dhi',
                 'clearsky_ghi', 'clearsky_dni', 'clearsky_dhi']
    for v in dump_vars:
        data_model.dump(v, fpath_out, None, mode='a')

    if temp_dir != out_dir:
        dst_path = os.path.join(out_dir, fpath)
        FileSystem.copy(fpath_out, dst_path)
