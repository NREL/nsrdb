"""
GIF ify lambda function
"""
from cloud_fs import FileSystem
import json
import logging
import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from rex import Resource, safe_json_load, init_logger
import sys
import tempfile
import time

logger = logging.getLogger(__name__)

DIR_PATH = NSRDBDIR = os.path.dirname(os.path.realpath(__file__))
LOGO = os.path.join(DIR_PATH, 'nrel.png')
SP_FPATH = os.path.join(DIR_PATH,
                        'WorldCountries50mShapefile',
                        'ne_50m_admin_0_countries.shp')


class LambdaArgs(dict):
    """
    Class to handle Lambda function args either from event or env vars
    """

    def __init__(self, event):
        self.update({k.lower(): v for k, v in os.environ.items()})

        if isinstance(event, str):
            event = safe_json_load(event)

        self.update(event)


def make_images(h5_fpath, img_dir, **kwargs):
    """
    [summary]

    Parameters
    ----------
    h5_fpath : [type]
        [description]
    img_dir : [type]
        [description]
    dset : str, optional
        [description], by default 'ghi'
    cbar_label : str, optional
        [description], by default 'Global Horizontal Irradiance (W/m2)'
    counties_fpath : [type], optional
        [description], by default None
    logo : [type], optional
        [description], by default None
    local_timezone : int, optional
        [description], by default 0
    """
    np.seterr(all='ignore')

    fn_base = os.path.join(img_dir, 'image_{}.png')

    dset = kwargs.get('dset', 'ghi')
    cbar_label = kwargs.get('cbar_label',
                            'Global Horizontal Irradiance (W/m2)')
    counties_fpath = kwargs.get('counties_fpath', SP_FPATH)
    logo = kwargs.get('logo', LOGO)
    local_timezone = kwargs.get('local_timezone', 0)

    figsize = kwargs.get('figsize', (10, 4))
    dpi = kwargs.get('dpi', 300)
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    print_timestamp = kwargs.get('print_timestamp', True)
    logo_scale = kwargs.get('logo_scale', 0.7)

    alpha = kwargs.get('alpha', 1.0)
    face_color = kwargs.get('face_color', 'k')
    text_color = kwargs.get('text_color', '#969696')
    cmap = kwargs.get('cmap', 'YlOrRd')
    max_irrad = kwargs.get('max_irrad', 1000)
    min_irrad = kwargs.get('min_irrad', 0)

    shape_color = kwargs.get('shape_color', '#2f2f2f')
    shape_edge_color = kwargs.get('shape_edge_color', 'k')
    shape_line_width = kwargs.get('shape_line_width', 0.25)
    shape_aspect = kwargs.get('shape_aspect', 1.15)

    resource_map_interval = kwargs.get('resource_map_interval', 1)
    marker_size = kwargs.get('marker_size', 10)

    logger.info('reading data...')
    with Resource(h5_fpath) as res:
        time_index = res.time_index
        data = res[dset]
        mask = np.any(res['clearsky_ghi'] > 0, axis=1)
        meta = res.meta

    data = data[mask]
    time_index = time_index[mask]

    logger.info(f'{dset} stats: ', data.min(), data.mean(), data.max())
    logger.info('read complete')

    if xlim is None:
        xlim = (meta.longitude.min() * 1.05, meta.longitude.max() * 1.05)

    if ylim is None:
        ylim = (meta.latitude.min() * 1.05, meta.latitude.max() * 1.05)

    cmap_obj = plt.get_cmap(cmap)
    cNorm = colors.Normalize(vmin=min_irrad, vmax=max_irrad)
    cmx.ScalarMappable(norm=cNorm, cmap=cmap_obj)
    mpl.rcParams['text.color'] = text_color
    mpl.rcParams['axes.labelcolor'] = text_color

    i_fname = 0
    for ti, timestamp in enumerate(time_index):
        logger.info('working on ti {}, timestamp: {}'.format(ti, timestamp))

        fig = plt.figure(figsize=figsize, facecolor=face_color)
        ax = fig.add_subplot()
        fig.patch.set_facecolor(face_color)
        ax.patch.set_facecolor(face_color)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if counties_fpath is not None:
            import geopandas as gpd
            gdf = gpd.GeoDataFrame.from_file(counties_fpath)
            gdf = gdf.to_crs({'init': 'epsg:4326'})
            gdf.plot(ax=ax, color=shape_color, edgecolor=shape_edge_color,
                     linewidth=shape_line_width)
        else:
            gdf = None

        col_slice = slice(None)
        if resource_map_interval:
            col_slice = slice(None, None, resource_map_interval)

        ax.scatter(meta.longitude.values[col_slice],
                   meta.latitude.values[col_slice],
                   c=data[ti, col_slice],
                   alpha=alpha, cmap=cmap, s=marker_size,
                   vmin=min_irrad, vmax=max_irrad, linewidth=0,
                   marker='s')

        if gdf is not None:
            gdf.geometry.boundary.plot(ax=ax, color=None,
                                       edgecolor=shape_edge_color,
                                       linewidth=shape_line_width)
            if shape_aspect:
                ax.set_aspect(shape_aspect)

        n_cbar = 100
        b = ax.scatter([1e6] * n_cbar, [1e6] * n_cbar,
                       c=np.linspace(min_irrad, max_irrad, n_cbar),
                       s=0.00001, cmap=cmap)
        cbar = plt.colorbar(b)
        ticks = plt.getp(cbar.ax, 'yticklabels')
        plt.setp(ticks, color=text_color)
        cbar.ax.tick_params(which='minor', color=text_color)
        cbar.ax.tick_params(which='major', color=text_color)
        cbar.set_label(cbar_label, color=text_color)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if print_timestamp:
            local_timestamp = timestamp + pd.DateOffset(hours=local_timezone)
            plt.text(0.12, 0.885, str(local_timestamp),
                     transform=fig.transFigure)

        if logo is not None:
            if not isinstance(logo_scale, float):
                logo_scale = 1.0

            im = Image.open(logo)
            size = (int(logo_scale * im.size[0]), int(logo_scale * im.size[1]))
            im = im.resize(size)
            im = np.array(im).astype(float) / 255
            fig.figimage(im, 0, 0)

        fn = fn_base.format(i_fname)
        i_fname += 1
        fig.savefig(fn, dpi=dpi, bbox_inches='tight', facecolor=face_color)
        logger.info('saved: {}'.format(fn))
        plt.close()


def make_gif(fpath_out, img_dir, **kwargs):
    """
    [summary]

    Parameters
    ----------
    fpath_out : [type]
        [description]
    img_dir : [type]
        [description]
    kwargs
    """
    file_tag = kwargs.get('file_tag', 'image_')
    duration = kwargs.get('duration', None)
    filenames = [f for f in os.listdir(img_dir)
                 if f.endswith('.png')
                 and file_tag in f]
    if duration is None:
        duration = len(filenames)

    filenames = sorted(filenames,
                       key=lambda x: int(x.replace('.png', '').split('_')[-1]))
    img, *imgs = [Image.open(os.path.join(img_dir, fn)) for fn in filenames]
    img.save(fp=fpath_out, format='GIF', append_images=imgs,
             save_all=True, duration=duration, loop=0)
    logger.info('Saved: {}'.format(fpath_out))


def main(event, context):
    """
    [summary]

    Parameters
    ----------
    event : [type]
        [description]
    context : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    args = LambdaArgs(event)
    h5_path = "s3://" + args['Records']['object']['key']
    out_dir = args['out_dir']

    logger = init_logger(__name__)
    logger.debug(f'event: {event}')
    logger.debug(f'context: {context}')
    logger.debug(f'GIFFY inputs:'
                 f'\nh5_path = {h5_path}'
                 f'\nout_dir = {out_dir}')
    img_dir = args.get('img_dir', '/tmp')
    with tempfile.TemporaryDirectory(prefix='images_',
                                     dir=img_dir) as temp_dir:
        fname = os.path.basename(h5_path)
        h5_local = os.path.join(temp_dir, fname)
        FileSystem.copy(h5_path, h5_local)
        make_images(h5_local, temp_dir, **args)
        out_fpath = os.path.join(out_dir, fname.replace('.h5', '.gif'))
        make_gif(out_fpath, temp_dir, **args)

    success = f'GIFify of {h5_path} ran successfully for'
    success = {'statusCode': 200, 'body': json.dumps(success)}

    return success


if __name__ == "__main__":
    if len(sys.argv) > 1:
        event = safe_json_load(sys.argv[1])
        ts = time.time()
        main(event, None)
        logger.info('Giffy runtime: {:.4f} minutes'
                    .format((time.time() - ts) / 60))
