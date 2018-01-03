# encoding: utf-8

"""
Copyright (C) 2014-2018 OceanDataLab

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import logging
import traceback
try:
    # Use of subprocess32 (backport from Python 3.2) is encouraged by Python
    # official documentation.
    # See https://docs.python.org/2/library/subprocess.html
    import subprocess32 as subprocess
except ImportError:
    import subprocess

from syntool_ingestor.interfaces import IFormatterPlugin

logger = logging.getLogger(__name__)

class RasterPlugin(IFormatterPlugin):
    """ """

    def fix_bands(self, input_path, workspace, cfg, src_meta):
        """ """
        output_path = os.path.join(workspace, 'fix_bands.vrt')
        if 2 == src_meta.get('nb_bands', 0) and src_meta.get('ispaletted', False):
            band = 1
            call = ['gdal_translate',
                    '-b', '{}'.format(band),
                    '-of', 'VRT',
                    input_path,
                    output_path]

            logger.debug(' '.join(call))
            try:
                subprocess.check_call(call)
            except subprocess.CalledProcessError as e:
                logger.error('Could not fix bands.')
                logger.debug(traceback.print_exc())
                raise

            # Update metadata
            src_meta['nb_bands'] = 1
            if 0 < len(src_meta['nodatavalues']):
                src_meta['nodatavalues'] = [src_meta['nodatavalues'][0]]

            return output_path

        # By default
        return input_path

    def warp(self, input_path, workspace, cfg, src_meta):
        """Warp in the portal projection and extent.
        Output size is fixed to 2048x2048 by default (warp-size option)."""

        input_proj = cfg.get('input_proj')
        output_proj = cfg.get('output_proj')
        viewport = cfg['viewport'].split(' ')
        resampling = cfg['output_options'].get('resampling', 'bilinear')
        warp_size = cfg['output_options'].get('warp-size', '2048')
        if 'x' in warp_size:
            x_size, y_size = warp_size.split('x', 1)
        else:
            x_size, y_size = warp_size, warp_size

        warped_path = os.path.join(workspace, 'warp.vrt')

        call = ['gdalwarp',
                '-s_srs', 'epsg:{}'.format(input_proj),
                '-t_srs', 'epsg:{}'.format(output_proj),
                '-te', viewport[0], viewport[1], viewport[2], viewport[3],
                '-ts', x_size, y_size,
                '-wo', 'SAMPLE_GRID=YES',
                '-wo', 'SOURCE_EXTRA=100',
                '-wo', 'NUM_THREADS=ALL_CPUS',
                '-r', resampling,
                '-multi',
                '-of', 'VRT',
                '-overwrite']

        if 0 < len(src_meta['nodatavalues']):
            nodatavalues = src_meta['nodatavalues']
            call.extend(['-dstnodata', ' '.join(map(str, nodatavalues))])

        if src_meta.get('use_gcp', False):
            call.append('-tps')

        call.extend([input_path, warped_path])

        logger.debug(' '.join(call))
        try:
            subprocess.check_call(call)
        except subprocess.CalledProcessError as e:
            logger.error('Could not warp.')
            logger.debug(traceback.print_exc())
            raise
        return warped_path

    def translate_to_png(self, input_path, workspace):
        """ """
        png_path = os.path.join(workspace, 'imageLayer.png')

        call = ['gdal_translate',
                '-of', 'PNG',
                input_path,
                png_path]

        logger.debug(' '.join(call))
        try:
            subprocess.check_call(call)
        except subprocess.CalledProcessError as e:
            logger.error('Could not translate to png.')
            logger.debug(traceback.print_exc())
            raise
        return png_path
    
    def can_handle(self, target_format):
        """ """
        return 'raster' == target_format

    def get_output_id(self, cfg):
        """ """
        return '_raster'

    def get_representation_type(self, cfg):
        """ """
        return 'IMAGE'

    def create_representation(self, input_path, input_data, workspace, cfg, src_meta):
        """ """
        temporary_files = []

        # Fix bands
        bands_ok_path = self.fix_bands(input_path, workspace, cfg, src_meta)
        temporary_files.append(bands_ok_path)

        # Warp
        warp_path = self.warp(bands_ok_path, workspace, cfg, src_meta)
        temporary_files.append(warp_path)

        # Translate to png
        output_path = self.translate_to_png(warp_path, workspace)

        # Clean
        if not cfg.get('keep_intermediary_files', False):
            to_remove = filter( lambda x: x!= input_path and os.path.exists(x)
                              , temporary_files)
            map(os.remove, to_remove)
            logger.debug('These temporary files have been removed: {}'.format(to_remove))

        resolutions = []
        zooms = [0]

        # WARNING : next has to be modified if warp is not anymore global
        west, south, east, north = map(float, cfg['viewport'].split(' '))
        bbox = {"w": west, "n": north, "s": south, "e": east}
        bbox_str = "POLYGON(({w:f} {n:f},{e:f} {n:f},{e:f} {s:f},{w:f} {s:f},{w:f} {n:f}))".format(**bbox)

        return {'resolutions': resolutions,
                'min_zoom_level': min(zooms),
                'max_zoom_level': max(zooms),
                'bbox_str': bbox_str,
                'output_path': os.path.abspath(output_path)}
