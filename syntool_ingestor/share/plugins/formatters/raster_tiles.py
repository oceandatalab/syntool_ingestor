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
import sys
import json
from osgeo import osr, gdal, ogr
import numpy as np
import shutil
import logging
import datetime
import traceback
import pkg_resources
try:
    # Use of subprocess32 (backport from Python 3.2) is encouraged by Python
    # official documentation.
    # See https://docs.python.org/2/library/subprocess.html
    import subprocess32 as subprocess
except ImportError:
    import subprocess

from syntool_ingestor.interfaces import IFormatterPlugin

logger = logging.getLogger(__name__)
logger.info('raster_tiles.py loaded')

XIDL_FIX_LON_DELTA = 0

class RasterTilesPlugin(IFormatterPlugin):
    """ """

    def _build_tmp_filename(self, input_path, suffix, ext):
        """ """
        input_dir = os.path.dirname(input_path)
        input_base = os.path.basename(input_path)
        input_name, input_ext = os.path.splitext(input_base)

        return os.path.join(input_dir, '{}{}.{}'.format(input_name, suffix, ext))

    def epsg2wkt(self, epsg):
        """ """
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        return srs.ExportToWkt()

    def default_zoom_range(self, input_path, cfg):
        """ """
        # WILL NOT WORK PROPERLY IN COMPLEX CASES, e.g. :
        # - ecmwf_model_wind.tiff 4326 --> 3413
        # - amsr_sea_ice_concentration.tiff 3411/3412 --> 900913
        src_ds = gdal.Open(input_path)
        src_wkt = self.epsg2wkt(cfg['input_proj'])
        dst_wkt = self.epsg2wkt(cfg['output_proj'])
        dst_ds = gdal.AutoCreateWarpedVRT(src_ds, src_wkt, dst_wkt)
        dst_geotr = dst_ds.GetGeoTransform()
        dst_xsize, dst_ysize = dst_ds.RasterXSize, dst_ds.RasterYSize
        src_ds, dst_ds = None, None

        # Default max zoom
        res = [abs(dst_geotr[1]), abs(dst_geotr[5])]
        max_zoom = max(self.res2zoom(res, cfg))

        # Default min zoom
        upplef = [dst_geotr[0], dst_geotr[3]]
        lowrig = gdal.ApplyGeoTransform(dst_geotr, dst_xsize, dst_ysize)
        widhei = [lowrig[0]-upplef[0], upplef[1]-lowrig[1]]
        min_zoom = min(self.res2zoom([widhei[i]/256. for i in [0, 1]], cfg))
        return [min_zoom, max_zoom]

    def zoom0_res(self, cfg):
        """ """
        extent = [float(ext) for ext in cfg['extent'].split(' ')]
        zoom0_res = [(extent[i+2]-extent[i])/256 for i in [0, 1]]
        return zoom0_res

    def res2zoom(self, res, cfg):
        """ """
        zoom0_res = self.zoom0_res(cfg)
        zoom = [int(np.floor(np.log2(zoom0_res[i]/res[i]))) for i in [0, 1]]
        zoom = [z if z >= 0 else 0 for z in zoom]
        return zoom

    def zoom2res(self, zoom, cfg):
        """ """
        zoom0_res = self.zoom0_res(cfg)
        res = [zoom0_res[i]/2.**zoom for i in [0, 1]]
        return res

    def get_shape_extent(self, src_meta):
        """ """
        geom = ogr.CreateGeometryFromWkt(src_meta['real_shape_str'])
        geom_type = geom.GetGeometryType()
        if geom_type == ogr.wkbPolygon:
            env = geom.GetEnvelope()
        elif geom_type == ogr.wkbGeometryCollection:
            # assume cylindrical output proj and input data around 180deg
            # in this cas, the shape was cut into two polygons
            # (one sticked at left and one sticked at right)
            ngeom = geom.GetGeometryCount()
            if ngeom != 2:
                raise Exception('Shape number of geometry not expected.')
            envs = []
            for i in range(ngeom):
                envs.append(geom.GetGeometryRef(i).GetEnvelope())
            envs = np.array(envs)
            ileft = envs[:, 0].argmin()
            iright = envs[:, 1].argmax()
            if ileft == iright or envs[ileft, 0] >= 0 or envs[iright, 1] <= 0 or \
               envs[ileft, 0] != -envs[iright, 1]:
                raise Exception('Unexpected left/right polygons in shape geometry.')
            shift = 2 * envs[iright, 1]
            env = [0, 0, 0, 0]
            env[2] = envs[:, 2].min()
            env[3] = envs[:, 3].max()
            lonlat_shape = src_meta['lonlat_shape']
            shape_lon = [lonlat[0] for lonlat in lonlat_shape]
            shape_lonmin = min(shape_lon)
            shape_lonmax = max(shape_lon)
            if shape_lonmax > 180. and shape_lonmin >= -180.:
                # left shape -> shift / right shape -> keep
                env[0] = envs[iright, 0]
                env[1] = envs[ileft, 1] + shift
            elif shape_lonmin < -180. and shape_lonmax <= 180.:
                # left shape -> keep / right shape -> shift
                env[0] = envs[iright, 0] - shift
                env[1] = envs[ileft, 1]
            else:
                raise Exception('Unexpected lonlat shape.')
            # env = [np.inf, -np.inf, np.inf, -np.inf]
            # for i in range(ngeom):
            #     tmp_env = geom.GetGeometryRef(i).GetEnvelope()
            #     env[2] = min(env[2], tmp_env[2])
            #     env[3] = max(env[3], tmp_env[3])
            #     if tmp_env[0] < 0: # left shape
            #         env[0] = min(env[0], tmp_env[0] + 2 * abs(tmp_env[0]))
            #         env[1] = max(env[1], tmp_env[1] + 2 * abs(tmp_env[0]))
            #     else: # right shape
            #         env[0] = min(env[0], tmp_env[0])
            #         env[1] = max(env[1], tmp_env[1])
        else:
            raise Exception('Shape geometry type not expected.')
        return [env[0], env[2], env[1], env[3]]

    def get_tiling_extent(self, zoom, cfg, src_meta):
        """ """
        extent = [float(x) for x in cfg['extent'].split(' ')]
        ## NEW metadata.py
        ## Before
        #shape_extent = self.get_shape_extent(src_meta)
        ## Now
        shape_extent = src_meta['warp_infos']['extent']
        ## \NEW metadata.py
        if all([x == y for x, y in zip(extent, shape_extent)]):
            return extent
        uls_c = [shape_extent[0], shape_extent[3]]
        lrs_c = [shape_extent[2], shape_extent[1]]
        ext0, ext1 = [extent[0], extent[3]], [extent[2], extent[1]]
        ul_xy = [np.floor((uls_c[i]-ext0[i])/(ext1[i]-ext0[i])*2**zoom) for i in [0, 1]]
        lr_xy = [np.ceil((lrs_c[i]-ext0[i])/(ext1[i]-ext0[i])*2**zoom) for i in [0, 1]]
        ul_c = [ext0[i]+(ext1[i]-ext0[i])*ul_xy[i]/2**zoom for i in [0, 1]]
        lr_c = [ext0[i]+(ext1[i]-ext0[i])*lr_xy[i]/2**zoom for i in [0, 1]]
        return [ul_c[0], lr_c[1], lr_c[0], ul_c[1]]

    def make_cutline(self, input_path, workspace, viewport):
        """ """
        output_path = os.path.join(workspace, 'cut.csv')
        bbox = {'l': viewport[0], 't': viewport[3], 'r': viewport[2], 'b': viewport[1]}
        wkt = 'POLYGON(({l} {t},{r} {t},{r} {b},{l} {b},{l} {t}))'.format(**bbox)
        with open(output_path, 'w') as f:
            f.write('WKT,dummy\n')
            f.write('"'+wkt+'",\n')
        return output_path

    def warp(self, input_path, workspace, cfg, src_meta):
        """ """

        # gdalwarp fails to interpret some input projections if they are passed
        # as EPSG codes, but will work correctly if they are passee as Proj4
        # definitions.
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(cfg['input_proj'])
        input_proj_as_proj4 = srs.ExportToProj4()

        output_path = os.path.join(workspace, 'warp.vrt')
        call = ['gdalwarp']

        opts = [ '-overwrite'
               , '-of', 'VRT'
               , '-r', cfg['output_options'].get('resampling', 'bilinear')
               #, '-t_srs', 'epsg:{}'.format(cfg['output_proj'])
               , '-s_srs', '{}'.format(input_proj_as_proj4)
               , '-wo', 'SAMPLE_GRID=YES'
               , '-wo', 'SOURCE_EXTRA=100'
               , '-wo', 'NUM_THREADS=ALL_CPUS'
               , '-multi'
               # , '-et', '0.125' # TEST
               # , '-wm', '6.71089e+07' # TEST
               ]

        ## NEW metadata.py
        ## Before
        # shape_geom = ogr.CreateGeometryFromWkt(src_meta['real_shape_str'])
        # shape_geom_type = shape_geom.GetGeometryType()
        # if shape_geom_type != ogr.wkbGeometryCollection: # standard case
        #     opts.extend(['-t_srs', 'epsg:{}'.format(cfg['output_proj'])])
        # else: # around dateline
        #     srs = osr.SpatialReference()
        #     srs.ImportFromEPSG(cfg['output_proj'])
        #     output_proj_as_proj4 = srs.ExportToProj4()
        #     output_proj_as_proj4 += ' +over'
        #     opts.extend(['-t_srs', '{}'.format(output_proj_as_proj4)])
        ## Now
        if not src_meta['warp_infos']['+over']:
            opts.extend(['-t_srs', 'epsg:{}'.format(cfg['output_proj'])])
        else:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(cfg['output_proj'])
            output_proj_as_proj4 = srs.ExportToProj4()
            output_proj_as_proj4 += ' +over'
            opts.extend(['-t_srs', '{}'.format(output_proj_as_proj4)])
        ## \NEW metadata.py

        res = src_meta['max_zoom_resolution'] # set in create_representation()
        opts.extend(['-tr'] + [str(r) for r in res])

        extent = src_meta['max_zoom_extent'] # set in create_representation()
        opts.extend(['-te'] + [str(e) for e in extent])
        # if src_meta.get('cropneeded', False):
        #     #viewport = [float(x) for x in cfg['viewport'].split(' ')]
        #     #opts.extend(['-te'] + [str(x) for x in viewport])
        #     opts.extend(['-te'] + [str(x) for x in src_meta.get('cropextent')])

        isrgb = src_meta.get('isrgb', False)
        if 0 < len(src_meta['nodatavalues']) and not isrgb:
            opts.extend(['-dstnodata', ' '.join(map(str, src_meta['nodatavalues']))])
        else:
            opts.extend(['-dstalpha'])

        # Thin plate splines can only be used if the dataset shape is described
        # with GCPs.
        if src_meta.get('use_gcp', False):
            opts.append('-tps')

        args = [input_path, output_path]

        call.extend(opts)
        call.extend(args)

        logger.debug(' '.join(call))
        try:
            subprocess.check_call(call)
        except subprocess.CalledProcessError as e:
            logger.error('Could not warp.')
            logger.debug(traceback.print_exc())
            raise

        return output_path

    def cut(self, input_path, workspace, cfg, src_meta, tmp_files):
        """ """
        ## NEW metadata.py
        ## Before
        # shape_geom = ogr.CreateGeometryFromWkt(src_meta['real_shape_str'])
        # shape_geom_type = shape_geom.GetGeometryType()
        # viewport = [float(x) for x in cfg['viewport'].split(' ')]
        # extent = src_meta['max_zoom_extent']

        # if shape_geom_type != ogr.wkbGeometryCollection and \
        #    (any(extent[i] < viewport[i] for i in [0, 1]) or \
        #     any(extent[i] > viewport[i] for i in [2, 3])):
        ## Now
        viewport = [float(x) for x in cfg['viewport'].split(' ')]
        extent = src_meta['max_zoom_extent']

        if cfg['output_proj_type'] != 'cylindric' and \
           (any(extent[i] < viewport[i] for i in [0, 1]) or \
            any(extent[i] > viewport[i] for i in [2, 3])):
        ## \NEW metadata.py

            output_path = os.path.join(workspace, 'cut.vrt')
            cutline_path = self.make_cutline(input_path, workspace, viewport)
            call = [  'gdalwarp'
                    , '-overwrite'
                    , '-of', 'VRT'
                    , '-cutline', cutline_path
                 ]

            if 0 < len(src_meta['nodatavalues']):
                call.extend(['-dstnodata', ' '.join(map(str, src_meta['nodatavalues']))])

            call.extend([input_path, output_path])

            logger.debug(' '.join(call))
            try:
                subprocess.check_call(call)
            except subprocess.CalledProcessError as e:
                logger.error('Could not warp (for cutting).')
                logger.debug(traceback.print_exc())
                raise

            tmp_files.append(cutline_path)
            return output_path

        return input_path

    def fix_bands(self, input_path, workspace, cfg, src_meta):
        """ """
        output_path = os.path.join(workspace, 'fix_bands.vrt')
        if 2 == src_meta.get('nb_bands', 0) and src_meta.get('ispaletted', False):
            band = 1
            call = [ 'gdal_translate'
                   , '-b', '{}'.format(band)
                   , '-of', 'VRT'
                   , input_path
                   , output_path
                   ]

            logger.debug(' '.join(call))
            try:
                logger.info('[%s] Start tiling' % str(datetime.datetime.now()))
                subprocess.check_call(call)
                logger.info('[%s] Tiling complete.' % str(datetime.datetime.now()))
            except subprocess.CalledProcessError as e:
                logger.error('Could not produce tiles.')
                logger.debug(traceback.print_exc())
                raise

            # Update metadata
            src_meta['nb_bands'] = 1
            if 0 < len(src_meta['nodatavalues']):
                src_meta['nodatavalues'] = [src_meta['nodatavalues'][0]]

            return output_path

        # By default
        return input_path

    # def fix_nodata(self, input_path, cfg, src_meta):
    #     """ """
    #     # Workaround for an issue with tiler (which transforms 1 paletted
    #     # band to 3 bands _RGB_, but src_nodata is not modified...
    #     monoband = 1 == src_meta['nb_bands']
    #     with_nodata = 0 < len(src_meta.get('nodatavalues', []))
    #     if monoband and with_nodata:
    #         band = 1
    #         dataset = gdal.Open(input_path, gdal.GA_ReadOnly)
    #         color_table = dataset.GetRasterBand(band).GetColorTable()
    #         if color_table is not None:
    #             no_data = src_meta['nodatavalues'][band - 1]

    #             # Check that the no_data value is in color table
    #             if no_data < color_table.GetCount() - 1:
    #                 raise Exception('No entry in color table for no-data value: {}'.format(no_data))
    #             r, g, b, a = color_table.GetColorEntry(no_data)
    #             src_meta['nodatavalues'] = [r, g, b]
    #             dataset = None
    #     return input_path

    def tile(self, input_path, workspace, cfg, src_meta):
        """ """
        min_zoom = cfg['output_options']['min-zoom'] # set in create_representation()
        max_zoom = cfg['output_options']['max-zoom'] # set in create_representation()
        overview_resampling = cfg['output_options'].get('tile-resampling', None)
        if overview_resampling == None:
            if src_meta.get('ispaletted', False):
                overview_resampling = 'nearest'
            else:
                overview_resampling = 'antialias'

        tiler = pkg_resources.resource_filename('syntool_ingestor',
                                                'share/tilers_tools/gdal_tiler.py')

        call = [ sys.executable, tiler
               #, '--base-resampling=nearest'
               , '--no-warp'
               , '--overview-resampling={}'.format(overview_resampling)
               , '-z', '{}:{}'.format(min_zoom, max_zoom)
               , '-p', 'generic'
               , '-t', workspace
               , '--tiles-srs=EPSG:{output_proj}'.format(**cfg)
               , '--tiles-te', cfg['extent'].replace(' ', ', ')
               ]

        if cfg['debug'] == True:
            call.append("--debug")

        no_data_values = src_meta.get('nodatavalues', None)
        if 0 < len(no_data_values):
            no_data_values_str = ','.join(map(str, no_data_values))
            call.append("--src-nodata={}".format(no_data_values_str))

        paletted = src_meta.get('isrgb', False)
        if paletted:
            call.append("--paletted")

        # Give the tiler a center long for the GDAL config option CENTER_LONG.
        # This is needed for warping data around dateline with input cylindric
        # proj and output stereographic proj.
        # This config option is not given to the gdal_warp call in warp()
        # function because it is not preserved in the VRT file and the warp
        # is actually done when the tiler reads the VRT.
        ## NEW metadata.py
        ## Before
        #center_long = src_meta.get('center_long', None)
        ## Now
        center_long = src_meta['warp_infos']['center_long']
        ## \NEW metadata.py
        if center_long is not None:
            call.append("--center-long={}".format(center_long))

        call.append(input_path)

        logger.debug(' '.join(call))
        try:
            logger.info('[%s] Start tiling' % str(datetime.datetime.now()))
            subprocess.check_call(call)
            logger.info('[%s] Tiling complete.' % str(datetime.datetime.now()))
        except subprocess.CalledProcessError as e:
            logger.error('Could not produce tiles.')
            logger.debug(traceback.print_exc())
            raise

        output_path = self._build_tmp_filename(input_path, '', 'generic')
        return output_path

    def can_handle(self, target_format):
        """ """
        return 'rastertiles' == target_format

    def get_output_id(self, cfg):
        """ """
        # !TODO: handle single input band
        return ''

    def get_representation_type(self, cfg):
        """ """
        return 'ZXY'

    def create_representation(self, input_path, input_data, workspace, cfg, src_meta):
        """ """

        temporary_files = []

        # Set zooms and max zoom resolution/extent
        min_zoom = cfg['output_options'].get('min-zoom', '3')
        max_zoom = cfg['output_options'].get('max-zoom', '+1')
        if min_zoom.startswith(('+', '-')) or max_zoom.startswith(('+', '-')):
            def_zooms = self.default_zoom_range(input_path, cfg)
            if min_zoom.startswith('-'):
                min_zoom = str(def_zooms[0]-int(min_zoom[1:]))
            elif min_zoom.startswith('+'):
                min_zoom = str(def_zooms[0]+int(min_zoom[1:]))
            if max_zoom.startswith('-'):
                max_zoom = str(def_zooms[1]-int(max_zoom[1:]))
            elif max_zoom.startswith('+'):
                max_zoom = str(def_zooms[1]+int(max_zoom[1:]))
        if int(min_zoom) > int(max_zoom):
            max_zoom = min_zoom
        cfg['output_options']['min-zoom'] = min_zoom
        cfg['output_options']['max-zoom'] = max_zoom
        src_meta['max_zoom_resolution'] = self.zoom2res(int(max_zoom), cfg)
        src_meta['max_zoom_extent'] = self.get_tiling_extent(int(max_zoom), cfg, src_meta)

        # Fix bands
        bands_ok_path = self.fix_bands(input_path, workspace, cfg, src_meta)
        temporary_files.append(bands_ok_path)

        # Warp with respect to the max zoom resolution/extent
        warp_ok_path = self.warp(bands_ok_path, workspace, cfg, src_meta)
        temporary_files.append(warp_ok_path)

        # Cut (if max zoom extent is greater than viewport)
        cut_ok_path = self.cut(warp_ok_path, workspace, cfg, src_meta, temporary_files)
        temporary_files.append(cut_ok_path)

        # Fix nodata values
        # nodata_ok_path = self.fix_nodata(warp_ok_path, cfg, src_meta)
        # temporary_files.append(nodata_ok_path)

        # Tile
        tiles_path = self.tile(cut_ok_path, workspace, cfg, src_meta)

        input_dir, input_name = os.path.split(tiles_path)
        output_dir = os.path.join(workspace, 'tiles.zxy')
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.rename(tiles_path, output_dir)

        # Clean temporary files
        if not cfg.get('keep_intermediary_files', False):
            to_remove = filter( lambda x: x != input_path and os.path.exists(x)
                              , temporary_files)
            map(os.remove, list(set(to_remove)))
            logger.debug('These temporary files have been removed: {}'.format(to_remove))

        ## not needed because of modif in tiler_misc.py (XYZtiling in GenericMap)
        # for zoom_dir in glob.glob(os.path.join(output_dir, 'z*')):
        #     zoom_num = os.path.basename(zoom_dir).replace('z', '')
        #     os.rename(zoom_dir, os.path.join(output_dir, zoom_num))

        with open(os.path.join(output_dir, 'tilemap.json')) as tile_file:
            tile_dict = json.load(tile_file)

        resolutions = []
        for zoom, tileset in tile_dict['tilesets'].iteritems():
            resolutions.append('{}:{}'.format(zoom, tileset['units_per_pixel']))
        zooms = map(int, tile_dict['tilesets'].keys())

        ## NEW metadata.py
        ## Before
        #west, south, east, north = tile_dict['bbox']
        ## Now
        west, south, east, north = src_meta['bbox_infos']['bbox']
        ## \NEW metadata.py
        bbox = {"west": west, "north": north, "south": south, "east": east}
        bbox_str = "POLYGON(({west:f} {north:f},{east:f} {north:f},{east:f} {south:f},{west:f} {south:f},{west:f} {north:f}))".format(**bbox)

        extra_meta = {'resolutions': resolutions,
                      'min_zoom_level': min(zooms),
                      'max_zoom_level': max(zooms),
                      'bbox_str': bbox_str,
                      'output_path': os.path.abspath(output_dir)}

        # Workaround for cross-IDL
        ## NEW metadata.py
        ## Before
        # shape_geom = ogr.CreateGeometryFromWkt(src_meta['shape_str'])
        # if ogr.wkbGeometryCollection == shape_geom.GetGeometryType() and \
        #    2 == shape_geom.GetGeometryCount():
        #     l0, r0, b0, t0 = shape_geom.GetGeometryRef(0).GetEnvelope() # West
        #     l1, r1, b1, t1 = shape_geom.GetGeometryRef(1).GetEnvelope() # East

        #     if l0 + r0 > l1 + r1:
        #         # Switch coordinates so that
        #         # l0, r0, t0, b0 are the coordinates of the western shape
        #         # l1, r1, t1, b1 are the coordinates of the eastern shape
        #         l0, r0, t0, b0, l1, r1, t1, b1 = l1, r1, t1, b1, l0, r0, t0, b0

        #     logger.debug('Checking XIDL...')
        #     logger.debug('{} {} {} {} vs {} {} {} {}'.format(l0, r0, b0, t0,
        #                                                      l1, r1, b1, t1))

        #     if XIDL_FIX_LON_DELTA + r0 < l1:
        #         bbox_pattern = 'POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))'
        #         extra_meta['w_bbox'] = bbox_pattern.format(l0, t0, r0, t0,
        #                                                    r0, b0, l0, b0,
        #                                                    l0, t0)
        #         extra_meta['e_bbox'] = bbox_pattern.format(l1, t1, r1, t1,
        #                                                    r1, b1, l1, b1,
        #                                                    l1, t1)
        ## Now
        if src_meta['bbox_infos']['xIDL'] == True:
            # bboxes contain [xmin, ymin, xmax, ymax]
            bbox_pattern = 'POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))'
            l0, b0, r0, t0 = src_meta['bbox_infos']['w_bbox']
            extra_meta['w_bbox'] = bbox_pattern.format(l0, t0, r0, t0, r0, b0,
                                                       l0, b0, l0, t0)
            l1, b1, r1, t1 = src_meta['bbox_infos']['e_bbox']
            extra_meta['e_bbox'] = bbox_pattern.format(l1, t1, r1, t1, r1, b1,
                                                       l1, b1, l1, t1)
        ## \NEW metadata.py
        return extra_meta
