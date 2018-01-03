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
import io
import sys
import json
import glob
import zlib
import base64
from osgeo import osr, gdal, ogr
import pyproj
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.interpolate import interp1d
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
import syntool_ingestor.metadata as mtdt

logger = logging.getLogger(__name__)
logger.info('trajectory_tiles.py loaded')

XIDL_FIX_LON_DELTA = 0

class TrajectoryTilesPlugin(IFormatterPlugin):
    """ """

    def can_handle(self, target_format):
        """ """
        return 'trajectorytiles' == target_format

    def get_output_id(self, cfg):
        """ """
        return ''

    def get_representation_type(self, cfg):
        """ """
        return 'ZXY'

    def create_representation(self, input_path, input_data, workspace, cfg, src_meta):
        """ """

        logger.info('[{}] Start initialisation.'.format(datetime.datetime.now()))

        temporary_files = []

        # Set projections as OSR spatial reference.
        # By the way we add '+over' to the output projection if cylindrical
        # (if data around dateline, we'll keep things continuous until the tiling
        # which will use a modulo to set a correct tile numbering).
        input_proj = cfg['input_proj']
        input_srs = osr.SpatialReference()
        input_srs.ImportFromEPSG(input_proj)
        output_proj = cfg['output_proj']
        output_srs = osr.SpatialReference()
        if output_proj in mtdt.CYLINDRIC_PROJ:
            tmp_srs = osr.SpatialReference()
            tmp_srs.ImportFromEPSG(output_proj)
            output_srs.ImportFromProj4(tmp_srs.ExportToProj4() + ' +over')
        else:
            output_srs.ImportFromEPSG(output_proj)

        # Get trajectory GCPs in output projection
        input_dset = gdal.Open(input_path)
        input_tf = gdal.Transformer(input_dset, None, ['MAX_GCP_ORDER=-1'])
        traj_gcps = get_trajectory_gcps(input_dset, input_srs, output_srs,
                                        input_transformer=input_tf)

        # Get trajectory resolution
        traj_res = get_trajectory_mean_resolution(input_dset, input_srs,
                                                  input_transformer=input_tf)
        # traj_res = get_trajectory_output_resolutions(input_dset, input_srs, output_srs,
        #                                              input_transformer=input_tf)

        # Set output options
        map_extent = [float(ext) for ext in cfg['extent'].split(' ')]
        tilesmap = TilesMap(map_extent)
        min_zoom = cfg['output_options'].get('min-zoom', '3')
        max_zoom = cfg['output_options'].get('max-zoom', '+1')
        if max_zoom.startswith(('+', '-')):
            _max_zoom = max(tilesmap.res2zoom([traj_res, traj_res]))
            if max_zoom.startswith('-'):
                max_zoom = _max_zoom - int(max_zoom[1:])
            elif max_zoom.startswith('+'):
                max_zoom = _max_zoom + int(max_zoom[1:])
        else:
            max_zoom = int(max_zoom)
        if min_zoom.startswith(('+', '-')):
            widhei = [traj_gcps['gcpmidx'].max() - traj_gcps['gcpmidx'].min(),
                      traj_gcps['gcpmidy'].max() - traj_gcps['gcpmidy'].min()]
            _min_res = [widhei[i] / tilesmap.tile_size[i] for i in [0, 1]]
            _min_zoom = min(tilesmap.res2zoom(_min_res))
            if min_zoom.startswith('-'):
                min_zoom = _min_zoom - int(min_zoom[1:])
            elif min_zoom.startswith('+'):
                min_zoom = _min_zoom + int(min_zoom[1:])
        else:
            min_zoom = int(min_zoom)
        if min_zoom > max_zoom:
            max_zoom = min_zoom
        cfg['output_options']['min-zoom'] = str(min_zoom)
        cfg['output_options']['max-zoom'] = str(max_zoom)
        linewidth_meter = float(cfg['output_options'].get('linewidth-meter', '5000'))
        min_linewidth_pixel = int(cfg['output_options'].get('min-linewidth-pixel', '4'))
        resampling = cfg['output_options'].get('resampling', 'average')
        cfg['output_options']['linewidth-meter'] = str(linewidth_meter)
        cfg['output_options']['min-linewidth-pixel'] = str(min_linewidth_pixel)
        cfg['output_options']['resampling'] = resampling

        logger.info('[{}] End initialisation.'.format(datetime.datetime.now()))

        # Remove unwanted bands
        nb_bands = src_meta.get('nb_bands', 0)
        ispaletted = src_meta.get('ispaletted', False)
        if nb_bands == 2 and ispaletted:
            bands_ok_path = os.path.join(workspace, 'fix_bands.vrt')
            remove_bands(input_path, bands_ok_path, bands2keep=[1])
            temporary_files.append(bands_ok_path)
            src_meta['nb_bands'] = 1
            if 0 < len(src_meta['nodatavalues']):
                src_meta['nodatavalues'] = [src_meta['nodatavalues'][0]]
        else:
            bands_ok_path = input_path

        # Loop on zooms (average traj, modify traj GCPs, shape computation, warp, cut, tile)
        viewport = cfg['viewport'].split(' ')
        viewport_geom = ogr.CreateGeometryFromWkt(mtdt._get_bbox_wkt(*viewport))
        zooms_tilemap = {}
        zooms_transparency = {}
        ## NEW metadata.py
        ## Before
        # zooms_shape_geom = {}
        # zooms_shape_extent = {}
        ## Now
        zooms_meta = {}
        ## \NEW metadata.py
        for zoom in range(min_zoom, max_zoom + 1):

            logger.info('[{}] Start processing zoom {}.'.format(datetime.datetime.now(), zoom))

            zoom_res = tilesmap.zoom2res(zoom)

            # Average
            avrg_res = max(zoom_res)
            if resampling == 'average' and traj_res < avrg_res:
                avrg_ok_path = os.path.join(workspace, 'fix_average_zoom{:02d}.tiff'.format(zoom))
                navrg = np.ceil(avrg_res / traj_res).astype('int')
                try:
                    logger.info('[{}] Start averaging.'.format(datetime.datetime.now()))
                    average_trajectory(bands_ok_path, avrg_ok_path, navrg)
                    logger.info('[{}] End averaging.'.format(datetime.datetime.now()))
                except:
                    logger.error('Could not average.')
                    raise
                temporary_files.append(avrg_ok_path)
            else:
                avrg_ok_path = bands_ok_path

            # Transform GCPs
            linewidth = [max([linewidth_meter, min_linewidth_pixel * r]) for r in zoom_res]
            gcps_ok_path = os.path.join(workspace, 'fix_gcps_zoom{:02d}.vrt'.format(zoom))
            try:
                logger.info('[{}] Start modifying gcps.'.format(datetime.datetime.now()))
                modify_trajectory_gcps(avrg_ok_path, gcps_ok_path, traj_gcps, linewidth)
                logger.info('[{}] End modifying gcps.'.format(datetime.datetime.now()))
            except:
                logger.error('Could not modify gcps.')
                raise
            temporary_files.append(gcps_ok_path)

            # Compute shape geometry in the same way it is done in metadata.py
            # (we redo it at each zoom since GCPs are changed).
            logger.info('[{}] Start computing shape.'.format(datetime.datetime.now()))
            gcps_ok_dset = gdal.Open(gcps_ok_path)
            gcps_ok_tf = gdal.Transformer(gcps_ok_dset, None, ['MAX_GCP_ORDER=-1'])
            shape = get_trajectory_shape(gcps_ok_dset, transformer=gcps_ok_tf, ndist=330,
                                         min_shape_res=750000., max_shape_points=33)
            srs4326 = osr.SpatialReference()
            srs4326.ImportFromEPSG(4326)
            proj_tf = osr.CoordinateTransformation(output_srs, srs4326)
            lonlat_shape = proj_tf.TransformPoints(shape)
            ## NEW metadata.py
            ## Before
            # shape_geom0 = mtdt._get_shape_geometry(lonlat_shape, gcps_ok_dset, input_proj, output_proj)
            # shape_geom, _, _ = mtdt._get_crop_info(shape_geom0, viewport_geom)
            # shape_extent = get_shape_extent(shape_geom, lonlat_shape)
            # zooms_shape_geom[zoom] = shape_geom
            # zooms_shape_extent[zoom] = shape_extent
            # center_long = None
            # if input_proj in mtdt.CYLINDRIC_PROJ and output_proj in mtdt.STEREO_PROJ:
            #     shape_lon = [lonlat[0] for lonlat in shape]
            #     minlon, maxlon = min(shape_lon), max(shape_lon)
            #     if (maxlon > 180 and minlon > -180) or (maxlon < 180 and minlon < -180):
            #         center_long = '{}'.format((maxlon + minlon) / 2.)
            #tiling_extent = tilesmap.tiling_extent(zoom, shape_extent)
            ## Now
            output_shape_geom, bbox_infos, warp_infos = mtdt._get_output_shape(lonlat_shape,
                                                                               gcps_ok_dset,
                                                                               input_proj,
                                                                               output_proj,
                                                                               viewport_geom)
            zooms_meta[zoom] = {}
            zooms_meta[zoom]['lonlat_shape'] = lonlat_shape
            zooms_meta[zoom]['output_shape_geom'] = output_shape_geom
            zooms_meta[zoom]['bbox_infos'] = bbox_infos
            zooms_meta[zoom]['warp_infos'] = warp_infos
            shape_extent = warp_infos['extent']
            tiling_extent = tilesmap.tiling_extent(zoom, shape_extent)
            ## \NEW metadata.py
            logger.info('[{}] End computing shape.'.format(datetime.datetime.now()))

            # TMP : Check GDAL transformer
            # print linewidth
            # gdaltf = gdal.Transformer(gcps_ok_dset, None, ['MAX_GCP_ORDER=-1'])
            # gridx = np.linspace(tiling_extent[0], tiling_extent[2], num=800)
            # gridy = np.linspace(tiling_extent[1], tiling_extent[3], num=800)
            # gridxy = np.array((np.tile(gridx[:, np.newaxis], (1, gridy.size)),
            #                    np.tile(gridy[np.newaxis, :], (gridx.size, 1))))
            # dimsxy = gridxy.shape[1:3]
            # gridxy = gridxy.reshape((2, -1)).transpose()
            # pixlin = np.array(gdaltf.TransformPoints(1, gridxy)[0])
            # pix = pixlin[:, 0].reshape(dimsxy).transpose()
            # lin = pixlin[:, 1].reshape(dimsxy).transpose()
            # gcps = gcps_ok_dset.GetGCPs()
            # gcpx = np.array([gcp.GCPX for gcp in gcps])
            # gcpy = np.array([gcp.GCPY for gcp in gcps])
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(pix, origin='lower', interpolation='nearest',
            #            extent=[gridx.min(), gridx.max(), gridy.min(), gridy.max()])
            # plt.colorbar(label='pixel') ; plt.xlabel('x') ; plt.ylabel('y')
            # plt.plot(gcpx, gcpy, 'k+')
            # plt.xlim((gridx.min(), gridx.max())) ; plt.ylim((gridy.min(), gridy.max()))
            # plt.figure()
            # plt.imshow(lin, origin='lower', interpolation='nearest',
            #            extent=[gridx.min(), gridx.max(), gridy.min(), gridy.max()])
            # plt.colorbar(label='line') ; plt.xlabel('x') ; plt.ylabel('y')
            # plt.plot(gcpx, gcpy, 'k+')
            # plt.xlim((gridx.min(), gridx.max())) ; plt.ylim((gridy.min(), gridy.max()))
            # plt.show()
            # \TMP

            # Estimate the tiles to be generated
            logger.info('[{}] Start estimating tiles.'.format(datetime.datetime.now()))
            traj_bboxes = get_trajectory_bboxes(gcps_ok_dset, nbbox=128,
                                                max_extent=tiling_extent,
                                                transformer=gcps_ok_tf)
            zoom_tiles = []
            for bbox in traj_bboxes:
                zoom_tiles.extend(tilesmap.bbox2tiles(zoom, bbox))
            tiles_list = {zoom: list(set(zoom_tiles))}
            logger.info('[{}] End estimating tiles.'.format(datetime.datetime.now()))

            # Warp
            warp_res = zoom_res
            warp_extent = tiling_extent
            if resampling == 'average':
                _resampling = 'near'
            else:
                _resampling = resampling
            isrgb = src_meta.get('isrgb', False)
            if 0 < len(src_meta['nodatavalues']) and not isrgb:
                dstnodata = src_meta['nodatavalues']
                dstalpha = False
            else:
                dstnodata = None
                dstalpha = True
            tps = src_meta.get('use_gcp', False)
            warp_ok_path = os.path.join(workspace, 'warp_zoom{:02d}.vrt'.format(zoom))
            warp(gcps_ok_path, output_srs, warp_ok_path, output_srs,
                 warp_res, warp_extent, _resampling, dstnodata, dstalpha, tps)
            temporary_files.append(warp_ok_path)

            # Cut
            ## NEW metadata.py
            ## Before
            # shape_geom_type = shape_geom.GetGeometryType()
            # if shape_geom_type != ogr.wkbGeometryCollection and \
            #    (any(tiling_extent[i] < float(viewport[i]) for i in [0, 1]) or \
            #     any(tiling_extent[i] > float(viewport[i]) for i in [2, 3])):
            ## Now
            if cfg['output_proj_type'] != 'cylindric' and \
               (any(tiling_extent[i] < float(viewport[i]) for i in [0, 1]) or \
                any(tiling_extent[i] > float(viewport[i]) for i in [2, 3])):
            ## \NEW metadata.py
                # Make cutline
                cut_extent = viewport
                cutline_path = os.path.join(workspace, 'cutline_zoom{:02d}.csv'.format(zoom))
                write_cutline(cutline_path, cut_extent)
                temporary_files.append(cutline_path)
                # Do cut
                dstnodata = src_meta.get('nodatavalues', None)
                cut_ok_path = os.path.join(workspace, 'cut_zoom{:02d}.vrt'.format(zoom))
                cut(warp_ok_path, cut_ok_path, cutline_path, dstnodata=dstnodata)
                temporary_files.append(cut_ok_path)
            else:
                cut_ok_path = warp_ok_path

            # Tile
            srcnodata = src_meta.get('nodatavalues', None)
            paletted = src_meta.get('isrgb', False)
            ## NEW metadata.py
            center_long = warp_infos['center_long']
            ## \NEW metadata.py
            debug = cfg['debug']
            tiles_list_path = os.path.join(workspace, 'tiles_list_zoom{:02d}.json'.format(zoom))
            with open(tiles_list_path, 'w') as tl_file:
                json.dump(tiles_list, tl_file)
            temporary_files.append(tiles_list_path)
            tiles_ok_path = tile(cut_ok_path, workspace, output_proj, map_extent,
                                 zoom, zoom, srcnodata=srcnodata, paletted=paletted,
                                 center_long=center_long, debug=debug,
                                 tiles_list=tiles_list_path)

            # Move tiles and read tilemap.json / transparency.json
            tiles_dir = os.path.join(workspace, 'tiles.zxy')
            if zoom == min_zoom:
                if os.path.isdir(tiles_dir):
                    shutil.rmtree(tiles_dir)
                os.mkdir(tiles_dir)
            os.rename(os.path.join(tiles_ok_path, '{}'.format(zoom)),
                      os.path.join(tiles_dir, '{}'.format(zoom)))
            with open(os.path.join(tiles_ok_path, 'tilemap.json')) as tile_file:
                zooms_tilemap[zoom] = json.load(tile_file)
            with open(os.path.join(tiles_ok_path, 'transparency.json')) as transp_file:
                zooms_transparency[zoom] = json.load(transp_file)
            shutil.rmtree(tiles_ok_path)

            logger.info('[{}] End processing zoom {}.'.format(datetime.datetime.now(), zoom))

        # Clean temporary files
        if not cfg.get('keep_intermediary_files', False):
            to_remove = filter( lambda x: x != input_path and os.path.exists(x)
                              , temporary_files)
            map(os.remove, list(set(to_remove)))
            logger.debug('These temporary files have been removed: {}'.format(to_remove))

        ## NEW metadata.py
        ## Before
        # Set bbox and shape with min zoom
        # ref_zoom = min_zoom
        # bbox = zooms_shape_extent[ref_zoom]
        # bbox_str = "POLYGON(({b[0]:f} {b[3]:f},{b[2]:f} {b[3]:f},{b[2]:f} {b[1]:f},"\
        #            "{b[0]:f} {b[1]:f},{b[0]:f} {b[3]:f}))".format(b=bbox)
        # shape_geom = zooms_shape_geom[ref_zoom]
        # shape_wkt = shape_geom.ExportToWkt().replace('POLYGON (', 'POLYGON(')
        # if not cfg['no_shape']:
        #     src_meta['shape_str'] = shape_wkt
        # src_meta['real_shape_str'] = shape_wkt
        ## Now
        # Update src_meta and set bbox_str
        # We use min zoom as the reference
        ref_zoom = min_zoom
        real_shape_wkt = zooms_meta[ref_zoom]['output_shape_geom'].ExportToWkt()
        real_shape_wkt = real_shape_wkt.replace('POLYGON (', 'POLYGON(')
        if cfg['no_shape']:
            shape_wkt = 'POINT(0 0)'
        else:
            shape_wkt = real_shape_wkt
        src_meta['lonlat_shape'] = zooms_meta[ref_zoom]['lonlat_shape']
        src_meta['real_shape_str'] = real_shape_wkt
        src_meta['shape_str'] = shape_wkt
        src_meta['bbox_infos'] = zooms_meta[ref_zoom]['bbox_infos']
        src_meta['warp_infos'] = zooms_meta[ref_zoom]['warp_infos']
        bbox = zooms_meta[ref_zoom]['bbox_infos']['bbox']
        bbox_str = "POLYGON(({b[0]:f} {b[3]:f},{b[2]:f} {b[3]:f},{b[2]:f} {b[1]:f},"\
                   "{b[0]:f} {b[1]:f},{b[0]:f} {b[3]:f}))".format(b=bbox)
        ## \NEW metadata.py

        # Reconstruct tilemap.json / transparency.json
        tilemap_dict = zooms_tilemap[ref_zoom]
        ## NEW metadata.py
        ## Before
        #tilemap_dict['bbox'] = bbox
        ## Now
        # do nothing: why modify bbox in tilemap.json if we don't do it for raster tiles ?
        ## \NEW metadata.py
        for z, d in zooms_tilemap.iteritems():
            if z != ref_zoom:
                tilemap_dict['tilesets'].update(d['tilesets'])
        with open(os.path.join(tiles_dir, 'tilemap.json'), 'w') as tile_file:
            json.dump(tilemap_dict, tile_file, indent=2)
        transparency_dict = zooms_transparency[ref_zoom]
        for z, d in zooms_transparency.iteritems():
            if z != ref_zoom:
                transparency_dict.update(d)
        with open(os.path.join(tiles_dir, 'transparency.json'), 'w') as transp_file:
            json.dump(transparency_dict, transp_file, indent=0)

        tiles_mask = create_tiles_mask(tiles_dir)
        logger.debug('Tiles mask: {}'.format(tiles_mask))
        resolutions = []
        for zoom, tileset in tilemap_dict['tilesets'].iteritems():
            resolutions.append('{}:{}'.format(zoom, tileset['units_per_pixel']))
        zooms = map(int, tilemap_dict['tilesets'].keys())
        resolutions.append('9998:{}*{}'.format(max(3, min(zooms)), max(zooms)))
        resolutions.append('9999:{}'.format(tiles_mask))

        extra_meta = {'resolutions': resolutions,
                      'min_zoom_level': min(zooms),
                      'max_zoom_level': max(zooms),
                      'bbox_str': bbox_str,
                      'output_path': os.path.abspath(tiles_dir)}

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
        if zooms_meta[ref_zoom]['bbox_infos']['xIDL'] == True:
            # bboxes contain [xmin, ymin, xmax, ymax]
            bbox_pattern = 'POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))'
            l0, b0, r0, t0 = zooms_meta[ref_zoom]['bbox_infos']['w_bbox']
            extra_meta['w_bbox'] = bbox_pattern.format(l0, t0, r0, t0, r0, b0,
                                                       l0, b0, l0, t0)
            l1, b1, r1, t1 = zooms_meta[ref_zoom]['bbox_infos']['e_bbox']
            extra_meta['e_bbox'] = bbox_pattern.format(l1, t1, r1, t1, r1, b1,
                                                       l1, b1, l1, t1)
        ## \NEW metadata.py
        return extra_meta

def create_tiles_mask(tiles_dir):
    """"""
    result_buffer = io.BytesIO()
    zooms = [ d for d in os.listdir(tiles_dir) if d.isdigit() ]
    zooms = map(int, zooms)
    zooms.sort() # numeric sort!
    for zoom in zooms:
        if zoom < 3:
            continue
        mask_dim = 2 ** zoom
        mask = np.zeros((mask_dim * mask_dim / 8), dtype='uint8')
        tiles_pattern = os.path.join(tiles_dir, str(zoom), '*', '*.png')
        png_paths = glob.glob(tiles_pattern)
        for png_path in png_paths:
            png_dir = os.path.dirname(png_path)
            png_base = os.path.basename(png_path)
            x_str = os.path.basename(png_dir)
            x = int(x_str)
            y_str, _ = os.path.splitext(png_base)
            y = int(y_str)
            i = y * mask_dim + x
            bytei = i // 8
            biti = i % 8
            mask[bytei] = mask[bytei] | (1 << biti)

        # Write mask in result buffer
        for byte in map(lambda x: x.tobytes(), mask):
            result_buffer.write(byte)

    deflated = zlib.compress(result_buffer.getvalue())
    encoded = base64.b64encode(deflated)
    return encoded



def get_trajectory_gcps(input_dset, input_srs, output_srs, input_transformer=None):
    """
    (for trajectory tiles) Get/Compute useful GCPs variables in order to set later
    the appropriate trajectory width.
    """
    # Read input GCPs and set GCPs transformer
    gcps = input_dset.GetGCPs()
    if input_transformer is None:
        input_transformer = gdal.Transformer(input_dset, None, ['MAX_GCP_ORDER=-1'])
    # Get/Compute useful GCPs variables.
    gcppixel = np.array([gcp.GCPPixel for gcp in gcps])
    gcpline = np.array([gcp.GCPLine for gcp in gcps])
    pix = np.zeros((gcppixel.size, 3)) + 0.5
    lin = gcpline[:, np.newaxis] + np.array([-0.5, 0., 0.5])[np.newaxis, :]
    pixlin = np.vstack((pix.flatten(), lin.flatten())).transpose()
    inpxyz = input_transformer.TransformPoints(0, pixlin)[0]
    if input_srs.IsSame(output_srs):
        outxyz = np.array(inpxyz)
    else:
        proj_transformer = osr.CoordinateTransformation(input_srs, output_srs)
        outxyz = np.array(proj_transformer.TransformPoints(inpxyz))
    outx = outxyz[:, 0].reshape(pix.shape)
    outy = outxyz[:, 1].reshape(pix.shape)
    gcpmiddir = np.arctan2(outy[:, 2] - outy[:, 0], outx[:, 2] - outx[:, 0])
    gcpmidx = outx[:, 1]
    gcpmidy = outy[:, 1]
    return {'gcppixel':gcppixel, 'gcpline':gcpline, 'gcpmiddir': gcpmiddir,
            'gcpmidx':gcpmidx, 'gcpmidy':gcpmidy, 'srs':output_srs}


def modify_trajectory_gcps(input_path, output_path, traj_gcps, traj_width):
    """
    (for trajectory tiles) Create a VRT where GCPs are transformed (projection and x/y)
    in order to have a given linewidth in output projection.
    """
    # Set newGCPs
    gcppixel = traj_gcps['gcppixel']
    gcpline = traj_gcps['gcpline']
    gcpmiddir = traj_gcps['gcpmiddir']
    gcpmidx = traj_gcps['gcpmidx']
    gcpx = gcpmidx + (gcppixel - 0.5) * traj_width[0] * np.cos(gcpmiddir - np.pi / 2)
    gcpmidy = traj_gcps['gcpmidy']
    gcpy = gcpmidy + (gcppixel - 0.5) * traj_width[1] * np.sin(gcpmiddir - np.pi / 2)
    gcpz = np.zeros(gcpx.shape)
    new_gcps = []
    for x, y, z, p, l in zip(gcpx, gcpy, gcpz, gcppixel, gcpline):
        new_gcps.append(gdal.GCP(x, y, z, p, l))
    # CHECK
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(gcpmidx, np.rad2deg(gcpmiddir), '+')
    # plt.figure()
    # plt.plot(gcpmidx, gcpmidy, '+')
    # plt.plot(gcpx, gcpy, '+')
    # plt.show()
    # import pdb ; pdb.set_trace()
    # \CHECK
    # Create an in-memory VRT copy with new GCPs
    vrt_driver = gdal.GetDriverByName('VRT')
    input_dset = gdal.Open(input_path)
    inmem_dset = vrt_driver.CreateCopy('', input_dset)
    inmem_dset.SetGCPs(new_gcps, traj_gcps['srs'].ExportToWkt())
    # Write in output_path
    output_dset = vrt_driver.CreateCopy(output_path, inmem_dset)
    input_dset, inmem_dset, output_dset = None, None, None


def get_trajectory_mean_resolution(input_dset, input_srs, input_transformer=None):
    """
    (for trajectory tiles) Compute trajectory mean resolution in meters.
    """
    if input_transformer is None:
        input_transformer = gdal.Transformer(input_dset, None, ['MAX_GCP_ORDER=-1'])
    pix = np.zeros(input_dset.RasterYSize + 1) + 0.5
    lin = np.arange(input_dset.RasterYSize + 1)
    pixlin = np.vstack((pix, lin)).transpose()
    inpxyz = input_transformer.TransformPoints(0, pixlin)[0]
    srs4326 = osr.SpatialReference()
    srs4326.ImportFromEPSG(4326)
    if input_srs.IsSame(srs4326):
        lonlat = np.array(inpxyz)
    else:
        proj_tf = osr.CoordinateTransformation(input_srs, srs4326)
        lonlat = np.array(proj_tf.TransformPoints(inpxyz))
    geod = pyproj.Geod(ellps='WGS84')
    _, _, dist = geod.inv(lonlat[:-1, 0], lonlat[:-1, 1],
                          lonlat[1:, 0], lonlat[1:, 1])
    return dist.mean()


def get_trajectory_output_resolutions(input_dset, input_srs, output_srs,
                                      input_transformer=None):
    """
    (for trajectory tiles) Compute trajectory resolution in output projection for
    each point of trajectory.
    """
    if input_transformer is None:
        input_transformer = gdal.Transformer(input_dset, None, ['MAX_GCP_ORDER=-1'])
    proj_tf = osr.CoordinateTransformation(input_srs, output_srs)
    pix = np.zeros(input_dset.RasterYSize + 1) + 0.5
    lin = np.arange(input_dset.RasterYSize + 1)
    pixlin = np.vstack((pix, lin)).transpose()
    inpxyz = input_transformer.TransformPoints(0, pixlin)[0]
    if input_srs.IsSame(output_srs):
        outxyz = np.array(inpxyz)
    else:
        proj_tf = osr.CoordinateTransformation(input_srs, output_srs)
        outxyz = np.array(proj_tf.TransformPoints(inpxyz))
    resxy = np.sqrt((outxyz[1:, 0] - outxyz[:-1, 0]) ** 2. + \
                    (outxyz[1:, 1] - outxyz[:-1, 1]) ** 2.)
    return resxy


def average_trajectory(input_path, output_path, naverage):
    """
    (for trajectory tiles) Average trajectory.
    """
    # Read input
    mem_drv = gdal.GetDriverByName('MEM')
    _input_dset = gdal.Open(input_path)
    input_dset = mem_drv.CreateCopy('', _input_dset)
    _input_dset = None
    input_band = input_dset.GetRasterBand(1)
    input_values = input_band.ReadAsArray()[:, 0]
    ndv = input_band.GetNoDataValue()
    if ndv is not None:
        input_mask = input_values == ndv
        input_values[input_mask] = 0
    else:
        input_mask = np.zeros(input_values.shape, dtype='bool')
    # Average
    kernel = np.ones(naverage)
    output_values = convolve(input_values.astype('float'), kernel, mode='constant', cval=0)
    norm = convolve(1. - input_mask, kernel, mode='constant', cval=0)
    output_mask = norm == 0
    output_values[~output_mask] /= norm[~output_mask]
    if np.issubdtype(input_values.dtype, np.integer):
        output_values = np.round(output_values).astype(input_values.dtype)
    else:
        output_values = output_values.astype(input_values.dtype)
    if ndv is not None:
        output_values[output_mask] = ndv
    # CHECK
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(np.ma.MaskedArray(input_values, mask=input_mask), '+-b')
    # plt.plot(np.ma.masked_equal(output_values, ndv), '+-r')
    # plt.show()
    # import pdb ; pdb.set_trace()
    # \CHECK
    # Write output
    output_array = output_values[:, np.newaxis]
    input_band.WriteArray(output_array)
    gtiff_drv = gdal.GetDriverByName('GTiff')
    output_dset = gtiff_drv.CreateCopy(output_path, input_dset)
    input_dset, output_dset = None, None
    # CHECK
    # import matplotlib.pyplot as plt
    # input_dset = gdal.Open(input_path)
    # input_values = input_dset.GetRasterBand(1).ReadAsArray()[:, 0]
    # output_dset = gdal.Open(output_path)
    # output_values = output_dset.GetRasterBand(1).ReadAsArray()[:, 0]
    # plt.figure()
    # plt.plot(np.ma.masked_equal(input_values, ndv), '+-b')
    # plt.plot(np.ma.masked_equal(output_values, ndv), '+-r')
    # plt.show()
    # import pdb ; pdb.set_trace()
    # \CHECK


def get_trajectory_bboxes(input_dset, nbbox=10, max_extent=None, transformer=None):
    """
    (for trajectory tiles) Compute bboxes along the trajectory.
    """
    if transformer is None:
        transformer = gdal.Transformer(input_dset, None, ['MAX_GCP_ORDER=-1'])
    pix = np.concatenate((np.zeros(nbbox + 1), np.ones(nbbox + 1)))
    _lin = np.linspace(0, input_dset.RasterYSize, num=nbbox + 1)
    lin = np.concatenate((_lin, _lin))
    pixlin = np.vstack((pix, lin)).transpose()
    xyz = np.array(transformer.TransformPoints(0, pixlin)[0])
    bboxes = []
    for ibbox in range(nbbox):
        x = xyz[[ibbox, ibbox + 1, ibbox + nbbox + 1, ibbox + nbbox + 2], 0]
        y = xyz[[ibbox, ibbox + 1, ibbox + nbbox + 1, ibbox + nbbox + 2], 1]
        if max_extent is None:
            bbox = [x.min(), y.min(), x.max(), y.max()]
            bboxes.append(bbox)
        else:
            bbox = [max([x.min(), max_extent[0]]), max([y.min(), max_extent[1]]),
                    min([x.max(), max_extent[2]]), min([y.max(), max_extent[3]])]
            if bbox[0] <= bbox[2] and bbox[1] <= bbox[3]:
                bboxes.append(bbox)
    # CHECK
    # import matplotlib.pyplot as plt
    # plt.plot(xyz[:, 0], xyz[:, 1], '+r')
    # for b in bboxes:
    #     plt.plot([b[0], b[0], b[2], b[2], b[0]],
    #              [b[1], b[3], b[3], b[1], b[1]], '-b')
    # if max_extent is not None:
    #     e = max_extent
    #     plt.plot([e[0], e[0], e[2], e[2], e[0]],
    #              [e[1], e[3], e[3], e[1], e[1]], '-g')
    # plt.show()
    # import pdb ; pdb.set_trace()
    # \CHECK
    return bboxes


def remove_bands(input_path, output_path, bands2keep=[1]):
    """ """
    call = ['gdal_translate']
    for band in bands2keep:
        call.extend(['-b', '{}'.format(band)])
    call.extend([ '-of', 'VRT'
                , input_path
                , output_path])
    logger.debug(' '.join(call))
    try:
        logger.info('[{}] Start removing bands'.format(datetime.datetime.now()))
        subprocess.check_call(call)
        logger.info('[{}] End removing bands.'.format(datetime.datetime.now()))
    except subprocess.CalledProcessError as e:
        logger.error('Could not remove bands.')
        logger.debug(traceback.print_exc())
        raise


def get_trajectory_shape(dataset, transformer=None, ndist=330,
                         min_shape_res=750000., max_shape_points=33):
    """ """
    if transformer is None:
        transformer = gdal.Transformer(dataset, None, ['MAX_GCP_ORDER=-1'])
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    pix = np.zeros(ndist + 1) + xsize / 2.
    lin = np.linspace(0., ysize, num=ndist + 1)
    pixlin = np.array([pix, lin]).transpose()
    xy = np.array(transformer.TransformPoints(0, pixlin)[0])[:, 0:2]
    dists = np.sqrt((xy[1:, 0] - xy[:-1, 0]) ** 2 + \
                    (xy[1:, 1] - xy[:-1, 1]) ** 2)
    distscum = np.concatenate(([0.], dists.cumsum()))
    dist = distscum[-1]
    if dist == 0.:
        print 'WARNING : null distance in shape computation.'
        npts = 2
    else:
        npts = np.ceil(dist / min_shape_res).astype('int') + 1
        npts = min(max_shape_points, npts)
    linfunc = interp1d(distscum, lin, kind='linear')
    _lin = linfunc(np.linspace(0., dist, num=npts))
    lin = np.concatenate([_lin, _lin[::-1], _lin[[0]]])
    pix = np.concatenate([np.zeros(npts) + xsize, np.zeros(npts), [xsize]])
    pixlin = np.array([pix, lin]).transpose()
    xy = np.array(transformer.TransformPoints(0, pixlin)[0])[:, 0:2]
    shape = xy.tolist()
    return shape


def get_shape_extent(shape_geom, lonlat_shape):
    """ """
    geom_type = shape_geom.GetGeometryType()
    if geom_type == ogr.wkbPolygon:
        env = shape_geom.GetEnvelope()
    elif geom_type == ogr.wkbGeometryCollection:
        # assume cylindrical output proj and input data around 180deg
        # in this case, the shape was cut into two polygons
        # (one sticked at left and one sticked at right)
        ngeom = shape_geom.GetGeometryCount()
        if ngeom != 2:
            raise Exception('Shape number of geometry not expected.')
        envs = []
        for i in range(ngeom):
            envs.append(shape_geom.GetGeometryRef(i).GetEnvelope())
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
        # shape_lon = [lonlat[0] for lonlat in lonlat_shape]
        # shape_lonmin = min(shape_lon)
        # shape_lonmax = max(shape_lon)
        # if shape_lonmax > 180. and shape_lonmin >= -180.:
        #     for i in range(ngeom):
        #         tmp_env = shape_geom.GetGeometryRef(i).GetEnvelope()
        #         env[2] = min(env[2], tmp_env[2])
        #         env[3] = max(env[3], tmp_env[3])
        #         if tmp_env[0] < 0: # left shape -> shift
        #             env[0] = min(env[0], tmp_env[0] + 2 * abs(tmp_env[0]))
        #             env[1] = max(env[1], tmp_env[1] + 2 * abs(tmp_env[0]))
        #         else: # right shape -> keep
        #             env[0] = min(env[0], tmp_env[0])
        #             env[1] = max(env[1], tmp_env[1])
        # elif shape_lonmin < -180. and shape_lonmax <= 180.:
        #     for i in range(ngeom):
        #         tmp_env = shape_geom.GetGeometryRef(i).GetEnvelope()
        #         env[2] = min(env[2], tmp_env[2])
        #         env[3] = max(env[3], tmp_env[3])
        #         if tmp_env[0] < 0: # left shape -> keep
        #             env[0] = min(env[0], tmp_env[0])
        #             env[1] = max(env[1], tmp_env[1])
        #         else: # right shape -> shift
        #             env[0] = min(env[0], tmp_env[0] - 2 * abs(tmp_env[1]))
        #             env[1] = max(env[1], tmp_env[1] - 2 * abs(tmp_env[1]))
        # else:
        #     raise Exception('Unexpected lonlat shape.')
    else:
        raise Exception('Shape geometry type not expected.')
    return [env[0], env[2], env[1], env[3]]


def warp(input_path, input_srs, output_path, output_srs,
         res=None, extent=None, resampling=None, dstnodata=None,
         dstalpha=False, tps=False):
    """ """
    call = [  'gdalwarp'
            , '-overwrite'
            , '-of', 'VRT'
            , '-s_srs', input_srs.ExportToProj4()
            , '-t_srs', output_srs.ExportToProj4()
            , '-wo', 'SAMPLE_GRID=YES'
            , '-wo', 'SOURCE_EXTRA=100'
            , '-wo', 'NUM_THREADS=ALL_CPUS'
            , '-multi'
    ]
    if res is not None:
        call.extend(['-tr'] + [str(r) for r in res])
    if extent is not None:
        call.extend(['-te'] + [str(e) for e in extent])
    if resampling is not None:
        call.extend(['-r', resampling])
    if dstnodata is not None:
        call.extend(['-dstnodata', ' '.join(map(str, dstnodata))])
    if dstalpha:
        call.append('-dstalpha')
    if tps:
        call.append('-tps')
    call.extend([input_path, output_path])
    logger.debug(' '.join(call))
    try:
        logger.info('[{}] Start warping'.format(datetime.datetime.now()))
        subprocess.check_call(call)
        logger.info('[{}] End warping.'.format(datetime.datetime.now()))
    except subprocess.CalledProcessError as e:
        logger.error('Could not warp.')
        logger.debug(traceback.print_exc())
        raise


def write_cutline(cutline_path, cut_extent):
    """ """
    wkt = 'POLYGON(({e[0]} {e[3]},{e[2]} {e[3]},{e[2]} {e[1]},'\
          '{e[0]} {e[1]},{e[0]} {e[3]}))'.format(e=cut_extent)
    with open(cutline_path, 'w') as f:
        f.write('WKT,dummy\n')
        f.write('"'+wkt+'",\n')


def cut(input_path, output_path, cutline_path, dstnodata=None):
    """ """
    call = [  'gdalwarp'
            , '-overwrite'
            , '-of', 'VRT'
            , '-cutline', cutline_path
    ]
    if dstnodata is not None:
        call.extend(['-dstnodata', ' '.join(map(str, dstnodata))])
    call.extend([input_path, output_path])
    logger.debug(' '.join(call))
    try:
        logger.info('[{}] Start cuting'.format(datetime.datetime.now()))
        subprocess.check_call(call)
        logger.info('[{}] End cuting.'.format(datetime.datetime.now()))
    except subprocess.CalledProcessError as e:
        logger.error('Could not warp (for cutting).')
        logger.debug(traceback.print_exc())
        raise


def _build_tmp_filename(input_path, suffix, ext):
    """ """
    input_dir = os.path.dirname(input_path)
    input_base = os.path.basename(input_path)
    input_name, input_ext = os.path.splitext(input_base)
    return os.path.join(input_dir, '{}{}.{}'.format(input_name, suffix, ext))


def tile(input_path, workspace, output_proj, extent, min_zoom, max_zoom,
         overview_resampling=None, srcnodata=None, paletted=False,
         center_long=None, debug=False, tiles_list=None):
    """ """
    tiler = pkg_resources.resource_filename('syntool_ingestor',
                                            'share/tilers_tools/gdal_tiler.py')
    call = [ sys.executable, tiler
           , '--no-warp'
           , '-p', 'generic'
           , '-t', workspace
           , '--tiles-srs=EPSG:{}'.format(output_proj)
           , '--tiles-te', ', '.join(map(str, extent))
           , '-z', '{}:{}'.format(min_zoom, max_zoom)
           ]
    if overview_resampling is not None:
        call.append('--overview-resampling={}'.format(overview_resampling))
    if srcnodata is not None:
        call.append('--src-nodata={}'.format(','.join(map(str, srcnodata))))
    if paletted:
        call.append('--paletted')
    # Give the tiler a center long for the GDAL config option CENTER_LONG.
    # This is needed for warping data around dateline with input cylindric
    # proj and output stereographic proj.
    # This config option is not given to the gdal_warp call in warp()
    # function because it is not preserved in the VRT file and the warp
    # is actually done when the tiler reads the VRT.
    if center_long is not None:
        call.append('--center-long={}'.format(center_long))
    if debug:
        call.append('--debug')
    if tiles_list is not None:
        call.append('--tiles-list={}'.format(tiles_list))
    call.append(input_path)
    logger.debug(' '.join(call))
    try:
        logger.info('[{}] Start tiling.'.format(datetime.datetime.now()))
        subprocess.check_call(call)
        logger.info('[{}] End tiling.'.format(datetime.datetime.now()))
    except subprocess.CalledProcessError as e:
        logger.error('Could not produce tiles.')
        logger.debug(traceback.print_exc())
        raise
    output_path = _build_tmp_filename(input_path, '', 'generic')
    return output_path


def epsg2wkt(epsg):
    """ """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    return srs.ExportToWkt()


class TilesMap(object):
    """ """

    def __init__(self, map_extent, tile_size=[256, 256]):
        """ """
        self.map_extent = map_extent # [left, bottom, right, top]
        self.tile_size = tile_size # [x, y]

    def zoom0_res(self):
        """ """
        res0 = [(self.map_extent[2] - self.map_extent[0]) / float(self.tile_size[0]),
                (self.map_extent[3] - self.map_extent[1]) / float(self.tile_size[1])]
        return res0

    def res2zoom(self, res):
        """ """
        res0 = self.zoom0_res()
        zoom = [int(np.floor(np.log2(res0[i] / res[i]))) for i in [0, 1]]
        zoom = [z if z >= 0 else 0 for z in zoom]
        return zoom

    def zoom2res(self, zoom):
        """ """
        res0 = self.zoom0_res()
        res = [res0[i] / 2. ** zoom for i in [0, 1]]
        return res

    def default_zoom_range(self, input_path, input_srs, output_srs):
        """ """
        # WILL NOT WORK PROPERLY IN COMPLEX CASES, e.g. :
        # - ecmwf_model_wind.tiff 4326 --> 3413
        # - amsr_sea_ice_concentration.tiff 3411/3412 --> 900913
        #
        # Use gdal.AutoCreateWarpedVRT in order to mimic a warp operation
        # and see what GDAL would have produced as output size and resolution.
        src_ds = gdal.Open(input_path)
        src_wkt = input_srs.ExportToWkt()
        dst_wkt = output_srs.ExportToWkt()
        dst_ds = gdal.AutoCreateWarpedVRT(src_ds, src_wkt, dst_wkt)
        dst_geotr = dst_ds.GetGeoTransform()
        dst_xsize, dst_ysize = dst_ds.RasterXSize, dst_ds.RasterYSize
        src_ds, dst_ds = None, None
        # Default max zoom
        res = [abs(dst_geotr[1]), abs(dst_geotr[5])]
        max_zoom = max(self.res2zoom(res))
        # Default min zoom
        upplef = [dst_geotr[0], dst_geotr[3]]
        lowrig = gdal.ApplyGeoTransform(dst_geotr, dst_xsize, dst_ysize)
        widhei = [lowrig[0] - upplef[0], upplef[1] - lowrig[1]]
        min_zoom = min(self.res2zoom([widhei[i] / float(self.tile_size[i]) for i in [0, 1]]))
        return [min_zoom, max_zoom]

    def tiling_extent(self, zoom, any_extent):
        """ """
        if all([x == y for x, y in zip(self.map_extent, any_extent)]):
            return self.map_extent
        uls_c = [any_extent[0], any_extent[3]]
        lrs_c = [any_extent[2], any_extent[1]]
        ext0 = [self.map_extent[0], self.map_extent[3]]
        ext1 = [self.map_extent[2], self.map_extent[1]]
        ul_xy = [np.floor((uls_c[i] - ext0[i]) / (ext1[i] - ext0[i]) * 2 ** zoom) for i in [0, 1]]
        lr_xy = [np.ceil((lrs_c[i] - ext0[i]) / (ext1[i] - ext0[i]) * 2 ** zoom) for i in [0, 1]]
        ul_c = [ext0[i] + (ext1[i] - ext0[i]) * ul_xy[i] / 2 ** zoom for i in [0, 1]]
        lr_c = [ext0[i] + (ext1[i] - ext0[i]) * lr_xy[i] / 2 ** zoom for i in [0, 1]]
        return [ul_c[0], lr_c[1], lr_c[0], ul_c[1]]

    def coord2pix(self, zoom, coord):
        """ """
        origin = [self.map_extent[0], self.map_extent[3]]
        res = self.zoom2res(zoom)
        res[1] = -res[1]
        pix = [int(np.round((coord[i] - origin[i]) / res[i])) for i in (0, 1)]
        return pix

    def pix2tile(self, pix):
        """ """
        tile = [pix[i] // self.tile_size[i] for i in (0, 1)]
        return tile

    def bbox2tiles(self, zoom, bbox):
        """ """
        p_ul = self.coord2pix(zoom, [bbox[0], bbox[3]])
        t_ul = self.pix2tile(p_ul)
        p_lr = self.coord2pix(zoom, [bbox[2], bbox[1]])
        if p_lr[0] % self.tile_size[0] == 0:
            p_lr[0] -= self.tile_size[0] // 2
        if p_lr[1] % self.tile_size[1] == 0:
            p_lr[1] -= self.tile_size[1] // 2
        t_lr = self.pix2tile(p_lr)
        xx = [t_ul[0], t_lr[0]]
        yy = [t_ul[1], t_lr[1]]
        tiles = [(zoom, x, y) \
                 for x in range(min(xx), max(xx) + 1) \
                 for y in range(min(yy), max(yy) + 1)]
        return tiles

