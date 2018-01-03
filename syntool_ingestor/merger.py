#!/usr/bin/env python2
# -*- coding: utf-8 -*-

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
import errno
import io
import glob
import json
import ogr
import numpy
from collections import defaultdict
from PIL import Image
import PIL.PngImagePlugin
import shutil
import logging
import base64
import numpy
import zlib

logger = logging.getLogger(__name__)

# Threshold for the X-IDL fix: roughly 1.8 degrees in WebMercator projection
XIDL_THRESHOLD = 200375


def merge_images(images_path, output_path):
    """ """
    # Open images and get some infos
    imgs = [Image.open(path) for path in images_path]
    imgs_mode = [img.mode for img in imgs]
    has_alpha = ['A' in mode for mode in imgs_mode]
    has_transparency = ['transparency' in img.info for img in imgs]

    # Check mode and transparency
    if min(imgs_mode) != max(imgs_mode):
        raise Exception('Try to merge images with different modes.')
    mode = imgs_mode[0]
    if mode != 'P' and not all(has_alpha) and not all(has_transparency):
        raise Exception('Try to merge images with at least one without transparency info.')

    # Paste images in the first one and save it
    if mode == 'P': # P -> RGBA
        imgs = [img.convert('RGBA') for img in imgs]
    merged_img = imgs[0]
    for img in imgs[1:]:
        if 'A' in img.mode:
            mask = img.split()[-1]
        else:
            transparency = img.info['transparency']
            mask = img.point(lambda p: p != transparency and 255)
        merged_img.paste(img, mask=mask)
    if mode == 'P': # RGBA -> P
        alpha = merged_img.split()[-1]
        mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
        merged_img = merged_img.convert('RGB')
        merged_img.paste((0, 0, 0), mask)
        merged_img = merged_img.convert('P', palette=Image.ADAPTIVE, colors=255)
        transparency = 255
        merged_img.paste(transparency, mask)
        merged_img.save(output_path, transparency=transparency)
    else:
        if 'A' in merged_img.mode:
            merged_img.save(output_path)
        else:
            transparency = merged_img.info['transparency']
            merged_img.save(output_path, transparency=transparency)


def path2zxy(path):
    """ """
    dir1, base1 = os.path.split(path)
    dir2, base2 = os.path.split(dir1)
    zxy = (os.path.basename(dir2), base2, os.path.splitext(base1)[0])
    return zxy


def merge_tiles(datagroup_dir, workspace):
    """ """
    # Find tiles
    pattern = os.path.join(datagroup_dir, '*', 'tiles.zxy', '*', '*', '*.png')
    tiles_path = glob.glob(pattern)
    tiles_zxy = [path2zxy(path) for path in tiles_path]

    # Link tiles path to every tiles zxy
    tiles_dict = {}
    for tile_zxy, tile_path in zip(tiles_zxy, tiles_path):
        if tile_zxy in tiles_dict:
            tiles_dict[tile_zxy].append(tile_path)
        else:
            tiles_dict[tile_zxy] = [tile_path]

    # Copy/Merge tiles
    output_dir = os.path.join(workspace, 'tiles.zxy')
    for zxy, paths in tiles_dict.iteritems():
        z, x, y = zxy
        tile_output_dir = os.path.join(output_dir, z, x)
        if not os.path.exists(tile_output_dir):
            os.makedirs(tile_output_dir)
        tile_output_path = os.path.join(tile_output_dir, y+'.png')
        if len(paths) == 1:
            shutil.copy2(paths[0], tile_output_path)
        else:
            #print zxy, len(paths)
            merge_images(paths, tile_output_path)

    return output_dir

def merge_vectorfields(pngs, workspace, metadata, bbox, cfg):
    """ """
    reserved = ('interlace', 'gamma', 'dpi', 'transparency', 'aspect')
    logger.debug('BBOX = {}'.format(bbox))

    x_extent = bbox['east'] - bbox['west']
    x_res = float(metadata['x_res'])
    x_size = int(numpy.round(x_extent / x_res))
    logger.debug(x_size)

    y_extent = bbox['north'] - bbox['south']
    y_res = float(metadata['y_res'])
    y_size = int(numpy.round(y_extent / y_res))
    logger.debug(y_size)

    result_bbox = ogr.CreateGeometryFromWkt(metadata['bbox_str'])
    result_left, result_right, _, result_top = result_bbox.GetEnvelope()
    logger.debug('Origin')
    logger.debug(result_left)
    logger.debug(result_top)

    output_proj = cfg.get('output_proj', None)
    cylindric_proj = output_proj in (900913, 3857)
    result_global_x = False
    if cylindric_proj:
        viewport = [float(x) for x in cfg['viewport'].split(' ')]
        if result_left == viewport[0] and result_right == viewport[2]:
            result_global_x = True
    logger.debug('Cylindric proj: {}'.format(cylindric_proj))
    logger.debug('Global x: {}'.format(result_global_x))

    results = None
    result_mode = None
    result_info = {}
    for png_path in pngs:
        png_meta = pngs[png_path]

        # Initialize results using information from the first PNG
        if results is None:
            nodatavalues = png_meta['nodatavalues']
            if 2 == len(nodatavalues):
                result_mode = 'LA'
                results = [
                            Image.new('L', (x_size, y_size), nodatavalues[0]),
                            Image.new('L', (x_size, y_size), nodatavalues[1]),
                          ]
            elif 3 == len(nodatavalues):
                result_mode = 'RGB'
                results = [
                            Image.new('L', (x_size, y_size), nodatavalues[0]),
                            Image.new('L', (x_size, y_size), nodatavalues[1]),
                            Image.new('L', (x_size, y_size), nodatavalues[2]),
                          ]
            else:
                raise Exception('Number of channels not supported: {}',
                                len(nodatavalues))

        img = Image.open(png_path, 'r')
        # Check PNG metadata
        for meta_key in img.info:
            if meta_key in reserved:
                continue
            meta_value = img.info[meta_key]
            if meta_key in result_info:
                if meta_value != result_info[meta_key]:
                    raise Exception('Metadata "{}" is not homogenous amongst' \
                                    ' PNG files to merge'.format(meta_key))
            else:
                result_info[meta_key] = meta_value

        logger.debug(png_path)
        logger.debug(img.mode)
        logger.debug(img.info)

        # Compute PNG positions in the coordinates system of the result
        png_bbox = ogr.CreateGeometryFromWkt(png_meta['bbox_str'])
        png_left, png_right, png_bottom, png_top = png_bbox.GetEnvelope()
        img_x0 = (png_left - result_left) / x_res
        img_x1 = (png_right - result_left) / x_res - 1
        img_y0 = (result_top - png_top) / y_res
        img_y1 = (result_top - png_bottom) / y_res - 1
        logger.debug('Component')
        logger.debug(png_left)
        logger.debug(png_top)
        logger.debug('Offsets (as floats)')
        logger.debug([img_x0, img_x1])
        logger.debug([img_y0, img_y1])
        img_x0 = int(numpy.round(img_x0))
        img_x1 = int(numpy.round(img_x1))
        img_y0 = int(numpy.round(img_y0))
        img_y1 = int(numpy.round(img_y1))

        # Check y positions: PNG bbox should not exceed result bbox
        if img_y0 < 0:
            raise Exception('PNG bbox exceeds result bbox (top).')
        if img_y1 >= y_size:
            raise Exception('PNG bbox exceeds result bbox (bottom).')

        # Check/Handle x positions
        if result_global_x:
            # Projection is cylindric and result bbox covers the viewport in x (ie 360°)
            # We don't accept:
            # - PNG completely outside result bbox (left or right sides)
            # - PNG exceeding result bbox on left and right sides at the same time
            if img_x1 < 0:
                raise Exception('PNG bbox is outside result bbox (left)')
            if img_x0 >= x_size:
                raise Exception('PNG bbox is outside result bbox (right)')
            if img_x0 < 0 and img_x1 >= x_size:
                raise Exception('PNG bbox exceeds result bbox (left and right)')
            # If PNG bbox exceeds result bbox (left or right side), we paste the over part
            # and let img be the part intersecting result_bbox
            if img_x0 < 0 or img_x1 >= x_size:
                if img_x0 < 0:
                    over_size = -img_x0
                    over_img = img.crop((0, 0, over_size, img.size[1]))
                    over_x0 = x_size - over_size
                    img = img.crop((over_size, 0, img.size[0], img.size[1]))
                    img_x0 = 0
                elif img_x1 >= x_size:
                    over_size = img_x1 - x_size + 1
                    over_img = img.crop((img.size[0] - over_size, 0, img.size[0], img.size[1]))
                    over_x0 = 0
                    img = img.crop((0, 0, img.size[0] - over_size, img.size[1]))
                    # note: img_x0 does not change
                logger.debug('over_size: {}'.format(over_size))
                logger.debug('png: {} {}'.format(png_left, png_right))
                logger.debug('result: {} {}'.format(result_left, result_right))
                # if 0 in over_img.size:
                #     continue
                for i in range(0, len(results)):
                    band_data = list(over_img.getdata(band=i))
                    mask_data = [255 if x != nodatavalues[i] else 0 for x in band_data]
                    band_img = Image.new('L', over_img.size)
                    band_img.putdata(band_data)
                    mask_img = Image.new('L', over_img.size)
                    mask_img.putdata(mask_data)
                    results[i].paste(band_img, (over_x0, img_y0), mask_img)
        else:
            # We don't accept PNG bbox exceeding result bbox on left or right sides.
            # But first, if cylindric proj is set we may try to shift PNG bbox.
            if cylindric_proj == True and (img_x1 < 0 or img_x0 >= x_size):
                if img_x1 < 0:
                    _shift = 2 * viewport[2]
                elif img_x0 >= x_size:
                    _shift = -2 * viewport[2]
                img_x0 = (png_left + _shift - result_left) / x_res
                img_x1 = (png_right + _shift - result_left) / x_res - 1
                logger.debug('shifted positions: {}'.format([img_x0, img_x1]))
                img_x0 = int(numpy.round(img_x0))
                img_x1 = int(numpy.round(img_x1))
            if img_x0 < 0:
                raise Exception('PNG bbox exceeds result bbox (left).')
            if img_x1 >= x_size:
                raise Exception('PNG bbox exceeds result bbox (right).')

        # Paste part of PNG which is contained in result bbox.
        # if 0 in img.size:
        #     continue
        for i in range(0, len(results)):
            band_data = list(img.getdata(band=i))
            mask_data = [255 if x != nodatavalues[i] else 0 for x in band_data]
            band_img = Image.new('L', img.size)
            band_img.putdata(band_data)
            mask_img = Image.new('L', img.size)
            mask_img.putdata(mask_data)
            results[i].paste(band_img, (img_x0, img_y0), mask_img)

    # Merge channels
    result = Image.merge(result_mode, results)

    # Save result
    output_path = os.path.join(workspace, 'vectorFieldLayer.png')
    metapng = PIL.PngImagePlugin.PngInfo()
    for key in result_info:
        if key not in reserved:
            metapng.add_text(key, str(result_info[key]), 0)
    result.save(output_path, 'PNG', pnginfo=metapng)
    return output_path


def debug_tiles_mask(z, z_bitfield, basename=None):
    import PIL.Image
    import random
    import string

    def randomword(length):
        return ''.join(random.choice(string.lowercase) for i in range(length))

    if basename is None:
        basename = randomword(5)

    # Debug
    z_dim = 2 ** z
    debug_data = numpy.ndarray(shape=(z_dim, z_dim), dtype='uint8')
    for y in xrange(0, z_dim):
        row_data = []
        for x in xrange(0, z_dim):
            i = y * z_dim + x
            bytei = i // 8
            biti = i % 8
            bit_value = numpy.bitwise_and(z_bitfield[bytei], (1 << biti))
            debug_data[y][x] = 255 if bit_value != 0 else 0
    debug = PIL.Image.fromarray(debug_data)
    debug.save('{}_{}.png'.format(basename, z))


def merge_tiles_masks(masks_levels, masks_bitfields):
    """"""
    # First recreate the mask dictionary with zoom levels as  keys
    # Then create the merged tiles masks by using a boolean OR for each zoom
    # level.
    result_by_zoom = {}
    for z_ext, bitfield in zip(masks_levels, masks_bitfields):
        z_min = z_ext['z_min']
        z_max = z_ext['z_max']
        offset = 0
        for z in xrange(z_min, z_max + 1):
            z_dim = 2 ** z
            z_shape = (z_dim * z_dim / 8)
            new_offset = offset + z_shape
            z_bitfield = numpy.ndarray(buffer=bitfield[offset:new_offset],
                                       shape=z_shape, dtype='uint8')
            #debug_tiles_mask(z, z_bitfield)
            offset = new_offset
            if z not in result_by_zoom:
                result_by_zoom[z] = numpy.zeros(z_shape, dtype='uint8')
            # Boolean union between the masks
            result_by_zoom[z] = numpy.bitwise_or(result_by_zoom[z], z_bitfield)

    # Serialize result
    result = []
    result_z_min = min(result_by_zoom.keys())
    result_z_max = max(result_by_zoom.keys())
    result.append('9998:{}*{}'.format(result_z_min, result_z_max))

    mask_buffer = io.BytesIO()
    for z in result_by_zoom:
        #debug_tiles_mask(z, result_by_zoom[z], 'result')
        z_mask_bytes = map(lambda x: x.tobytes(), result_by_zoom[z])
        for byte in z_mask_bytes:
            mask_buffer.write(byte)
    deflated = zlib.compress(mask_buffer.getvalue())
    encoded = base64.b64encode(deflated)
    result.append('9999:{}'.format(encoded))

    return result


def xidl_shapes(w_shape, e_shape, geom):
    """Distribute geometries between eastern and western areas based on the
    position of the centroid.
    WARNING: the limit between the two areas evolves when geometries are added
    so it might lead to incoherent results but it should not matter for the
    targetted datasets (WV for Sentinel1, ASCAT L2B swaths)."""

    centroid = geom.Centroid()
    x, y, _ = centroid.GetPoint()

    e_min = 20037508.3427892
    e_count = e_shape.GetGeometryCount()
    if 0 < e_count:
        e_min, _, _, _ = e_shape.GetEnvelope()

    w_max = -20037508.3427892
    w_count = w_shape.GetGeometryCount()
    if 0 < w_count:
        _, w_max, _, _ = w_shape.GetEnvelope()

    limit = .5 * (w_max + e_min)
    if x >= limit:
        # Center of geometry is closer to eastern shape
        e_shape.AddGeometry(geom.Clone())
    else:
        # Center of geometry is closer to western shape
        w_shape.AddGeometry(geom.Clone())


def merge_metadata(datagroup_dir, cfg):
    """ """
    # Read metadata
    pattern = os.path.join(datagroup_dir, '*', 'metadata.json')
    metadata_path = glob.glob(pattern)
    metadata_lists = defaultdict(list)
    for path in metadata_path:
        with open(path, 'r') as f:
            for key, val in json.load(f).items():
                metadata_lists[key].append(val)

    # Merge constant fields (after checking unicity)
    metadata = {}
    fields = ['datagroup', 'output_format', 'output_level', 'output_timeline',
              'output_type', 'product', 'syntool_id']
    for key in fields:
        logger.debug('Checking key: {}'.format(key))
        if min(metadata_lists[key]) != max(metadata_lists[key]):
            raise Exception('{} field is not uniq in metadata.'.format(key))
        metadata[key] = metadata_lists[key][0]
    metadata['dataset'] = metadata['datagroup']

    # Merge "minimum" fields
    fields = ['begin_datetime', 'min_zoom_level']
    for key in fields:
        metadata[key] = min(metadata_lists[key])

    # Merge "maximum" fields
    fields = ['end_datetime', 'max_zoom_level']
    for key in fields:
        metadata[key] = max(metadata_lists[key])

    # Merge resolutions
    resolutions = {}
    masks_levels = []
    masks_bitfields = []
    for res_list in metadata_lists['resolutions']:
        for res in res_list:
            z, r = res.split(':')
            if '9998' == z:
                # tiles mask zoom levels use a '*' separator between min and
                # max values
                z_min, z_max = map(int, r.split('*'))
                masks_levels.append({'z_min': z_min, 'z_max': z_max})
            elif '9999' == z:
                # tiles masks are serialized as base64(deflate(mask))
                decoded = base64.b64decode(r)
                inflated = zlib.decompress(decoded)
                masks_bitfields.append(inflated)
            elif z not in resolutions:
                resolutions[z] = r
            else:
                if resolutions[z] != r:
                    raise Exception('Different resolutions in metadata.')
    metadata['resolutions'] = [z+':'+resolutions[z] for z in resolutions]

    # Merge tiles masks
    if len(masks_levels) != len(masks_bitfields):
        raise Exception('Incomplete tiles mask information, check the "9998"' \
                        'and "9999" entries in the "resolutions" metadata')

    if 0 < len(masks_levels):
        results = merge_tiles_masks(masks_levels, masks_bitfields)
        metadata['resolutions'].extend(results)

    # Merge shape_str ("addition")
    # shape_geom = ogr.Geometry(ogr.wkbGeometryCollection)
    # for wkt in metadata_lists['shape_str']:
    #     geom = ogr.CreateGeometryFromWkt(wkt)
    #     geom_type = geom.GetGeometryType()
    #     if geom_type == ogr.wkbPolygon:
    #         shape_geom.AddGeometry(geom.Clone())
    #     elif geom_type == ogr.wkbGeometryCollection or \
    #          geom_type == ogr.wkbMultiPolygon:
    #         ngeom = geom.GetGeometryCount()
    #         for g in range(ngeom):
    #             shape_geom.AddGeometry(geom.GetGeometryRef(g).Clone())
    #     else:
    #         raise Exception('Unexpected geometry.')
    # metadata['shape_str'] = shape_geom.ExportToWkt().replace(' (', '(')

    # Merge shape_str (union)
    output_proj = cfg.get('output_proj', None)
    cylindric_proj = output_proj in (900913, 3857)
    e_shape = ogr.Geometry(ogr.wkbGeometryCollection)
    w_shape = ogr.Geometry(ogr.wkbGeometryCollection)
    shape_geom = None
    for wkt in metadata_lists['shape_str']:
        geom = ogr.CreateGeometryFromWkt(wkt)
        geom_type = geom.GetGeometryType()

        # Handle cross-IDL cases
        # Distribute geometries (or sub-geometries for composed shapes) between
        # the eastern and western blocks.
        if cylindric_proj:
            if geom_type in (ogr.wkbGeometryCollection, ogr.wkbMultiPolygon):
                geom_count = geom.GetGeometryCount()
                for x in range(geom_count):
                    _geom = geom.GetGeometryRef(x).Clone()
                    xidl_shapes(w_shape, e_shape, _geom)
            else:
                xidl_shapes(w_shape, e_shape, geom)

        if shape_geom is None:
            shape_geom = geom
        else:
            shape_geom = shape_geom.Union(geom)

    shape_geom_type = shape_geom.GetGeometryType()
    if shape_geom_type == ogr.wkbPolygon:
        pass
    elif shape_geom_type == ogr.wkbGeometryCollection:
        pass
    elif shape_geom_type == ogr.wkbMultiPolygon:
        _shape_geom = shape_geom.Clone()
        shape_geom = ogr.Geometry(ogr.wkbGeometryCollection)
        for x in range(_shape_geom.GetGeometryCount()):
            shape_geom.AddGeometry(_shape_geom.GetGeometryRef(x).Clone())
    else:
        raise Exception('Unexpected shape geometry.')
    metadata['shape_str'] = shape_geom.ExportToWkt().replace(' (', '(')

    # Extra checks for vectorfields: PNGs must have the same resolutions
    # (x and y) and result from an ingestion where the
    # "aligned-bbox" option has been activated
    if 'VECTOR_FIELD' == metadata_lists['output_type'][0]:
        # X resolution
        if min(metadata_lists['x_res']) != max(metadata_lists['x_res']):
            raise Exception('x_res field is not unique in metadata.')
        metadata['x_res'] = metadata_lists['x_res'][0]

        # Y resolution
        if min(metadata_lists['y_res']) != max(metadata_lists['y_res']):
            raise Exception('y_res field is not unique in metadata.')
        metadata['y_res'] = metadata_lists['y_res'][0]

        # Aligned-bbox
        for output_opts in metadata_lists['output_options']:
            if 'aligned-bbox' not in output_opts or \
               output_opts['aligned-bbox'].lower() not in ['true', 'yes']:
                raise Exception('Vectorfield datasets can only be merged if ' \
                                'the "aligned-bbox" output option has been ' \
                                'used during ingestion')

    # E|W bbox trick
    w_count = w_shape.GetGeometryCount()
    e_count = e_shape.GetGeometryCount()
    if 0 < w_count * e_count:
        # The bbox trick is only useful when the western and eastern blocks
        # are not tool close to each other
        e_min, _, _, _ = e_shape.GetEnvelope()
        _, w_max, _, _ = w_shape.GetEnvelope()
        if XIDL_THRESHOLD < e_min - w_max:
            bbox_pattern = 'POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))'
            l, r, b, t = e_shape.GetEnvelope()
            metadata['e_bbox'] = bbox_pattern.format(l, t, r, t, r, b, l, b,
                                                     l, t)
            l, r, b, t = w_shape.GetEnvelope()
            metadata['w_bbox'] = bbox_pattern.format(l, t, r, t, r, b, l, b,
                                                     l, t)

    # Merge bbox_str
    if 'VECTOR_FIELD' != metadata_lists['output_type'][0]:
        # rastertiles/trajectorytiles: we simply take the envelope of the merged shape.
        west, east, south, north = shape_geom.GetEnvelope()
        if 'e_bbox' in metadata and 'w_bbox' in metadata:
            # xIDL detected: we modify west/east in order to be consistent with
            # the outputs of rastertiles and trajectorytiles plugins.
            viewport = [float(x) for x in cfg['viewport'].split(' ')]
            west = e_min
            east = w_max + 2 * viewport[2]
    else:
        # vectorfield: bbox_str is used to place correctly the PNG on the map.
        # All individual PNGs bbox_str should belong to the same grid (x/y
        # resolutions and aligned-bbox were checked before). Then, it seems
        # natural to compute the final bbox_str from individual bbox_str.
        west, east, south, north = numpy.inf, -numpy.inf, numpy.inf, -numpy.inf
        viewport = [float(x) for x in cfg['viewport'].split(' ')]
        vleft, _, vright, _ = viewport
        if 'e_bbox' in metadata and 'w_bbox' in metadata:
            # xIDL detected: in order to minimize final PNG size, we accept a
            # bbox_str exceeding the viewport. By convention, this bbox will be
            # around right limit of viewport (180°)
            ew_lim = (e_min + w_max) / 2.
            for bbox_str in metadata_lists['bbox_str']:
                bbox_geom = ogr.CreateGeometryFromWkt(bbox_str)
                _west, _east, _south, _north = bbox_geom.GetEnvelope()
                south = min([south, _south])
                north = max([north, _north])
                if (_west + _east) / 2. < ew_lim:
                    _west += 2 * vright
                    _east += 2 * vright
                west = min([west, _west])
                east = max([east, _east])
        else:
            # no xIDL detected, if some bbox_str exceed viewport (in x):
            # - with cylindric proj: final bbox_str will be the viewport (in x)
            # - else: it raises an error
            for bbox_str in metadata_lists['bbox_str']:
                bbox_geom = ogr.CreateGeometryFromWkt(bbox_str)
                _west, _east, _south, _north = bbox_geom.GetEnvelope()
                south = min([south, _south])
                north = max([north, _north])
                if _west < vleft or _east > vright:
                    if not cylindric_proj:
                        raise Exception('Individual bbox_str outside viewport.')
                    west = vleft
                    east = vright
                else:
                    west = min([west, _west])
                    east = max([east, _east])
    bbox = {"west": west, "north": north, "south": south, "east": east}
    metadata['bbox_str'] = "POLYGON(({west:f} {north:f},{east:f} {north:f},"\
                           "{east:f} {south:f},{west:f} {south:f},"\
                           "{west:f} {north:f}))".format(**bbox)

    # NOT MERGED : cropextent, cropneeded, ispaletted, isrgb, max_zoom_extent,
    # max_zoom_resolution, nb_bands, nodatavalues, output_options,
    # real_shape_str, use_gcp
    # FILLED IN merge() : portal and output_path

    return metadata, bbox


def merge(datagroup_dir, cfg):
    """ """

    metadata, bbox = merge_metadata(datagroup_dir, cfg)

    if cfg['output_proj'] != int(metadata['syntool_id'].split('_')[0]):
        raise Exception('Config projection does not match tiles projection.')

    # Create workspace
    workspace_root = cfg.get('workspace_root', os.getcwd())
    workspace = os.path.join(workspace_root, metadata['syntool_id'],
                             metadata['dataset'])
    if not os.path.exists(workspace):
        try:
            os.makedirs(workspace)
        except OSError:
            _, e, _ = sys.exc_info()
            if e.errno != errno.EEXIST:
                raise

    if 'vectorfield' == metadata['output_format']:
        pattern = os.path.join(datagroup_dir, '*', 'vectorFieldLayer.png')
        tiles_path = glob.glob(pattern)
        pngs = {}
        for png_path in tiles_path:
            dataset_dir = os.path.dirname(png_path)
            metadata_path = os.path.join(dataset_dir, 'metadata.json')
            if not os.path.exists(metadata_path):
                msg = 'Missing metadata.json in {}'.format(dataset_dir)
                raise Exception(msg)
            with open(metadata_path, 'r') as f:
                png_meta = json.load(f)
            pngs[png_path] = png_meta
        output_path = merge_vectorfields(pngs, workspace, metadata, bbox, cfg)
    else:
        output_path = merge_tiles(datagroup_dir, workspace)
    metadata['output_path'] = output_path

    metadata_path = os.path.join(workspace, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    return metadata_path
