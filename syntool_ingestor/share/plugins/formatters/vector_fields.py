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
import PIL.Image
import PIL.PngImagePlugin
import pyproj
import numpy
import logging
import traceback
from osgeo import osr, gdal, ogr
try:
    # Use of subprocess32 (backport from Python 3.2) is encouraged by Python
    # official documentation.
    # See https://docs.python.org/2/library/subprocess.html
    import subprocess32 as subprocess
except ImportError:
    import subprocess

from syntool_ingestor import load_dataset
from syntool_ingestor.interfaces import IFormatterPlugin


logger = logging.getLogger(__name__)


## NEW metadata.py
## Before
# def get_proj4(epsg, src_meta):
#     """"""
#     shape_wkt = src_meta['real_shape_str']
#     shape_geom = ogr.CreateGeometryFromWkt(shape_wkt)
#     shape_geom_type = shape_geom.GetGeometryType()
#     srs = osr.SpatialReference()
#     srs.ImportFromEPSG(int(epsg))
#     proj4 = srs.ExportToProj4()
#     if shape_geom_type == ogr.wkbGeometryCollection:
#         proj4 += ' +over'
#     return proj4
## Now
def get_proj4(epsg, plusover=False):
    """"""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    proj4 = srs.ExportToProj4()
    if plusover == True:
        proj4 += ' +over'
    return proj4
## \NEW metadata.py

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
        envs = numpy.array(envs)
        env = [0, 0, 0, 0]
        env[2] = envs[:, 2].min()
        env[3] = envs[:, 3].max()
        shape_lon = [lonlat[0] for lonlat in lonlat_shape]
        shape_lonmin = min(shape_lon)
        shape_lonmax = max(shape_lon)
        ileft = (envs[:, 0] + envs[:, 1]).argmin()
        iright = 1 - ileft
        if shape_lonmax > 180. and shape_lonmin >= -180.:
            # left shape -> shift / right shape -> keep
            env[0] = envs[iright, 0]
            env[1] = envs[ileft, 1] + 2 * abs(envs[ileft, 0])
        elif shape_lonmin < -180. and shape_lonmax <= 180.:
            # left shape -> keep / right shape -> shift
            env[0] = envs[iright, 0] - 2 * abs(envs[iright, 1])
            env[1] = envs[ileft, 1]
        else:
            raise Exception('Unexpected lonlat shape.')
    else:
        raise Exception('Shape geometry type not expected.')
    return [env[0], env[2], env[1], env[3]]


def epsg2wkt(epsg):
    """Translate EPSG code into a WKT"""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    return srs.ExportToWkt()


def guess_resolutions(input_ds, input_proj, output_proj):
    """Use GDAL to infer x and y resolutions."""
    src_wkt = epsg2wkt(input_proj)
    dst_wkt = epsg2wkt(output_proj)
    dst_ds = gdal.AutoCreateWarpedVRT(input_ds, src_wkt, dst_wkt)
    dst_geotr = dst_ds.GetGeoTransform()
    return (abs(dst_geotr[1]), abs(dst_geotr[5]))


def get_aligned_extent(cfg, src_meta, shape_extent):
    """Pad the shape extent so that its coordinates in the result extent is
    a multiple of the resolution. Required for PNGs that must be merged."""
    extent = [float(x) for x in cfg['extent'].split(' ')]
    if all([x == y for x, y in zip(extent, shape_extent)]):
        return extent

    shape_left, shape_bottom, shape_right, shape_top = shape_extent
    vport_bottom, vport_left, vport_top, vport_right = extent
    vport_hextent = vport_right - vport_left
    vport_vextent = vport_top - vport_bottom

    x_res = float(src_meta['x_res'])
    y_res = float(src_meta['y_res'])

    left = numpy.floor(x_res * (shape_left - vport_left) / vport_hextent)
    top = numpy.ceil(y_res * (shape_top - vport_top) / vport_vextent)
    right = numpy.ceil(x_res * (shape_right - vport_right) / vport_hextent)
    bottom = numpy.floor(y_res * (shape_bottom - vport_bottom) / vport_vextent)

    _left = vport_left + vport_hextent * left / x_res
    _top = vport_top + vport_vextent * top / y_res
    _right = vport_right + vport_hextent * right / x_res
    _bottom = vport_bottom + vport_vextent * bottom / y_res
    return [_top, _right, _bottom, _left]


def get_resolutions(f, res_str, input_proj, output_proj):
    """ """
    x_res = None
    y_res = None
    if res_str is not None:
        if 'x' in res_str:
            x_res, y_res = map(float, res_str.split('x', 2))
        else:
            x_res = float(res_str)
            y_res = x_res
    else:
        x_res, y_res = guess_resolutions(f, input_proj, output_proj)
    return x_res, y_res


def x_idl_workaround(shape_wkt):
    """Split the bounding box of shapes which corss the IDL."""
    XIDL_FIX_LON_DELTA = 0
    w_bbox = None
    e_bbox = None
    shape_geom = ogr.CreateGeometryFromWkt(shape_wkt)
    if ogr.wkbGeometryCollection == shape_geom.GetGeometryType() and \
       2 == shape_geom.GetGeometryCount():
        # West
        l0, r0, b0, t0 = shape_geom.GetGeometryRef(0).GetEnvelope()
        # East
        l1, r1, b1, t1 = shape_geom.GetGeometryRef(1).GetEnvelope()

        if l0 + r0 > l1 + r1:
            # Switch coordinates so that
            # l0, r0, t0, b0 are the coordinates of the western shape
            # l1, r1, t1, b1 are the coordinates of the eastern shape
            l0, r0, t0, b0, l1, r1, t1, b1 = l1, r1, t1, b1, l0, r0, t0, b0

        logger.debug('Checking XIDL...')
        logger.debug('{} {} {} {} vs {} {} {} {}'.format(l0, r0, b0, t0,
                                                         l1, r1, b1, t1))

        if XIDL_FIX_LON_DELTA + r0 < l1:
            bbox_pattern = 'POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))'
            w_bbox = bbox_pattern.format(l0, t0, r0, t0, r0, b0, l0, b0,
                                         l0, t0)
            e_bbox = bbox_pattern.format(l1, t1, r1, t1, r1, b1, l1, b1,
                                         l1, t1)
    return w_bbox, e_bbox


class VectorFieldsPlugin(IFormatterPlugin):
    """ """

    def raw2png(self, uv_path, workspace, cfg, src_meta, png_meta):
        # Project data in the output projection
        warp_path = self.warp(uv_path, workspace, cfg, src_meta)

        # Extract projected data
        color = None
        channels = []
        png_mode = 'LA'
        with load_dataset(warp_path) as dataset:
            modulus = self._extract_modulus(dataset, cfg, src_meta)
            png_meta['modulus_channel'] = 1
            channels.append(modulus)

            angle = self._extract_angle(dataset, cfg, src_meta)
            png_meta['angle_channel'] = 2
            channels.append(angle)

            if 'color_channel' in src_meta:
                color = self._extract_color(dataset, cfg, src_meta)
                png_meta['color_channel'] = 3
                channels.append(color)
                png_mode = 'RGB'

        # Create PNG metadata
        img_channels = [PIL.Image.fromarray(ch, 'L') for ch in channels]
        image = PIL.Image.merge(png_mode, img_channels)

        # Update PNG metadata
        reserved = ('interlace', 'gamma', 'dpi', 'transparency', 'aspect')
        metastr = {k: str(v) for k, v in png_meta.items()}
        image.info.update(metastr)
        metapng = PIL.PngImagePlugin.PngInfo()
        for key, value in image.info.iteritems():
            if key not in reserved:
                metapng.add_text(key, value, 0)

        # Save PNG
        output_path = os.path.join(workspace, 'vectorFieldLayer.png')
        image.save(output_path, 'PNG', pnginfo=metapng)
        return output_path, warp_path

    def make_uv(self, dataset, workspace, cfg, src_meta):
        """Replace modulus/angle bands with u/v bands in a new geotiff"""
        nodatavalues = src_meta.get('nodatavalues', [])

        # Get modulus
        modulus_channel = src_meta['modulus_channel']
        modulus_offset = src_meta['modulus_offset']
        modulus_scale = src_meta['modulus_scale']
        modulus_band = dataset.GetRasterBand(modulus_channel)
        modulus_array = modulus_band.ReadAsArray()
        modulus = modulus_array * modulus_scale + modulus_offset

        # Get angle
        angle_channel = src_meta['angle_channel']
        angle_offset = src_meta['angle_offset']
        angle_scale = src_meta['angle_scale']
        angle_band = dataset.GetRasterBand(angle_channel)
        angle_array = angle_band.ReadAsArray()
        angle = angle_array * angle_scale + angle_offset

        # No data values
        modulus_nodatavalue = None
        angle_nodatavalue = None
        color_nodatavalue = src_meta.get('color_nodata_value', None)
        if 0 < len(nodatavalues):
            modulus_nodatavalue = nodatavalues[modulus_channel - 1]
            angle_nodatavalue = nodatavalues[angle_channel - 1]

        # Compute u/v
        u = modulus * numpy.cos(angle * numpy.pi / 180.)
        v = modulus * numpy.sin(angle * numpy.pi / 180.)
        if modulus_nodatavalue is None:
            uv_nodatavalue = None
            uv_array_max = 255
            modulus_max = 255 * modulus_scale + modulus_offset
        else:
            uv_nodatavalue = 255
            uv_array_max = 254
            if modulus_nodatavalue == 255:
                modulus_max = 254 * modulus_scale + modulus_offset
            else:
                modulus_max = 255 * modulus_scale + modulus_offset
        uv_offset = -modulus_max
        uv_scale = 2. * modulus_max / uv_array_max
        u_array = numpy.round((u - uv_offset) / uv_scale).astype('uint8')
        v_array = numpy.round((v - uv_offset) / uv_scale).astype('uint8')
        if uv_nodatavalue is not None:
            nodata = numpy.where((modulus_array == modulus_nodatavalue) |
                                 (angle_array == angle_nodatavalue))
            u_array[nodata] = uv_nodatavalue
            v_array[nodata] = uv_nodatavalue

        # Write a new tiff
        uv_path = os.path.join(workspace, 'uv.tiff')
        driver = gdal.GetDriverByName('GTiff')
        ysize, xsize = u_array.shape
        nband = 2 if 'color_channel' not in src_meta else 3
        uv_dataset = driver.Create(uv_path, xsize, ysize, nband, gdal.GDT_Byte)
        if dataset.GetGCPCount() > 0:
            uv_dataset.SetGCPs(dataset.GetGCPs(), dataset.GetGCPProjection())
        else:
            uv_dataset.SetProjection(dataset.GetProjection())
            uv_dataset.SetGeoTransform(dataset.GetGeoTransform())
        u_channel = 1
        v_channel = 2
        u_band = uv_dataset.GetRasterBand(u_channel)
        v_band = uv_dataset.GetRasterBand(v_channel)
        u_band.WriteArray(u_array)
        v_band.WriteArray(v_array)
        if uv_nodatavalue is not None:
            u_band.SetNoDataValue(uv_nodatavalue)
            v_band.SetNoDataValue(uv_nodatavalue)

        color_channel = src_meta.get('color_channel', None)
        if color_channel is not None:
            # read
            color_band = dataset.GetRasterBand(color_channel)
            color_array = color_band.ReadAsArray()
            # write
            color_channel = 3
            color_band = uv_dataset.GetRasterBand(color_channel)
            color_band.WriteArray(color_array)
            if color_nodatavalue is not None:
                color_band.SetNoDataValue(color_nodatavalue)
        uv_dataset = None

        # Update src_meta
        src_meta['u_channel'] = u_channel
        src_meta['v_channel'] = v_channel
        src_meta['uv_offset'] = uv_offset
        src_meta['uv_scale'] = uv_scale
        src_meta['uv_nodatavalue'] = uv_nodatavalue
        if 'color_channel' in src_meta:
            src_meta['color_channel'] = color_channel

        return uv_path

    def warp(self, input_path, workspace, cfg, src_meta):
        """Warp the input file in the portal projection.
        Output size depends on the value specified with the "resolution"
        option (guessed using GDAL if no value is provided).
        Data is cropped to fit the portal extent."""

        # gdalwarp fails to interpret some input projections if they are passed
        # as EPSG codes, but will work correctly if they are passee as Proj4
        # definitions.
        ## NEW metadata.py
        ## Before
        # input_proj4 = get_proj4(cfg['input_proj'], src_meta)
        # output_proj4 = get_proj4(cfg['output_proj'], src_meta)
        ## Now
        input_proj4 = get_proj4(cfg['input_proj'])
        output_proj4 = get_proj4(cfg['output_proj'], plusover=src_meta['warp_infos']['+over'])
        ## \NEW metadata.py

        out_ext = [str(e) for e in cfg['warped_extent']]
        resampling = cfg['output_options'].get('resampling', 'bilinear')
        use_gcp = src_meta.get('use_gcp', False)
        uv_nodatavalue = src_meta.get('uv_nodatavalue', None)
        color_nodatavalue = src_meta.get('color_nodata_value', None)
        nodatavalues = None
        if uv_nodatavalue is not None:
            nodatavalues = [uv_nodatavalue]*2
            if color_nodatavalue is not None:
                nodatavalues.append(color_nodatavalue)

        warped_path = os.path.join(workspace, 'warp.vrt')
        args = [input_path, warped_path]

        opts = ['-overwrite',
                '-of', 'VRT',
                '-r', resampling,
                '-s_srs', input_proj4,
                '-t_srs', output_proj4,
                '-te', out_ext[0], out_ext[1], out_ext[2], out_ext[3],
                '-tr', src_meta['x_res'], src_meta['y_res'],
                '-wo', 'SAMPLE_GRID=YES',
                '-wo', 'SOURCE_EXTRA=100',
                '-wo', 'NUM_THREADS=ALL_CPUS',
                '-multi']

        if nodatavalues is not None:
            opts.extend(['-dstnodata', ' '.join(map(str, nodatavalues))])

        # Thin plate splines can only be used if the dataset shape is described
        # with GCPs.
        if use_gcp:
            opts.append('-tps')

        call = ['gdalwarp']
        call.extend(opts)
        call.extend(args)

        logger.debug(' '.join(call))
        try:
            subprocess.check_call(call)
        except subprocess.CalledProcessError as e:
            logger.error('Could not warp.')
            logger.debug(traceback.print_exc())
            raise
        return warped_path

    def _extract_modulus(self, dataset, cfg, src_meta):
        """Extract data for vector norm from dataset."""

        u_channel = src_meta['u_channel']
        v_channel = src_meta['v_channel']
        uv_scale = src_meta['uv_scale']
        uv_offset = src_meta['uv_offset']
        uv_nodatavalue = src_meta.get('uv_nodatavalue', None)
        modulus_channel = src_meta['modulus_channel']
        offset = src_meta['modulus_offset']
        scale = src_meta['modulus_scale']
        nodatavalues = src_meta['nodatavalues']

        u_band = dataset.GetRasterBand(u_channel)
        u_array = u_band.ReadAsArray()
        u = u_array * uv_scale + uv_offset

        v_band = dataset.GetRasterBand(v_channel)
        v_array = v_band.ReadAsArray()
        v = v_array * uv_scale + uv_offset

        modulus = numpy.sqrt(u**2. + v**2.)
        modulus_array = numpy.round((modulus - offset) / scale).astype('uint8')
        if uv_nodatavalue is not None:
            nodatavalue = nodatavalues[modulus_channel - 1]
            nodata = numpy.where(u_array == uv_nodatavalue)
            modulus_array[nodata] = nodatavalue
        return modulus_array

    def _extract_angle(self, dataset, cfg, src_meta):
        """Extract data for vector direction from dataset.
        Angles are expected to be counter-clockwise from east in degrees.
        Angles must be recomputed to match the output projection."""

        # Get metadata
        u_channel = src_meta['u_channel']
        v_channel = src_meta['v_channel']
        uv_offset = src_meta['uv_offset']
        uv_scale = src_meta['uv_scale']
        uv_nodatavalue = src_meta['uv_nodatavalue']
        angle_channel = src_meta['angle_channel']
        angle_offset = src_meta['angle_offset']
        angle_scale = src_meta['angle_scale']

        # Get projection settings
        x_origin, pixel_col_width, pixel_row_width, y_origin, \
            pixel_col_height, pixel_row_height = dataset.GetGeoTransform()

        # Get u/v and projected x/y
        u_band = dataset.GetRasterBand(u_channel)
        v_band = dataset.GetRasterBand(v_channel)
        u_array = u_band.ReadAsArray()
        v_array = v_band.ReadAsArray()
        output_shape = u_array.shape
        x_size, y_size = output_shape
        if uv_nodatavalue is not None:
            valid = numpy.where(u_array != uv_nodatavalue)
            u = u_array[valid]
            v = v_array[valid]
            x1 = valid[1] * pixel_col_width + valid[0] * pixel_row_width
            y1 = valid[1] * pixel_col_height + valid[0] * pixel_row_height
        else:
            u = u_array
            v = v_array
            col = numpy.arange(y_size)
            row = numpy.arange(x_size)
            x1 = numpy.tile(col * pixel_col_width, x_size) + \
                numpy.repeat(row * pixel_row_width, y_size)
            y1 = numpy.tile(col * pixel_col_height, x_size) + \
                numpy.repeat(row * pixel_row_height, y_size)

        u = u * uv_scale + uv_offset
        v = v * uv_scale + uv_offset
        x1 = x_origin + x1
        y1 = y_origin + y1

        # Compute angle from u/v
        real_angle = numpy.degrees(numpy.arctan2(v, u))
        del u, v, u_array, v_array

        # Reverse projection to get (lon,lat) for each (x,y)
        output_proj = get_proj4(cfg['output_proj'], src_meta)
        proj = pyproj.Proj(output_proj)

        # Get lon/lat for each x/y
        lons1, lats1 = proj(x1, y1, inverse=True)

        # Use an arbitrary distance (1km)
        dists = numpy.ndarray(shape=lons1.shape)
        dists.fill(1000.0)

        # pyproj.Geod.fwd expects bearings to be clockwise angles from north
        # (in degrees).
        fixed_angles = 90.0 - real_angle

        # Interpolate a destination from grid point and data direction
        geod = pyproj.Geod(ellps='WGS84')
        lons2, lats2, bearings2 = geod.fwd(lons1, lats1, fixed_angles, dists)
        del bearings2, lons1, lats1, real_angle, fixed_angles, dists

        # Warp destination to output projection
        x2, y2 = proj(lons2, lats2)
        del lons2, lats2

        # Fix issue when an interpolated point is not reprojected in the same
        # longitude range as it origin (applies to cylindric projections only)
        ## NEW metadata.py
        ## Before
        #if int(cfg['output_proj']) in (4326, 3857, 900913):
        ## Now
        if cfg['output_proj_type'] == 'cylindric':
        ## \NEW metadata.py
            extent = [float(x) for x in cfg['extent'].split(' ')]
            vport_bottom, vport_left, vport_top, vport_right = extent
            vport_x_extent = vport_right - vport_left
            x2 = numpy.mod(x2 - (x1 + vport_left), vport_x_extent) \
                + (x1 + vport_left)

        # Compute angle in output projection between [0, 360] degrees
        projected_angles = numpy.arctan2(y2 - y1, x2 - x1)
        ranged_angles = numpy.mod(360 + numpy.degrees(projected_angles), 360)
        del x1, y1, x2, y2

        # Rescale angle
        scaled_angles = (ranged_angles - angle_offset) / angle_scale
        scaled_angles = numpy.round(scaled_angles).astype('uint8')

        # Rebuild matrix from flattened data
        if uv_nodatavalue is not None:
            nodatavalue = src_meta['nodatavalues'][angle_channel - 1]
            angle = numpy.empty(output_shape, dtype='uint8')
            angle.fill(nodatavalue)
            angle[valid] = scaled_angles
            return angle
        else:
            scaled_angles.shape = output_shape
            return scaled_angles

    def _extract_color(self, dataset, cfg, src_meta):
        """Extract data for vector color from dataset."""
        color_channel = src_meta['color_channel']
        color_band = dataset.GetRasterBand(color_channel)
        color_array = color_band.ReadAsArray()
        return color_array

    def can_handle(self, target_format):
        """ """
        return 'vectorfield' == target_format

    def get_output_id(self, cfg):
        """ """
        return '_vectorfield'

    def get_representation_type(self, cfg):
        """ """
        return 'VECTOR_FIELD'

    def create_representation(self, input_path, input_data, workspace, cfg,
                              src_meta):
        """ """
        temporary_files = []

        input_proj = cfg['input_proj']
        output_proj = cfg['output_proj']
        # Keep offset and scale from original file because gdal_warp eats
        # them...
        opts = cfg['output_options']
        res_str = opts.get('resolution', None)
        modulus_channel = int(opts.get('modulus-band', 1))
        modulus_min = opts.get('scale-min', None)
        modulus_max = opts.get('scale-max', None)
        angle_channel = int(opts.get('angle-band', 2))
        color_channel = opts.get('color-band', None)
        color_min = opts.get('color-min', None)
        color_max = opts.get('color-max', None)
        color_info = {}
        with load_dataset(input_path) as f:
            modulus_band = f.GetRasterBand(modulus_channel)
            modulus_scale = modulus_band.GetScale()
            modulus_offset = modulus_band.GetOffset()
            modulus_nodatavalue = modulus_band.GetNoDataValue()
            angle_band = f.GetRasterBand(angle_channel)
            angle_scale = angle_band.GetScale()
            angle_offset = angle_band.GetOffset()
            if color_channel is not None:
                color_channel = int(color_channel)
                color_band = f.GetRasterBand(color_channel)
                color_scale = color_band.GetScale()
                color_offset = color_band.GetOffset()
                color_nodatavalue = color_band.GetNoDataValue()
                color_info = {'color_channel': color_channel,
                              'color_offset': color_offset,
                              'color_scale': color_scale,
                              'color_nodata_value': color_nodatavalue}

            if modulus_min is None:
                byte_min = 0 if modulus_nodatavalue != 0 else 1
                modulus_min = byte_min * modulus_scale + modulus_offset
            if modulus_max is None:
                byte_max = 255 if modulus_nodatavalue != 255 else 254
                modulus_max = byte_max * modulus_scale + modulus_offset

            if color_channel is not None:
                if color_min is None:
                    byte_min = 0 if color_nodatavalue != 0 else 1
                    color_min = byte_min * color_scale + color_offset
                if color_max is None:
                    byte_max = 255 if color_nodatavalue != 255 else 254
                    color_max = byte_max * color_scale + color_offset
                color_info['color_min'] = float(color_min)
                color_info['color_max'] = float(color_max)

            x_res, y_res = get_resolutions(f, res_str, input_proj, output_proj)

            png_meta = {'angle_offset': angle_offset,
                        'angle_scale': angle_scale,
                        'angle_channel': angle_channel,
                        'modulus_offset': modulus_offset,
                        'modulus_scale': modulus_scale,
                        'modulus_channel': modulus_channel,
                        'scale_min': float(modulus_min),
                        'scale_max': float(modulus_max),
                        'nodata_value': modulus_nodatavalue,
                        'x_res': '{}'.format(x_res),
                        'y_res': '{}'.format(y_res)}
            png_meta.update(color_info)

            # Merge with source metadata
            src_meta.update(png_meta)

            # Extract data
            uv_path = self.make_uv(f, workspace, cfg, src_meta)
            temporary_files.append(uv_path)

        ## NEW metadata.py
        ## Before
        # Adapt extent for warp
        # align_bbox = opts.get('aligned-bbox', 'False')
        # if align_bbox.lower() in ('true', 'yes'):
        #     shape_geom = ogr.CreateGeometryFromWkt(src_meta['real_shape_str'])
        #     lonlat_shape = src_meta['lonlat_shape']
        #     shape_extent = get_shape_extent(shape_geom, lonlat_shape)
        #     t, r, b, l = get_aligned_extent(cfg, src_meta, shape_extent)
        #     cfg['warped_extent'] = [l, b, r, t]
        # else:
        #     shape_str = src_meta['real_shape_str']
        #     shape_geom = ogr.CreateGeometryFromWkt(shape_str)
        #     west, east, south, north = shape_geom.GetEnvelope()
        #     cfg['warped_extent'] = [west, south, east, north]
        ## Now
        # Adapt extent/resolution for warp
        left, bottom, right, top = src_meta['warp_infos']['extent']
        align_bbox = opts.get('aligned-bbox', 'False')
        if align_bbox.lower() in ('true', 'yes'): # Required for merging PNGs
            # Adapt resolution to make a grid in viewport
            viewport = [float(x) for x in cfg['viewport'].split(' ')]
            vleft, vbottom, vright, vtop = viewport
            xsize = numpy.ceil((vright - vleft) / float(src_meta['x_res']))
            xres = (vright - vleft) / xsize
            ysize = numpy.ceil((vtop - vbottom) / float(src_meta['y_res']))
            yres = (vtop - vbottom) / ysize
            # Adapt extent to this grid
            left = vleft + numpy.floor((left - vleft) / xres) * xres
            right = vleft + numpy.ceil((right - vleft) / xres) * xres
            bottom = vbottom + numpy.floor((bottom - vbottom) / yres) * yres
            top = vbottom + numpy.ceil((top - vbottom) / yres) * yres
            src_meta['x_res'] = '{}'.format(xres)
            src_meta['y_res'] = '{}'.format(yres)
            cfg['warped_extent'] = [left, bottom, right, top]
        else:
            # Change resolution in order to make gdalwarp preserving extent
            xsize = numpy.ceil((right - left) / float(src_meta['x_res']))
            xres = (right - left) / xsize
            ysize = numpy.ceil((top - bottom) / float(src_meta['y_res']))
            yres = (top - bottom) / ysize
            src_meta['x_res'] = '{}'.format(xres)
            src_meta['y_res'] = '{}'.format(yres)
            cfg['warped_extent'] = [left, bottom, right, top]
        ## \NEW metadata.py

        output_path, warp_path = self.raw2png(uv_path, workspace, cfg,
                                              src_meta, png_meta)
        temporary_files.append(warp_path)

        # Workaround for cross-IDL
        ## NEW metadata.py
        ## Before
        # shape_wkt = src_meta['shape_str']
        # w_bbox, e_bbox = x_idl_workaround(shape_wkt)
        # if None not in (w_bbox, e_bbox):
        #     src_meta['w_bbox'] = w_bbox
        #     src_meta['e_bbox'] = e_bbox
        ## Now
        if src_meta['bbox_infos']['xIDL'] == True:
            # bboxes contain [xmin, ymin, xmax, ymax]
            bbox_pattern = 'POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))'
            l0, b0, r0, t0 = src_meta['bbox_infos']['w_bbox']
            src_meta['w_bbox'] = bbox_pattern.format(l0, t0, r0, t0, r0, b0,
                                                     l0, b0, l0, t0)
            l1, b1, r1, t1 = src_meta['bbox_infos']['e_bbox']
            src_meta['e_bbox'] = bbox_pattern.format(l1, t1, r1, t1, r1, b1,
                                                     l1, b1, l1, t1)
        ## \NEW metadata.py

        # Clean
        if not cfg.get('keep_intermediary_files', False):
            def removal_filter(x):
                return x != input_path and os.path.exists(x)
            to_remove = filter(removal_filter, temporary_files)
            map(os.remove, to_remove)
            msg = 'These temporary files have been removed: {}'
            msg = msg.format(to_remove)
            logger.debug(msg)

        resolutions = []
        zooms = [0]

        # Set bbox_str: it will be used for placing the PNG on the map
        # We use warped_extent since it is actually PNG extent (result of warp)
        # (bbox_infos['bbox'] should not be used here because it corresponds to
        # warp_infos['extent'] which may have been adapted to give warped_extent)
        west, south, east, north = cfg['warped_extent']
        ## NEW metadata.py
        if cfg['output_proj_type'] == 'cylindric':
            viewport = [float(x) for x in cfg['viewport'].split(' ')]
            if east <= viewport[0]:
                west = west + 2 * viewport[2]
                east = east + 2 * viewport[2]
            elif west >= viewport[2]:
                west = west - 2 * viewport[2]
                east = east - 2 * viewport[2]
            elif west < viewport[0]:
                # xIDL: by convention we set bbox around 180° rather than -180°
                west = west + 2 * viewport[2]
                east = east + 2 * viewport[2]
        ## \NEW metadata.py
        bbox = {"w": west, "n": north, "s": south, "e": east}
        bbox_str = 'POLYGON(({w:f} {n:f},{e:f} {n:f},{e:f} {s:f},' \
                   '{w:f} {s:f},{w:f} {n:f}))'.format(**bbox)

        return {'resolutions': resolutions,
                'min_zoom_level': min(zooms),
                'max_zoom_level': max(zooms),
                'bbox_str': bbox_str,
                'output_path': os.path.abspath(output_path)}
