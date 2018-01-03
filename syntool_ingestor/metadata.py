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

import logging
import ConfigParser
from osgeo import gdal, osr, ogr
import numpy as np
import pyproj
from syntool_ingestor.time_range import get_range

logger = logging.getLogger(__name__)

# Limit for the number of points in the shape
MAX_SHAPE_POINTS = 30
MAX_SHAPE_POINTS_LIMIT = 100
MIN_SHAPE_RES = 25000.

STEREO_PROJ_NORTH = [3411, 3413]
STEREO_PROJ_SOUTH = [3412, 3976]
STEREO_PROJ = STEREO_PROJ_NORTH + STEREO_PROJ_SOUTH
CYLINDRIC_PROJ = [4326, 3857, 900913] + \
                 range(32601, 32661) + \
                 range(32701, 32761)

def _get_bbox_wkt(left, bottom, right, top):
    """ """
    bbox = { 'left': left
           , 'top': top
           , 'right': right
           , 'bottom': bottom
           }
    wkt = 'POLYGON(({left} {top}, {right} {top}, {right} {bottom}, {left} {bottom}, {left} {top}))'.format(**bbox)
    return wkt

def _get_poly_wkt(pts):
    """ """
    coords = ['{} {}'.format(xy[0], xy[1]) for xy in pts]
    wkt = 'POLYGON(({}))'.format(', '.join(coords))
    return wkt

def _points_to_xy(points, transformer, input_proj, to_proj):
    """ """
    xyi = np.array(transformer.TransformPoints(0, points)[0])[:, 0:2]
    logger.debug('pixel: {}'.format(np.array(points)[:, 0]))
    logger.debug('line: {}'.format(np.array(points)[:, 1]))
    logger.debug('input x: {}'.format(xyi[:, 0]))
    logger.debug('input y: {}'.format(xyi[:, 1]))
    if input_proj != to_proj:
        proji = pyproj.Proj('+init=EPSG:{}'.format(input_proj))
        if to_proj == 900913:
            projo = pyproj.Proj('+init=EPSG:{}'.format(3857))
        else:
            projo = pyproj.Proj('+init=EPSG:{}'.format(to_proj))
        xyt = pyproj.transform(proji, projo, xyi[:, 0], xyi[:, 1])
        xyt = np.array(xyt).transpose()
        logger.debug('to x: {}'.format(xyt[:, 0]))
        logger.debug('to y: {}'.format(xyt[:, 1]))
    else:
        xyt = xyi
    return xyt

def _get_shape(dataset, input_proj, shape_proj=4326, dist_proj=4326, ndist=10,
               transformer=None, min_shape_res=MIN_SHAPE_RES,
               max_shape_points=MAX_SHAPE_POINTS):
    """ """
    if transformer is None:
        transformer = gdal.Transformer(dataset, None, ['MAX_GCP_ORDER=-1'])
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    if dist_proj == 4326:
        geod = pyproj.Geod(ellps='WGS84')

    # Loop on bottom, right, top, left
    xlims = [[0, xsize], [xsize, xsize], [xsize, 0], [0, 0]]
    ylims = [[ysize, ysize], [ysize, 0], [0, 0], [0, ysize]]
    shape = []
    for xlim, ylim in zip(xlims, ylims):
        pts = np.array([np.linspace(xlim[0], xlim[1], ndist+1),
                        np.linspace(ylim[0], ylim[1], ndist+1)]).transpose()
        xyd = _points_to_xy(pts, transformer, input_proj, dist_proj)
        if dist_proj == 4326:
            dists = geod.inv(xyd[:-1, 0], xyd[:-1, 1],
                             xyd[1:, 0], xyd[1:, 1])[2]
        else:
            dists = np.sqrt((xyd[1:, 0]-xyd[:-1, 0])**2+\
                            (xyd[1:, 1]-xyd[:-1, 1])**2)
        dist = np.sum(dists)
        if dist == 0.:
            logger.warning('[WARNING] null distance in shape computation.')
            npts = 2
        else:
            npts = np.ceil(dist / min_shape_res).astype('int') + 1
            npts = min(max_shape_points, npts)
        pts = np.array([np.linspace(xlim[0], xlim[1], npts),
                        np.linspace(ylim[0], ylim[1], npts)]).transpose()
        xys = _points_to_xy(pts, transformer, input_proj, shape_proj)
        shape.extend([[x, y] for x, y in zip(xys[:-1, 0], xys[:-1, 1])])
    shape.append(shape[0])

    return shape

def _clean_geometry(geom):
    """ """
    geom_type = geom.GetGeometryType()
    if geom_type == ogr.wkbPolygon:
        clean_geom = geom.Clone()
    elif geom_type == ogr.wkbGeometryCollection or \
         geom_type == ogr.wkbMultiPolygon:
        ngeom = geom.GetGeometryCount()
        ind = [x for x in range(ngeom) if geom.GetGeometryRef(x).GetGeometryType() == ogr.wkbPolygon]
        if len(ind) == 0:
            raise Exception('Unexpected geometry.')
        elif len(ind) == 1:
            clean_geom = geom.GetGeometryRef(ind[0]).Clone()
        else:
            clean_geom = ogr.Geometry(ogr.wkbGeometryCollection)
            for i in ind:
                clean_geom.AddGeometry(geom.GetGeometryRef(i).Clone())
    else:
        raise Exception('Unexpected geometry.')
    return clean_geom

def _bbox_intersection(geom, bbox):
    """ """
    bbox_wkt = _get_bbox_wkt(*bbox)
    bbox_geom = ogr.CreateGeometryFromWkt(bbox_wkt)
    int_geom = bbox_geom.Intersection(geom)
    return int_geom

def _xtranslation(geom, offset):
    """ """
    trans_pts = [[xy[0]+offset, xy[1]] for xy in geom.GetGeometryRef(0).GetPoints()]
    trans_wkt = _get_poly_wkt(trans_pts)
    trans_geom = ogr.CreateGeometryFromWkt(trans_wkt)
    return trans_geom

def _xtranslation2(geom, offset):
    """ """
    ngeom = geom.GetGeometryCount()
    if ngeom == 0:
        for ipt, xy in enumerate(geom.GetPoints()):
            geom.SetPoint(ipt, xy[0] + offset, xy[1])
    else:
        for igeom in range(ngeom):
            _geom = geom.GetGeometryRef(igeom)
            _xtranslation2(_geom, offset)
    return

def _get_shape_geometry(shape, dataset, input_proj, output_proj):
    """ """
    if input_proj not in CYLINDRIC_PROJ + STEREO_PROJ:
        raise Exception('Is EPSG:{} cylindric or stereo ?'.format(input_proj))
    if output_proj not in CYLINDRIC_PROJ + STEREO_PROJ:
        raise Exception('Is EPSG:{} cylindric or stereo ?'.format(output_proj))
    shape_geom0 = ogr.CreateGeometryFromWkt(_get_poly_wkt(shape))

    # Create shape geometry in  lon/lat
    if input_proj in CYLINDRIC_PROJ:

        shape_env = shape_geom0.GetEnvelope()
        if output_proj in CYLINDRIC_PROJ:
            if shape_env[0] < -180 or shape_env[1] > 180 or \
               shape_env[2] < -85.6 or shape_env[3] > 85.6:
                shape_geom = _bbox_intersection(shape_geom0, [-180, -85.6, 180, 85.6])
                if shape_geom.IsEmpty():
                    raise Exception('Input data does not intersect [-180, -85.6, 180, 85.6].')
                if shape_env[0] < -180:
                    int_geom = _bbox_intersection(shape_geom0, [-540, -85.6, -180, 85.6])
                    shape_geom = shape_geom.Union(_xtranslation(int_geom, 360))
                if shape_env[1] > 180:
                    int_geom = _bbox_intersection(shape_geom0, [180, -85.6, 540, 85.6])
                    shape_geom = shape_geom.Union(_xtranslation(int_geom, -360))
                # tmp_geom = ogr.Geometry(ogr.wkbGeometryCollection)
                # int_geom = _bbox_intersection(shape_geom0, [-180, -90, 180, 90])
                # tmp_geom.AddGeometry(int_geom)
                # if shape_env[0] < -180:
                #     int_geom = _bbox_intersection(shape_geom0, [-540, -90, -180, 90])
                #     tmp_geom.AddGeometry(_xtranslation(int_geom, 360))
                # if shape_env[1] > 180:
                #     int_geom = _bbox_intersection(shape_geom0, [180, -90, 540, 90])
                #     tmp_geom.AddGeometry(_xtranslation(int_geom, -360))
                # shape_geom = _bbox_intersection(tmp_geom, [-180, -90, 180, 90])
            else:
                shape_geom = shape_geom0

        elif output_proj in STEREO_PROJ:
            do_convex_hull = (shape_env[1]-shape_env[0]) >= 360
            shape_geom = shape_geom0

    elif input_proj in STEREO_PROJ:

        if output_proj in CYLINDRIC_PROJ:
            transformer = gdal.Transformer(dataset, None, ['MAX_GCP_ORDER=-1'])
            xsize, ysize = dataset.RasterXSize, dataset.RasterYSize
            xpo, ypo = transformer.TransformPoints(1, [[0, 0]])[0][0][0:2]
            has_pole = xpo >= 0 and xpo <= xsize and ypo >= 0 and ypo <= ysize
            if has_pole:
                if input_proj in STEREO_PROJ_NORTH:
                    pole = 89.9
                elif input_proj in STEREO_PROJ_SOUTH:
                    pole = -89.9
                shape = shape[0:-1] # remove last (duplicated) point
                indmin = np.array(shape)[:, 0].argmin()
                indmax = np.array(shape)[:, 0].argmax()
                if indmax < indmin or (indmin == 0 and indmax == len(shape)-1):
                    shape = shape[::-1]
                    indmin = np.array(shape)[:, 0].argmin()
                if indmin != len(shape)-1:
                    for x in range(indmin+1):
                        shape.append(shape.pop(0))
                pt1, pt2 = shape[-1], shape[0]
                if pt1[0] != -180:
                    shape.append([-180., pt1[1]])
                shape.extend([[-180., pole], [180., pole]])
                if pt2[0] != 180:
                    shape.append([180., pt2[1]])
                shape.append(shape[0])
                shape_geom = ogr.CreateGeometryFromWkt(_get_poly_wkt(shape))
            else:
                shape_geom = shape_geom0

        elif output_proj in STEREO_PROJ:
            shape_geom = shape_geom0

    # Project shape geometry to output_proj
    shape_sr = osr.SpatialReference()
    shape_sr.ImportFromEPSG(4326)
    output_sr = osr.SpatialReference()
    output_sr.ImportFromEPSG(output_proj)
    shape_geom.AssignSpatialReference(shape_sr)
    shape_geom.TransformTo(output_sr)
    if input_proj in CYLINDRIC_PROJ and output_proj in STEREO_PROJ:
        if do_convex_hull:
            shape_geom = shape_geom.ConvexHull()
    shape_geom = _clean_geometry(shape_geom)
    logger.debug('Shape WKT: {}'.format(shape_geom.ExportToWkt()))
    return shape_geom

def _get_crop_info(shape_geom, viewport_geom):
    """ """
    if viewport_geom.Disjoint(shape_geom):
        raise Exception('Dataset is out of the portal bounding box')

    in_viewport = viewport_geom.Contains(shape_geom)
    shape_env = shape_geom.GetEnvelope()
    viewp_env = viewport_geom.GetEnvelope()
    full_width = shape_env[0] == viewp_env[0] and shape_env[1] == viewp_env[1]
    uniq_poly = shape_geom.GetGeometryType() == ogr.wkbPolygon
    crop_needed = (not in_viewport or full_width) and uniq_poly
    logger.debug('In viewport: {}'.format(in_viewport))
    logger.debug('Full width: {}'.format(full_width))
    logger.debug('Uniq polygon: {}'.format(uniq_poly))
    logger.debug('Crop needed: {}'.format(crop_needed))

    final_shape_geom = viewport_geom.Intersection(shape_geom)
    final_shape_geom = _clean_geometry(final_shape_geom)
    logger.debug('Final Shape WKT: {}'.format(final_shape_geom.ExportToWkt()))

    if not crop_needed:
        crop_extent = None
    else:
        shape_env = final_shape_geom.GetEnvelope()
        crop_extent = [shape_env[0], shape_env[2], shape_env[1], shape_env[3]]
    logger.debug('Crop extent: {}'.format(crop_extent))

    return (final_shape_geom, crop_needed, crop_extent)

def _project_geom(geom, input_srs, output_srs, plus_over=False):
    """ """
    if plus_over == True:
        output_proj4 = output_srs.ExportToProj4()
        output_proj4 += ' +over'
        output_srs.ImportFromProj4(output_proj4)
    proj_geom = geom.Clone()
    proj_geom.AssignSpatialReference(input_srs)
    proj_geom.TransformTo(output_srs)
    proj_geom.FlattenTo2D()
    # try:
    #     # Recent GDAL
    #     proj_geom.Set3D(False)
    # except AttributeError:
    #     # Old GDAL
    #     pass
    return proj_geom

def _get_output_shape(lonlat_shape, dataset, input_proj, output_proj, viewport_geom):
    """ """

    lonlat_geom = ogr.CreateGeometryFromWkt(_get_poly_wkt(lonlat_shape))
    lonlat_env = lonlat_geom.GetEnvelope()
    viewport_env = viewport_geom.GetEnvelope()
    lonlat_srs = osr.SpatialReference()
    lonlat_srs.ImportFromEPSG(4326)
    output_srs = osr.SpatialReference()
    output_srs.ImportFromEPSG(output_proj)

    bbox_infos = {'bbox': None,
                  'xIDL': False,
                  'w_bbox': None,
                  'e_bbox': None}
    warp_infos = {'extent': None,
                  '+over': False,
                  'center_long': None}

    # Cylindric -> Cylindric
    if input_proj in CYLINDRIC_PROJ and output_proj in CYLINDRIC_PROJ:
        if lonlat_env[0] < -540:
            raise Exception('Shape longitudes less than -540.')
        if lonlat_env[1] > 540:
            raise Exception('Shape longitudes greater than +540.')
        shift_geoms = []
        for shift in [-1, 0, 1]:
            shift_bbox = [-180 + shift * 360, -86., 180 + shift * 360, 86.]
            if lonlat_env[1] <= shift_bbox[0] or lonlat_env[0] >= shift_bbox[2]:
                continue
            _over_lonlat_geom = _bbox_intersection(lonlat_geom, shift_bbox)
            if _over_lonlat_geom.IsEmpty():
                continue
            _lonlat_geom = _over_lonlat_geom.Clone()
            if shift != 0:
                _xtranslation2(_lonlat_geom, -shift * 360)
            _output_geom = _project_geom(_lonlat_geom, lonlat_srs, output_srs)
            if viewport_geom.Disjoint(_output_geom):
                continue
            _output_geom = viewport_geom.Intersection(_output_geom)
            _over_output_geom = _output_geom.Clone()
            if shift != 0:
                _xtranslation2(_over_output_geom, shift * 2 * viewport_env[1])
            shift_geoms.append({'output': _output_geom,
                                'over': _over_output_geom,
                                'shift': shift})
        ngeoms = len(shift_geoms)
        logger.debug('Number of geometries: {}'.format(ngeoms))
        logger.debug('Shifts: {}'.format([s['shift'] for s in shift_geoms]))
        output_envs = [s['output'].GetEnvelope() for s in shift_geoms]
        logger.debug('Output envelopes: {}'.format(output_envs))
        over_envs = [s['over'].GetEnvelope() for s in shift_geoms]
        logger.debug('Over envelopes: {}'.format(over_envs))
        if ngeoms == 0:
            raise Exception('Input data is outside viewport.')
        elif ngeoms == 1:
            # Simple case: input data inside the viewport leads to one geometry
            # -> no IDL crossing here
            output_geom = shift_geoms[0]['output']
            output_geom = _clean_geometry(output_geom)
            output_env = output_geom.GetEnvelope()
            bbox_infos['bbox'] = [output_env[0], output_env[2], output_env[1], output_env[3]]
            over_env = shift_geoms[0]['over'].GetEnvelope()
            warp_infos['extent'] = [over_env[0], over_env[2], over_env[1], over_env[3]]
            if shift_geoms[0]['shift'] != 0:
                warp_infos['+over'] = True
        elif lonlat_env[1] - lonlat_env[0] >= 360.:
            # We expect here a global grid:
            # - either a grid covering more than 360° (with duplicated data)
            # - or a grid a little bit shifted from [-180, 180]
            # -> We do not expect/want a swath of 360° !
            output_geom = shift_geoms[0]['output']
            for igeom in range(1, ngeoms):
                output_geom = output_geom.Union(shift_geoms[igeom]['output'])
            output_geom = viewport_geom.Intersection(output_geom)
            output_geom = _clean_geometry(output_geom)
            output_env = output_geom.GetEnvelope()
            check_grid = output_geom.GetGeometryType() == ogr.wkbPolygon and \
                         output_env[0] == viewport_env[0] and \
                         output_env[1] == viewport_env[1]
            if not check_grid:
                raise Exception('Does not look like a global grid ?')
            if lonlat_env[0] > -180 or lonlat_env[1] < 180:
                logger.warning('[WARNING] Input do not cover [-180, 180] -> uncertain warp.')
            bbox_infos['bbox'] = [output_env[0], output_env[2], output_env[1], output_env[3]]
            warp_infos['extent'] = [output_env[0], output_env[2], output_env[1], output_env[3]]
        elif ngeoms == 2:
            # We should have here data covering less than 360° and crossing IDL.
            output_geom = shift_geoms[0]['output']
            output_geom = output_geom.Union(shift_geoms[1]['output'])
            output_geom = viewport_geom.Intersection(output_geom)
            output_geom = _clean_geometry(output_geom)
            if output_geom.GetGeometryType() != ogr.wkbGeometryCollection:
                raise Exception('Unexpected geometry.')
            over_geom = shift_geoms[0]['over']
            over_geom = over_geom.Union(shift_geoms[1]['over'])
            over_env = over_geom.GetEnvelope()
            warp_infos['extent'] = [over_env[0], over_env[2], over_env[1], over_env[3]]
            warp_infos['+over'] = True
            e_env = shift_geoms[0]['output'].GetEnvelope()
            w_env = shift_geoms[1]['output'].GetEnvelope()
            if w_env[1] < e_env[0]:
                # by convention we set bbox around 180° rather than -180°
                bbox_infos['bbox'] = [e_env[0], min([w_env[2], e_env[2]]),
                                      w_env[1] + 2 * viewport_env[1], max([w_env[3], e_env[3]])]
                bbox_infos['xIDL'] = True
                bbox_infos['w_bbox'] = [w_env[0], w_env[2], w_env[1], w_env[3]]
                bbox_infos['e_bbox'] = [e_env[0], e_env[2], e_env[1], e_env[3]]
                #bbox_infos['w_bbox'] = _get_bbox_wkt(w_env[0], w_env[2], w_env[1], w_env[3])
                #bbox_infos['e_bbox'] = _get_bbox_wkt(e_env[0], e_env[2], e_env[1], e_env[3])
            else:
                raise Exception('Unexpected west/east geometries.')
        else:
            raise Exception('Which case is that ?')

    # Cylindric -> Stereo
    elif input_proj in CYLINDRIC_PROJ and output_proj in STEREO_PROJ:
        output_geom = _project_geom(lonlat_geom, lonlat_srs, output_srs)
        if (lonlat_env[1] - lonlat_env[0]) >= 360:
            output_geom = output_geom.ConvexHull()
        if viewport_geom.Disjoint(output_geom):
            raise Exception('Input data is outside viewport.')
        output_geom = viewport_geom.Intersection(output_geom)
        output_geom = _clean_geometry(output_geom)
        if output_geom.GetGeometryType() != ogr.wkbPolygon:
            raise Exception('Unexpected geometry.')
        output_env = output_geom.GetEnvelope()
        bbox_infos['bbox'] = [output_env[0], output_env[2], output_env[1], output_env[3]]
        warp_infos['extent'] = [output_env[0], output_env[2], output_env[1], output_env[3]]
        # for data around dateline with input cylindric proj and output
        # stereographic proj, GDAL needs the config option CENTER_LONG for
        # warping correctly. Here we compute it if needed.
        if (lonlat_env[1] > 180 and lonlat_env[0] > -180) or \
           (lonlat_env[1] < 180 and lonlat_env[0] < -180):
            warp_infos['center_long'] = '{}'.format((lonlat_env[1] + lonlat_env[0]) / 2.)

    # Stereo -> Cylindric
    elif input_proj in STEREO_PROJ and output_proj in CYLINDRIC_PROJ:
        transformer = gdal.Transformer(dataset, None, ['MAX_GCP_ORDER=-1'])
        xsize, ysize = dataset.RasterXSize, dataset.RasterYSize
        xpo, ypo = transformer.TransformPoints(1, [[0, 0]])[0][0][0:2]
        has_pole = xpo >= 0 and xpo <= xsize and ypo >= 0 and ypo <= ysize
        if has_pole:
            # We modify lon/lat shape to include pole
            if input_proj in STEREO_PROJ_NORTH:
                pole = 89.9
            elif input_proj in STEREO_PROJ_SOUTH:
                pole = -89.9
            lonlat_shape = lonlat_shape[0:-1] # remove last (duplicated) point
            indmin = np.array(lonlat_shape)[:, 0].argmin()
            indmax = np.array(lonlat_shape)[:, 0].argmax()
            if indmax < indmin or (indmin == 0 and indmax == len(lonlat_shape)-1):
                lonlat_shape = lonlat_shape[::-1]
                indmin = np.array(lonlat_shape)[:, 0].argmin()
            if indmin != len(lonlat_shape)-1:
                for x in range(indmin+1):
                    lonlat_shape.append(lonlat_shape.pop(0))
            pt1, pt2 = lonlat_shape[-1], lonlat_shape[0]
            if pt1[0] != -180:
                lonlat_shape.append([-180., pt1[1]])
            lonlat_shape.extend([[-180., pole], [180., pole]])
            if pt2[0] != 180:
                lonlat_shape.append([180., pt2[1]])
            lonlat_shape.append(lonlat_shape[0])
            lonlat_geom = ogr.CreateGeometryFromWkt(_get_poly_wkt(lonlat_shape))
            lonlat_env = lonlat_geom.GetEnvelope()
        else:
            raise Exception('Input data does not contain geographical pole ?')
        output_geom = _project_geom(lonlat_geom, lonlat_srs, output_srs)
        if viewport_geom.Disjoint(output_geom):
            raise Exception('Input data is outside viewport.')
        output_geom = viewport_geom.Intersection(output_geom)
        output_geom = _clean_geometry(output_geom)
        if output_geom.GetGeometryType() != ogr.wkbPolygon:
            raise Exception('Unexpected geometry.')
        output_env = output_geom.GetEnvelope()
        bbox_infos['bbox'] = [output_env[0], output_env[2], output_env[1], output_env[3]]
        warp_infos['extent'] = [output_env[0], output_env[2], output_env[1], output_env[3]]

    # Stereo -> Stereo
    elif input_proj in STEREO_PROJ and output_proj in STEREO_PROJ:
        output_geom = _project_geom(lonlat_geom, lonlat_srs, output_srs)
        if viewport_geom.Disjoint(output_geom):
            raise Exception('Input data is outside viewport.')
        output_geom = viewport_geom.Intersection(output_geom)
        output_geom = _clean_geometry(output_geom)
        if output_geom.GetGeometryType() != ogr.wkbPolygon:
            raise Exception('Unexpected geometry.')
        output_env = output_geom.GetEnvelope()
        bbox_infos['bbox'] = [output_env[0], output_env[2], output_env[1], output_env[3]]
        warp_infos['extent'] = [output_env[0], output_env[2], output_env[1], output_env[3]]

    return output_geom, bbox_infos, warp_infos

def extract_band_info(band):
    """ """
    result = {}
    result['description'] = band.GetDescription()
    result['units'] = band.GetUnitType()
    metadata = band.GetMetadata_Dict()
    if 'parameter_range' in metadata:
        vmin, vmax = map(float, metadata['parameter_range'].split(' '))
        result['min'] = vmin
        result['max'] = vmax
    return result

def extract_from_dataset(dataset, cfg):
    """ """
    # Global metadata
    metadata = dataset.GetMetadata()
    dataset_name = metadata["name"][1:-1]
    product_name = metadata["product_name"][1:-1]
    datetime = metadata["datetime"][1:-1]
    date_modifiers = metadata['time_range']
    start, stop = get_range(datetime, date_modifiers.split(' '))
    nb_bands = dataset.RasterCount

    # Band metadata
    bands_info = [extract_band_info(dataset.GetRasterBand(ibnd)) \
                  for ibnd in range(1, nb_bands+1)]

    # No data values
    no_data_values = [dataset.GetRasterBand(ibnd).GetNoDataValue() \
                      for ibnd in range(1, nb_bands+1)]
    none_count = no_data_values.count(None)
    if none_count == 0:
        no_data_values = map(int, no_data_values)
    elif none_count == len(no_data_values):
        no_data_values = []
    else:
        raise Exception('All bands or none must have a nodatavalue.')

    # Color scheme
    is_rgb = False
    is_paletted = False
    if 3 == nb_bands:
        gci = [dataset.GetRasterBand(ibnd).GetColorInterpretation() \
               for ibnd in range(1, nb_bands+1)]
        gci_rgb = [gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand]
        if gci == gci_rgb:
            is_rgb = True
    elif 3 > nb_bands:
        is_paletted = dataset.GetRasterBand(1).GetColorTable() is not None

    # GCPs
    use_gcp = 0 < dataset.GetGCPCount()

    # Geometry
    ## Projections
    read_input_proj = cfg['input_options'].get('read_input_projection', None)
    if read_input_proj == 'True':
        srs = osr.SpatialReference()
        srs.ImportFromWkt(dataset.GetProjection())
        srs.AutoIdentifyEPSG()
        input_proj = int(srs.GetAuthorityCode(None))
        cfg['input_proj'] = input_proj
    input_proj = cfg['input_proj']
    output_proj = cfg['output_proj']
    if input_proj in CYLINDRIC_PROJ:
        cfg['input_proj_type'] = 'cylindric'
    elif input_proj in STEREO_PROJ:
        cfg['input_proj_type'] = 'stereographic'
    else:
        raise Exception('Is EPSG:{} cylindric or stereo ?'.format(input_proj))
    if output_proj in CYLINDRIC_PROJ:
        cfg['output_proj_type'] = 'cylindric'
    elif output_proj in STEREO_PROJ:
        cfg['output_proj_type'] = 'stereographic'
    else:
        raise Exception('Is EPSG:{} cylindric or stereo ?'.format(output_proj))
    ## Make a shape in lon/lat by traveling along dataset borders
    min_shape_res = int(cfg['output_options'].get('min-shape-res', MIN_SHAPE_RES))
    max_shape_points = int(cfg['output_options'].get('max-shape-points', MAX_SHAPE_POINTS))
    if max_shape_points > MAX_SHAPE_POINTS_LIMIT:
        raise Exception('max_shape_points exceeds the limit ({}).'.format(MAX_SHAPE_POINTS_LIMIT))
    transformer = gdal.Transformer(dataset, None, ['MAX_GCP_ORDER=-1'])
    lonlat_shape = _get_shape(dataset, input_proj, shape_proj=4326, dist_proj=4326,
                              ndist=10, transformer=transformer,
                              min_shape_res=min_shape_res, max_shape_points=max_shape_points)
    logger.debug('Shape longitude: {}'.format([x[0] for x in lonlat_shape]))
    logger.debug('Shape latitude: {}'.format([x[1] for x in lonlat_shape]))
    ## Check shape geometry if input cylindric
    if input_proj in CYLINDRIC_PROJ:
        lonlat_shape_geom = ogr.CreateGeometryFromWkt(_get_poly_wkt(lonlat_shape))
        if not lonlat_shape_geom.IsValid():
            iter_shape_max = int(cfg['output_options'].get('iter-shape-max', 0))
            if iter_shape_max == 0:
                raise Exception('lon/lat shape is not valid.')
            iter_shape = 0
            iter_shape_step = int(cfg['output_options'].get('iter-shape-step', 10))
            while iter_shape < iter_shape_max and not lonlat_shape_geom.IsValid():
                iter_shape += 1
                max_shape_points += iter_shape_step
                logger.debug('Shape iteration #{}: {}'.format(iter_shape, max_shape_points))
                if max_shape_points > MAX_SHAPE_POINTS_LIMIT:
                    raise Exception('max_shape_points exceeds the limit ({}).'.format(MAX_SHAPE_POINTS_LIMIT))
                lonlat_shape = _get_shape(dataset, input_proj,
                                          shape_proj=4326, dist_proj=4326,
                                          ndist=10, transformer=transformer,
                                          min_shape_res=min_shape_res,
                                          max_shape_points=max_shape_points)
                lonlat_shape_geom = ogr.CreateGeometryFromWkt(_get_poly_wkt(lonlat_shape))
            logger.debug('Shape longitude: {}'.format([x[0] for x in lonlat_shape]))
            logger.debug('Shape latitude: {}'.format([x[1] for x in lonlat_shape]))
            if not lonlat_shape_geom.IsValid():
                raise Exception('lon/lat shape is not valid.')
    ## Transform it to a shape geometry in output projection by taking care of:
    ## - input and output projections
    ## - output viewport
    ## We want here the shape to be displayed and to be used for spatial queries.
    ## By the way, we also get useful infos for bbox and warping.
    ## Some explanations:
    ## - output_shape_geom: surrounding geometry of data intersecting viewport.
    ## It should be contained inside viewport. Usually a polygon but it may be a
    ## geometry collection of multiple polygons if data go in and out the viewport
    ## (eg with output mercator and data crossing IDL or a piece of data exceeding
    ## latitude limits).
    ## - bbox_infos: bbox_infos['bbox'] is the bounding box of data intersecting
    ## viewport. Usually it is contained inside viewport except for output mercator
    ## and data crossing IDL. In that case, bbox is set by convention around 180°,
    ## bbox_infos['xIDL'] is set to True and bbox_infos['w_bbox']/bbox_infos['e_bbox']
    ## give west and east bboxes.
    ## - warp_infos: warp_infos['extent'] is a bounding box to be used for warping.
    ## This bbox is set accordingly to the georeferencing of input data so unlike
    ## bbox_infos['bbox'] it may be anywhere compared to viewport. Other keys
    ## in warp_infos give some options to be used with warping.
    viewport = cfg['viewport'].split(' ')
    viewport_geom = ogr.CreateGeometryFromWkt(_get_bbox_wkt(*viewport))
    output_shape_geom, bbox_infos, warp_infos = _get_output_shape(lonlat_shape,
                                                                  dataset,
                                                                  input_proj,
                                                                  output_proj,
                                                                  viewport_geom)
    logger.debug('Viewport WKT: {}'.format(viewport_geom.ExportToWkt()))
    logger.debug('Output shape WKT: {}'.format(output_shape_geom.ExportToWkt()))
    logger.debug('Output bbox infos: {}'.format(bbox_infos))
    logger.debug('Warp infos: {}'.format(warp_infos))
    real_shape_wkt = output_shape_geom.ExportToWkt()
    real_shape_wkt = real_shape_wkt.replace('POLYGON (', 'POLYGON(')
    if cfg['no_shape']:
        shape_wkt = 'POINT(0 0)'
    else:
        shape_wkt = real_shape_wkt

    src_meta = { 'dataset': dataset_name
               , 'product': product_name
               , 'begin_datetime': start
               , 'end_datetime': stop
               , 'nb_bands': nb_bands
               , 'bands_info': bands_info
               , 'nodatavalues': no_data_values
               , 'isrgb': is_rgb
               , 'ispaletted': is_paletted
               , 'use_gcp': use_gcp
               , 'lonlat_shape': lonlat_shape
               , 'real_shape_str': real_shape_wkt
               , 'shape_str': shape_wkt
               , 'bbox_infos': bbox_infos
               , 'warp_infos': warp_infos
               # , 'cropneeded': crop_needed
               # , 'cropextent': crop_extent
               # , 'center_long': center_long
               }

    if 'datagroup' in metadata:
        src_meta['datagroup'] = metadata['datagroup'][1:-1]

    return src_meta

def write_ini(path, cfg):
    """ """
    if 'bands_info' not in cfg or 0 >= len(cfg['bands_info']):
        return False

    config_parser = ConfigParser.ConfigParser()
    with open(path, 'w') as feature_file:
        config_parser.add_section('global')
        config_parser.set('global', 'feature_type', 'metadata')
        config_parser.add_section('metadata')
        for key in cfg['bands_info'][0]:
            value = cfg['bands_info'][0][key]
            while isinstance(value, basestring) and value.startswith('"'):
                value = value[1:]
            while isinstance(value, basestring) and value.endswith('"'):
                value = value[:-1]
            config_parser.set('metadata', key, value)
        config_parser.write(feature_file)
    return True
