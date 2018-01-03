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
import osr
import ogr
import math
import json
import logging
from syntool_ingestor.interfaces import IFormatterPlugin
from syntool_ingestor.utils import NumpyAwareJSONEncoder
import syntool_ingestor.metadata

logger = logging.getLogger(__name__)

class GeoJSONTrajectoryPlugin(IFormatterPlugin):
    """ """

    def can_handle(self, target_format):
        """ """
        return 'geojson_trajectory' == target_format

    def get_output_id(self, cfg):
        """ """
        return ''

    def get_representation_type(self, cfg):
        """ """
        return 'TRAJECTORY'

    def create_representation(self, input_path, input_data, workspace,
                              cfg, src_meta):
        """ """
        if input_data is None:
            msg = 'GeoJSON trajectories plugin must receive pre-parsed data,'\
                  'not a file path.'
            raise Exception(msg)

        product_id = src_meta['syntool_id']
        dataset_id = src_meta['dataset']
        epsg = cfg['output_proj']
        options = cfg['output_options']
        start = src_meta['begin_datetime']
        stop = src_meta['end_datetime']

        viewport = map(float, cfg['viewport'].split(' '))
        srs4326 = osr.SpatialReference()
        srs4326.ImportFromEPSG(4326)
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(epsg)
        transform = osr.CoordinateTransformation(srs4326, target_srs)

        ranges = {}
        _provided_ranges = options.get('ranges', [])
        if 0 < len(_provided_ranges):
            _provided_ranges = map(str.strip, _provided_ranges.split(','))
        for field in _provided_ranges:
            try:
                field_name, field_min, field_max = field.split(':')
                ranges[field_name] = [float(field_min), float(field_max)]
            except ValueError:
                msg = '"{}": Invalid format for "ranges" output option: {} ' \
                      '(should be "name:min:max")'.format(dataset_id, field)
                raise Exception(msg)

        _single_feature = options.get('singleFeature', False)
        if _single_feature is not False:
            create_feature = self._create_geojson_single_feature
        else:
            create_feature = self._create_geojson_multi_feature

        try:
            geojson, shape, bbox, w_bbox, e_bbox = create_feature(input_data,
                                                                  viewport,
                                                                  transform,
                                                                  start, stop)
        except Exception, e:
            raise Exception('"{}": {}'.format(dataset_id, str(e)))

        geojson['properties']['productId'] = product_id
        geojson['properties']['datasetId'] = dataset_id
        geojson['properties']['fields'] = ranges
        geojson_path = os.path.join(workspace, 'geojson.json')
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f, cls=NumpyAwareJSONEncoder,indent=4,
                      sort_keys=True)

        metadata = self._create_metadata(geojson, geojson_path, shape, bbox,
                                         w_bbox, e_bbox)
        return metadata


    def _create_metadata(self, geojson, output_path, shape, bbox, w_bbox,
                         e_bbox):
        """ """
        metadata = { 'begin_datetime': geojson['properties']['timeRange'][0]
                   , 'end_datetime': geojson['properties']['timeRange'][1]
                   , 'min_zoom_level': 0
                   , 'max_zoom_level': 0
                   , 'resolutions': []
                   , 'bbox_str': bbox
                   , 'shape_str': shape
                   , 'output_path': os.path.abspath(output_path)
                   }
        if w_bbox is not None and e_bbox is not None:
            metadata['w_bbox'] = w_bbox
            metadata['e_bbox'] = e_bbox
        return metadata

    def handle_dateline(self, x1, y1, x2, y2, tlbr):
        """ """
        extent = tlbr[3] - tlbr[1]
        # If one point is in the upper quarter of the map 
        # and the other is in the lower one, consider that the dateline is
        # crossed
        if 0 > x1 * x2 and abs(x1) > 0.25 * extent and abs(x2) > 0.25 * extent:
            if 0 < x1:
                p = (y2 - y1) / (x2 - (x1 - extent))
                yi = y1 + (tlbr[3] - x1) * p
                return [tlbr[3] - 1 , yi], [tlbr[1] + 1, yi]
            else:
                p = (y2 - y1) / (x2 - (x1 + extent))
                yi = y1 + (tlbr[1] - x1) * p
                return [tlbr[1] + 1, yi], [tlbr[3] - 1, yi]
        return None, None

    def _bbox_shape(self, xidl, tlbr, western_area, eastern_area):
        """"""
        shape_wkt = None
        bbox_wkt = None
        w_bbox = None
        e_bbox = None

        if xidl:
            min_lon = tlbr[1]
            max_lon = max([p[0] for p in western_area])
            min_lat = min([p[1] for p in western_area])
            max_lat = max([p[1] for p in western_area])
            w_bbox = 'POLYGON(({} {},{} {},{} {},{} {},{} {}))'.format(
                                                        min_lon, max_lat,
                                                        max_lon, max_lat,
                                                        max_lon, min_lat,
                                                        min_lon, min_lat,
                                                        min_lon, max_lat)

            min_lon = min([p[0] for p in eastern_area])
            max_lon = tlbr[3]
            min_lat = min([p[1] for p in eastern_area])
            max_lat = max([p[1] for p in eastern_area])
            e_bbox = 'POLYGON(({} {},{} {},{} {},{} {},{} {}))'.format(
                                                        min_lon, max_lat,
                                                        max_lon, max_lat,
                                                        max_lon, min_lat,
                                                        min_lon, min_lat,
                                                        min_lon, max_lat)
            shape_wkt = 'GEOMETRYCOLLECTION ({},{})'.format(w_bbox, e_bbox)
            shape_geom = ogr.CreateGeometryFromWkt(shape_wkt)
            min_lon, max_lon, min_lat, max_lat = shape_geom.GetEnvelope()
            bbox_wkt = 'POLYGON(({} {},{} {},{} {},{} {},{} {}))'.format(
                                                        min_lon, max_lat,
                                                        max_lon, max_lat,
                                                        max_lon, min_lat,
                                                        min_lon, min_lat,
                                                        min_lon, max_lat)
        else:
            western_area.extend(eastern_area)
            min_lon = min([p[0] for p in western_area])
            max_lon = max([p[0] for p in western_area])
            min_lat = min([p[1] for p in western_area])
            max_lat = max([p[1] for p in western_area])
            shape_wkt = 'POLYGON(({} {},{} {},{} {},{} {},{} {}))'.format(
                                                        min_lon, max_lat,
                                                        max_lon, max_lat,
                                                        max_lon, min_lat,
                                                        min_lon, min_lat,
                                                        min_lon, max_lat)
            bbox_wkt = shape_wkt

        return shape_wkt, bbox_wkt, w_bbox, e_bbox


    def _create_geojson_single_feature(self, drift_data, tlbr, ct, begin_date,
                                       end_date):
        """ """
        total_extent = tlbr[3] - tlbr[1]

        western_area = []
        eastern_area = []
        features = []
        last_x = None
        last_y = None
        l = len(drift_data)

        idl_crossed = False
        props = {}
        lines = []
        line = ogr.Geometry(ogr.wkbLineString)
        for i, coords in enumerate(drift_data):
            lon = coords["lon"]
            lat = coords["lat"]

            for coord_name, coord_value in coords.iteritems():
                if coord_name not in props:
                    props[coord_name] = [coord_value]
                else:
                    props[coord_name].append(coord_value)

            # None in lat or lon means that the trajectory has been cut (land?)
            if None in [lat, lon]:
                last_x = None
                last_y = None
                lines.append(line)
                line = ogr.Geometry(ogr.wkbLineString)
                continue

            proj_coords = ct.TransformPoint(float(lon), float(lat), 0.0)
            x = proj_coords[0]
            y = proj_coords[1]

            if last_x is not None and last_y is not None:
                # Handle cross dateline case
                p1, p2 = self.handle_dateline(last_x, last_y, x, y, tlbr)
                if None not in [p1, p2]:
                    idl_crossed = True
                    line.AddPoint_2D(p1[0], p1[1])
                    if p1[0] >= 0:
                        eastern_area.append(p1)
                    else:
                        western_area.append(p1)
                    lines.append(line)
                    line = ogr.Geometry(ogr.wkbLineString)
                    line.AddPoint_2D(p2[0], p2[1])
                    if p2[0] >= 0:
                        eastern_area.append(p2)
                    else:
                        western_area.append(p2)

            line.AddPoint_2D(x, y)
            if x >= 0:
                eastern_area.append([x,y])
            else:
                western_area.append([x,y])

        if 0 < len(lines):
            geometry = ogr.Geometry(ogr.wkbMultiLineString)
            for _line in lines:
                geometry.AddGeometry(_line)
            geometry.AddGeometry(line)
        else:
            geometry = line

        feature = { "type": "Feature"
                  , "properties": props
                  , "geometry": json.loads(geometry.ExportToJson())
                  }
        features.append(feature)

        shape_wkt, bbox_wkt, w_bbox, e_bbox = self._bbox_shape(idl_crossed,
                                                               tlbr,
                                                               western_area,
                                                               eastern_area)

        geojson = { "type": "FeatureCollection"
                  , "features": features
                  , "crs": { "type": "name"
                           , "properties": { "name": "urn:ogc:def:crs:EPSG::3857" }
                           }
                  , "properties": {"timeRange": [begin_date, end_date]}
                  }

        return geojson, shape_wkt, bbox_wkt, e_bbox, w_bbox

    def _create_geojson_multi_feature(self, drift_data, tlbr, ct, begin_date, end_date):
        """ """
        total_extent = tlbr[3] - tlbr[1]

        western_area = []
        eastern_area = []
        geom_col =  ogr.Geometry(ogr.wkbGeometryCollection)
        idl_crossed = False
        features = []
        last_x = None
        last_y = None
        l = len(drift_data)
        for i, coords in enumerate(drift_data):
            lon = coords["lon"]
            lat = coords["lat"]

            # None in lat or lon means that the trajectory has been cut (land?)
            if None in [lat, lon]:
                last_x = None
                last_y = None
                continue

            proj_coords = ct.TransformPoint(float(lon), float(lat), 0.0)
            x = proj_coords[0]
            y = proj_coords[1]

            lines = []
            line = ogr.Geometry(ogr.wkbLineString)

            if last_x is not None and last_y is not None:
                line.AddPoint_2D(last_x, last_y)
                if last_x >= 0:
                    eastern_area.append([last_x,last_y])
                else:
                    western_area.append([last_x,last_y])
                # Handle cross dateline case
                p1, p2 = self.handle_dateline(last_x, last_y, x, y, tlbr)
                if None not in [p1, p2]:
                    idl_crossed = True
                    line.AddPoint_2D(p1[0], p1[1])
                    if p1[0] >= 0:
                        eastern_area.append(p1)
                    else:
                        western_area.append(p1)
                    lines.append(line)
                    line = ogr.Geometry(ogr.wkbLineString)
                    line.AddPoint_2D(p2[0], p2[1])
                    if p2[0] >= 0:
                        eastern_area.append(p2)
                    else:
                        western_area.append(p2)


            line.AddPoint_2D(x, y)
            if x >= 0:
                eastern_area.append([x,y])
            else:
                western_area.append([x,y])

            if i + 1 < l:
                nlon = drift_data[i+1]['lon']
                nlat = drift_data[i+1]['lat']
                if None not in [nlon, nlat]:
                    # Compute midpoint to next coords
                    next_segment = ogr.Geometry(ogr.wkbLineString)
                    next_segment.AddPoint_2D(x,y)
                    proj_ncoords = ct.TransformPoint(float(nlon), float(nlat), 0.0)
                    next_segment.AddPoint_2D(proj_ncoords[0],proj_ncoords[1])
                    midpoint = next_segment.Centroid()
                    last_x = midpoint.GetX()
                    last_y = midpoint.GetY()

                    # If the distance between current and next position exceeds half of the globe
                    # (ie. distance between current and midpoint > 0.25 * extent)
                    # then consider that the dateline has been crossed and adapt midpoint coordinates
                    if abs(x - last_x) > 0.25 * total_extent:
                        last_x = 0.5 * total_extent - last_x

                    p1, p2 = self.handle_dateline(x, y, last_x, last_y, tlbr)
                    if None not in [p1, p2]:
                        idl_crossed = True
                        line.AddPoint_2D(p1[0], p1[1])
                        if p1[0] >= 0:
                            eastern_area.append(p1)
                        else:
                            western_area.append(p1)
                        lines.append(line)
                        line = ogr.Geometry(ogr.wkbLineString)
                        line.AddPoint_2D(p2[0], p2[1])
                        if p2[0] >= 0:
                            eastern_area.append(p2)
                        else:
                            western_area.append(p2)
                    line.AddPoint_2D(last_x, last_y)
                    if last_x >= 0:
                        eastern_area.append([last_x,last_y])
                    else:
                        western_area.append([last_x,last_y])

            if 0 < len(lines):
                geometry = ogr.Geometry(ogr.wkbMultiLineString)
                for _line in lines:
                    geometry.AddGeometry(_line)
                geometry.AddGeometry(line)
            else:
                geometry = line

            logger.debug('Add geometry to collection')
            geom_col.AddGeometry(geometry)
            logger.debug('collection: {}'.format(geom_col.ExportToWkt()))
            feature = { "type": "Feature"
                      , "properties": coords
                      , "geometry": json.loads(geometry.ExportToJson())
                      }
            features.append(feature)

        shape_wkt, bbox_wkt, w_bbox, e_bbox = self._bbox_shape(idl_crossed,
                                                               tlbr,
                                                               western_area,
                                                               eastern_area)

        geojson = { "type": "FeatureCollection"
                  , "features": features
                  , "crs": { "type": "name"
                           , "properties": { "name": "urn:ogc:def:crs:EPSG::3857" }
                           }
                  , "properties": {"timeRange": [begin_date, end_date]}
                  }

        return geojson, shape_wkt, bbox_wkt, w_bbox, e_bbox
