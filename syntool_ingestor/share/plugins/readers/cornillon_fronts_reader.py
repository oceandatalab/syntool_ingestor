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
import ogr
import netCDF4
import numpy
import logging
from datetime import datetime
from syntool_ingestor.interfaces import IReaderPlugin

logger = logging.getLogger(__name__)

class CornillonFrontsReaderPlugin(IReaderPlugin):
    """ """

    def can_handle(self, source_format):
        """ """
        return 'cornillon_netcdf' == source_format.lower()

    def extract_from_dataset(self, input_path, cfg):
        """ """
        if 'product_name' not in cfg['input_options']:
            raise Exception('You must specify a product name with --input-options product_name="XXXXX")')

        if 'dataset_duration' not in cfg['input_options']:
            raise Exception('You must specify a duration (in seconds) for the dataset with --input-options dataset_duration=XXXXX')

        date_fmt = '%Y-%m-%d %H:%M:%S'

        filename = os.path.basename(input_path)
        dataset_name = os.path.splitext(filename)[0]
        if 'dataset_suffix' in cfg['input_options']:
            dataset_name = '{}{}'.format(dataset_name, cfg['input_options']['dataset_suffix'])
        product_name = cfg['input_options'].get('product_name', None)
        dataset_duration = int(cfg['input_options']['dataset_duration'])
        dataset = netCDF4.Dataset(input_path, 'r')

        # Valid extent
        valid_extent = cfg['input_options'].get('valid_extent', None)
        if valid_extent is not None:
            valid_extent = map(str.strip, valid_extent.split(','))
            valid_extent = map(float, valid_extent)
            if 4 < len(valid_extent):
                logger.warn('valid_extent ignored because it contains less than 4 values.') 
                valid_extent = None
            else:
                geometry = ogr.Geometry(ogr.wkbLinearRing)
                geometry.AddPoint(valid_extent[0], valid_extent[1])
                geometry.AddPoint(valid_extent[2], valid_extent[1])
                geometry.AddPoint(valid_extent[2], valid_extent[3])
                geometry.AddPoint(valid_extent[0], valid_extent[3])
                geometry.AddPoint(valid_extent[0], valid_extent[1])
                polygon = ogr.Geometry(ogr.wkbPolygon)
                polygon.AddGeometry(geometry)
                valid_extent = polygon

        timestamp = dataset.variables['DateTime'][0]
        data = []
        n = len(dataset.dimensions['segment'])
        for i in xrange(0,n):
            seg_length = dataset.variables['segment_length'][i]
            if 0 >= seg_length:
                # no data
                continue

            # -1 because it seems segment_start refers to a 1-indexed array
            seg_start = dataset.variables['segment_start'][i] - 1

            lats = dataset.variables['latitude'][seg_start:seg_start+seg_length]
            lons = dataset.variables['longitude'][seg_start:seg_start+seg_length]
            sst_diff = dataset.variables['sst_difference'][seg_start:seg_start+seg_length]

            values = zip(lats, lons, sst_diff)
            values = filter(lambda x: None not in x and numpy.ma.masked not in x, values)

            if valid_extent is not None:
                p = ogr.Geometry(ogr.wkbPoint)
                p.AddPoint_2D(float(values[0][1]), float(values[0][0]))
                if not p.Within(valid_extent):
                    # Arbitrary policy: fronts belong to the BBOX that contain
                    # their first point.
                    # First point is not in the BBOX => skip the front entirely
                    continue

            if 0 >= len(values):
                # All values have been filtered, nothing left
                continue
            logger.error(values)
            segment_data = map(lambda x: {'lat': x[0], 'lon': x[1], 'fields': {'sst_difference': abs(x[2])}}, values)
            data.extend(segment_data)

            # Mark the end of the front
            data.append({'lat': None, 'lon': None})
        dataset.close()
        
        start = datetime.fromtimestamp(timestamp - 0.5 * dataset_duration)
        stop = datetime.fromtimestamp(timestamp + 0.5 * dataset_duration)
        metadata = { 'begin_datetime': start.strftime(date_fmt)
                   , 'end_datetime': stop.strftime(date_fmt)
                   , 'product': product_name
                   , 'dataset': dataset_name
                   }
        yield(metadata, data)
