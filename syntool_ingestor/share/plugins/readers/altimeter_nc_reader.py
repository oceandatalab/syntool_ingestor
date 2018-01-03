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
from datetime import datetime, timedelta
from syntool_ingestor.interfaces import IReaderPlugin

logger = logging.getLogger(__name__)

class AltimeterNCReaderPlugin(IReaderPlugin):
    """ """

    def can_handle(self, source_format):
        """ """
        return 'altimeter_netcdf' == source_format.lower()

    def extract_from_dataset(self, input_path, cfg):
        """ """
        filename = os.path.basename(input_path)
        dataset_name = os.path.splitext(filename)[0]
        identifiers = ['Altimeter']
        dataset = netCDF4.Dataset(input_path, 'r')
        attributes = dataset.ncattrs()
        if 'mission_name' in attributes:
            identifiers.append(dataset.getncattr('mission_name').replace('/', '_'))
        if 'altimeter_sensor_name' in attributes:
            identifiers.append(dataset.getncattr('altimeter_sensor_name').replace('/', '_'))
        if 'product_suffix' in cfg['input_options']:
            identifiers.append(cfg['input_options']['product_suffix'])
        product_name = '_'.join(identifiers)

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

        # Get landmask
        valid_surface_types = cfg['input_options'].get('valid_surface_types', [])
        if 0 < len(valid_surface_types):
            valid_surface_types = map(lambda x: int(x.strip()), valid_surface_types)
        logger.debug('valid types: {}'.format(valid_surface_types))
        surface_types = dataset.variables['surface_type'][:].ravel()
        valid_indices = numpy.in1d(surface_types, valid_surface_types)
        if not numpy.any(valid_indices):
            raise Exception('No valid data found in this file (maybe check the "valid_surface_types" option?)')

        lats = dataset.variables['lat'][:].ravel()
        lons = dataset.variables['lon'][:].ravel()
        times = dataset.variables["time"][:].ravel()

        field_values = {}
        fields = cfg['input_options'].get('fields', [])
        if 0 < len(fields):
            fields = map(str.strip, fields.split(','))
        for field in fields:
            field_values[field] = dataset.variables[field][:].ravel()
            
        dataset.close()
            

        base_date = datetime.strptime("1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        time_offset = cfg['input_options'].get('time_offset', 0)
        time_offset = int(time_offset)
        base_date += timedelta(seconds=time_offset)

        date_fmt = '%Y-%m-%d %H:%M:%S'
        data = []
        min_time = sys.maxint
        max_time = 0
        skipped = False
        v_count = 0
        for i in xrange(0, len(valid_indices)):
            if valid_indices[i]:
                if valid_extent is not None:
                    p = ogr.Geometry(ogr.wkbPoint)
                    p.AddPoint_2D(lons[i], lats[i])
                    if not p.Within(valid_extent):
                        data.append({'lat': None, 'lon': None})
                        skipped = True
                        continue
                        


                t = int(times[i])
                d = base_date + timedelta(seconds=t)
                if min_time > t:
                    min_time = t
                if max_time < t:
                    max_time = t
                item = { 'date': datetime.strftime(d, date_fmt)
                       , 'lat': lats[i]
                       , 'lon': lons[i]
                       , 'fields': {}
                       }
                all_fields_ok = True
                for field in fields:
                    if numpy.ma.core.MaskedConstant == type(field_values[field][i]):
                        all_fields_ok = False
                        break
                    item['fields'][field] = field_values[field][i]
                if all_fields_ok:
                    data.append(item)
                    skipped = False
                    v_count = v_count + 1
            elif not skipped:
                data.append({'lat': None, 'lon': None})
                skipped = True
        
        if 0 >= v_count:
            raise Exception('No valid data found in this file (maybe check the "valid_extent" option?)')

        start_date = base_date + timedelta(seconds=min_time)
        start_datetime = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_date = base_date + timedelta(seconds=max_time)
        end_datetime = end_date.strftime("%Y-%m-%d %H:%M:%S")

        metadata = { 'begin_datetime': start_datetime
                   , 'end_datetime': end_datetime
                   , 'product': product_name
                   , 'dataset': dataset_name
                   , 'time_offset': time_offset
                   }
        yield(metadata, data)

