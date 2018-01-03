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
import re
import pyproj
import logging
import datetime
import unicodecsv
from syntool_ingestor.interfaces import IReaderPlugin

logger = logging.getLogger(__name__)

def pos_to_decimal(tude):
    """ """
    if 0 >= len(tude):
        raise ValueError
    multiplier = 1 if tude[-1].upper() in ['N', 'E'] else -1
    split_tude = filter(lambda x: 0 < len(x), re.split(r'[^0-9.]', tude[:-1]))
    return multiplier * sum(float(x) / 60 ** n for n, x in enumerate(split_tude))

class ShiptrackCSVReaderPlugin(IReaderPlugin):
    """ """

    def can_handle(self, source_format):
        """ """
        return source_format.lower() == 'shiptrack_csv'

    def extract_from_dataset(self, input_path, cfg):
        """ """
        field_names = cfg['input_options'].get('fields', [])
        if 0 < len(field_names):
            field_names = map(str.strip, field_names.split(','))
        data = self._parse_csv(input_path, field_names)
        metadata = self._extract_metadata(data, input_path, cfg)
        yield(metadata, data)

    def _extract_metadata(self, data, input_path, cfg):
        """ """
        dataset_prefix = cfg['input_options'].get('dataset', 'GC_Ship')
        filename = os.path.basename(input_path)
        name_base = os.path.splitext(filename)[0]
        metadata = { 'begin_datetime': data[0]['date']
                   , 'end_datetime': data[-1]['date']
                   , 'product': cfg['input_options'].get('product', 'GlobCurrent_ship_track')
                   , 'dataset': '{}-{}'.format(dataset_prefix, name_base)
                   }
        return metadata

    def _parse_csv(self, input_path, field_names):
        """ """
        data = []
        with open(input_path, 'rb') as f:
            reader = unicodecsv.DictReader( [row for row in f if not row.startswith('#')]
                                          , skipinitialspace=True
                                          , delimiter=' '
                                          , fieldnames=field_names)
            to_extract = [field_name for field_name in field_names if field_name not in ['time', 'lat', 'lon']]
            geod = pyproj.Geod(ellps='WGS84')
            last_lon = None
            last_lat = None
            last_time = None
            for row in reader:
                # Each row must at least contain Date, Lat, Lon
                if 3 > len(row):
                    logger.warn('Skipped line "{}": not enough entries'.format(str(row)))
                    continue

                try:
                    d = row['time'].replace('T', ' ').replace('Z', '')
                    _time = datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
                    lat = pos_to_decimal(row['lat'])
                    lon = pos_to_decimal(row['lon'])
                    lon = lon if lon < 180.0 else lon - 360.0
                    speed = 0.0
                    if None not in [last_lat, last_lon, last_time]:
                        _, _, dist = geod.inv(last_lon, last_lat, lon, lat)
                        dt = (_time - last_time).total_seconds()
                        if dt > 0.0:
                            speed = dist / dt
                    last_lat = lat
                    last_lon = lon
                    last_time = _time
                    item = {'lon': lon, 'lat': lat, 'date': d, 'fields': {'speed': speed}}
                    for field in to_extract:
                        item['fields'][field] = row[field]
                    data.append(item) 
                    logger.debug('{}'.format(item))
                except ValueError:
                    logger.warn('Skipped line "{}": could not parse numerical values'.format(str(row)))
                    continue
        return data
