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

import re
import unicodecsv
from datetime import datetime
from syntool_ingestor.interfaces import IReaderPlugin


class DriftersANDROReaderPlugin(IReaderPlugin):
    """ """

    def can_handle(self, source_format):
        """ """
        return 'argo_deep' == source_format.lower()

    def extract_from_dataset(self, input_path, cfg):
        """ """
        # If nodatavalue is provided by the user, it must be cast to float
        # since all input_options are strings.
        no_data_value = cfg['input_options'].get('nodatavalue', -99999.0)
        no_data_value = float(no_data_value)

        # Use a pattern to filter which buoy is included in the product
        pattern = cfg['input_options'].get('id_pattern', r'.*')

        # Define the minimal time delta between two points
        # (in seconds, defaults to 60 minutes)
        # min_timedelta = cfg['input_options'].get('min_timedelta', 3600)
        # min_timedelta = int(min_timedelta)

        field_names = cfg['input_options'].get('fields', [])
        if 0 < len(field_names):
            field_names = map(str.strip, field_names.split(','))
        data = self._parse_csv(input_path, field_names, no_data_value)
        for drifter_id, drifter_data in data.iteritems():
            if re.match(pattern, drifter_id) is None:
                # Drifter identifier does not match pattern
                continue
            if 0 >= len(drifter_data):
                # Skip drifters with no values
                # (they are in the CSV but all lines contain no data values
                # for the fields to extract)
                continue
            drifter_metadata = self._extract_metadata(drifter_id,
                                                      drifter_data, cfg)
            yield (drifter_metadata, drifter_data)

    def _extract_metadata(self, drifter_id, drifter_data, cfg):
        """ """
        dataset_prefix = cfg['input_options'].get('dataset', 'ARGO_Deep_NATL'
                                                  + drifter_data[0]['fields']
                                                  ['depth'])
        metadata = {'begin_datetime': drifter_data[0]['date'],
                    'end_datetime': drifter_data[-1]['date'],
                    'product': cfg['input_options'].get('product',
                                                        'ARGO_Deep_NATL'
                                                        + drifter_data[0]
                                                        ['fields']['depth']),
                    'dataset': '{}-{}'.format(dataset_prefix, drifter_id),
                    }
        return metadata

    def _parse_csv(self, path, field_names, no_data_value):
        """ """
        # input
        buoys = {}
        date_fmt = '%Y-%m-%d %H:%M:%S'
        with open(path, "rb") as csv_file:
            csv_reader = unicodecsv.DictReader(csv_file, skipinitialspace=True,
                                               delimiter=',',
                                               fieldnames=field_names
                                               )
            for row in csv_reader:
                full_id = '{0:d}'.format(int(row["IDbuoy"]))
                depth = row["DEPTH"]
                if full_id not in buoys:
                    buoys[full_id] = []
                lon = float(row['LON'])
                lon = lon if lon < 180.0 else lon - 360.0
                if abs(lon) > 180:
                    continue
                lon = "{:.5f}".format(lon)
                lat = row['LAT']
                if abs(float(lat)) > 90:
                    print(lon, lat)
                    continue
                time = datetime.strptime(row['TIME'],  date_fmt)
                time = row['TIME']
                item = {'lon': lon, 'lat': lat, 'date': time, 'fields': {}}
                item['fields']['speed'] = row["SPEED"]
                item['fields']['temp'] = row["TEMP"]
                item['fields']['depth'] = depth
                # if not depth: continue
                # else: item['fields']['depth'] = depth
                buoys[full_id].append(item)
        return buoys
