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


class DriftersLaserReaderPlugin(IReaderPlugin):
    """ """

    def can_handle(self, source_format):
        """ """
        return 'drifters_laser' == source_format.lower()

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
        min_timedelta = cfg['input_options'].get('min_timedelta', 3600)
        min_timedelta = int(min_timedelta)

        field_names = cfg['input_options'].get('fields', [])
        if 0 < len(field_names):
            field_names = map(str.strip, field_names.split(','))
        data = self._parse_csv(input_path, field_names, no_data_value,
                               min_timedelta)
        for drifter_id, drifter_data in data.iteritems():
            if re.match(pattern, drifter_id) is None:
                # Drifter identifier does not match pattern
                continue
            if 0 >= len(drifter_data):
                # Skip drifters with no values
                # (they are in the CSV but all lines contain no data values
                # for the fields to extract)
                continue
            drifter_metadata = self._extract_metadata(drifter_id, drifter_data,
                                                      cfg)
            yield (drifter_metadata, drifter_data)

    def _extract_metadata(self, drifter_id, drifter_data, cfg):
        """ """
        dataset_prefix = cfg['input_options'].get('dataset', 'LASER_Drifter')
        metadata = {'begin_datetime': drifter_data[0]['date'],
                    'end_datetime': drifter_data[-1]['date'],
                    'product': cfg['input_options'].get('product',
                                                        'LASER_drifters'),
                    'dataset': '{}-{}'.format(dataset_prefix, drifter_id)
                    }
        return metadata

    def _parse_csv(self, path, field_names, no_data_value, min_timedelta):
        """ """
        # input
        buoys = {}
        date_fmt = '%Y-%m-%d %H:%M:%S'
        with open(path, "rb") as csv_file:
            csv_reader = unicodecsv.DictReader(csv_file, skipinitialspace=True,
                                               delimiter=',',
                                               fieldnames=field_names)
            last_time = None
            for row in csv_reader:
                buoy_id = row["esnName"]
                lon = float(row['longitude']) if (float(row['longitude']) < 180.0) else float(row['longitude']) - 360.0
                lat = float(row['latitude'])
                if (1 > abs(lon - no_data_value)
                   or 1 > abs(lat - no_data_value)):
                    continue
                _time = datetime.strptime(row['date'], date_fmt)
                date_str = _time.strftime(date_fmt)
                full_id = '{}'.format(buoy_id)
                if full_id not in buoys:
                    buoys[full_id] = []
                    last_time = None
                    dt_total = 0

                dt = 0
                item = {'lon': lon, 'lat': lat, 'date': date_str, 'fields': {}}
                if None not in [last_time]:
                    dt = (_time - last_time).total_seconds()
                item['fields']['speed'] = float(row['speed'])
                dt_total = dt+dt_total
                if (0 >= len(buoys[full_id])
                   or (0 <= dt and dt <= min_timedelta)):
                    if dt_total >= min_timedelta or dt == 0:
                        buoys[full_id].append(item)
                        dt_total = 0
                last_time = datetime.strptime(row['date'], date_fmt)
            buoys[full_id].append(item)
        return buoys
