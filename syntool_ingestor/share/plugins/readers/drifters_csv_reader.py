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
import math
import unicodecsv
from datetime import datetime, timedelta
from syntool_ingestor.interfaces import IReaderPlugin

class DriftersCSVReaderPlugin(IReaderPlugin):
    """ """

    def can_handle(self, source_format):
        """ """
        return 'drifters_csv' == source_format.lower()

    def extract_from_dataset(self, input_path, cfg):
        """ """
        # If nodatavalue is provided by the user, it must be cast to float
        # since all input_options are strings.
        no_data_value = cfg['input_options'].get('nodatavalue', 9999.0)
        no_data_value = float(no_data_value)

        field_names = cfg['input_options'].get('fields', [])
        if 0 < len(field_names):
            field_names = map(str.strip, field_names.split(','))
        data = self._parse_csv(input_path, field_names, no_data_value)
        for drifter_id, drifter_data in data.iteritems():
            if 0 >= len(drifter_data):
                # Skip drifters with no values
                # (they are in the CSV but all lines contain no data values
                # for the fields to extract)
                continue
            drifter_metadata = self._extract_metadata(drifter_id, drifter_data, cfg)
            yield (drifter_metadata, drifter_data)

    def _extract_metadata(self, drifter_id, drifter_data, cfg):
        """ """
        dataset_prefix = cfg['input_options'].get('dataset', 'GC_Buoy')
        metadata = { 'begin_datetime': drifter_data[0]['date']
                   , 'end_datetime': drifter_data[-1]['date']
                   , 'product': cfg['input_options'].get('product', 'GlobCurrent_drogue_15m')
                   , 'dataset': '{}-{}'.format(dataset_prefix, drifter_id)
                   }
        return metadata

    def _parse_csv(self, path, field_names, no_data_value):
        """ """
        # input
        file_name = os.path.basename(path)
        file_dir = os.path.dirname(path)

        buoys = {}
        julian_start_date = datetime(1950, 1, 1)
        date_fmt = '%Y-%m-%d %H:%M:%S'
        #to_extract = [field_name for field_name in field_names if field_name not in ['Jnces', 'LAT', 'LON']]
        to_extract = ['speed']
        with open(path, "rb") as csv_file:
            csv_reader = unicodecsv.DictReader( csv_file, skipinitialspace=True
                                              , delimiter=' '
                                              , fieldnames=field_names
                                              , quoting=unicodecsv.QUOTE_NONNUMERIC)
            for row in csv_reader:
                buoy_id = '{0:d}'.format(int(row["IDbuoy"]))
                lon = row['LON'] if row['LON'] < 180.0 else row['LON'] - 360.0
                lat = row['LAT']
                date = julian_start_date + timedelta(row['Jcnes'])
                date_str = date.strftime(date_fmt)
                if buoy_id not in buoys:
                    buoys[buoy_id] = []

                item = {'lon': lon, 'lat': lat, 'date': date_str, 'fields': {}}
                missing_data = False
                for field in to_extract:
                    u_value = row['U'+field]
                    v_value = row['V'+field]
                    if no_data_value in [u_value, v_value]:
                        # Skip row
                        missing_data = True
                        break;
                    value = math.sqrt(u_value ** 2 + v_value ** 2)
                    item['fields'][field] = value

                if not missing_data:
                    buoys[buoy_id].append(item)

        return buoys
