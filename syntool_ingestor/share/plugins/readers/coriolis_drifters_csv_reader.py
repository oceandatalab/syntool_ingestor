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

class CoriolisDriftersCSVReaderPlugin(IReaderPlugin):
    """ """

    def can_handle(self, source_format):
        """ """
        return 'coriolis_drifters_csv' == source_format.lower()

    def extract_from_dataset(self, input_path, cfg):
        """ """
        # If nodatavalue is provided by the user, it must be cast to float
        # since all input_options are strings.
        no_data_value = cfg['input_options'].get('nodatavalue', 9999.0)
        no_data_value = float(no_data_value)

        """
        default_fields = '''PLATFORM, DATE, LATITUDE, LONGITUDE, 
                            WSPE_MODEL LEVEL0, WSPN_MODEL LEVEL0,
                            WSTE_MODEL LEVEL0, WSTN_MODEL LEVEL0,
                            TEMP LEVEL1, CURRENT_TEST LEVEL2, DEPH LEVEL2,
                            EWCT LEVEL2, NSCT LEVEL2, QC'''
        field_names = cfg['input_options'].get('fields', default_fields)
        if 0 < len(field_names):
            field_names = map(str.strip, field_names.split(','))
        """
        field_names = []
        data = self._parse_csv(input_path, field_names, no_data_value)
        for drifter_id, drifter_data in data.iteritems():
            drifter_metadata = self._extract_metadata(drifter_id, drifter_data, cfg)
            yield (drifter_metadata, drifter_data)

    def _extract_metadata(self, drifter_id, drifter_data, cfg):
        """ """
        dataset_prefix = cfg['input_options'].get('dataset', 'Coriolis_Buoy')
        metadata = { 'begin_datetime': drifter_data[0]['date']
                   , 'end_datetime': drifter_data[-1]['date']
                   , 'product': cfg['input_options'].get('product', 'Coriolis_drifters_15m')
                   , 'dataset': '{}-{}'.format(dataset_prefix, drifter_id)
                   }
        return metadata

    def _parse_csv(self, path, field_names, no_data_value):
        """ """
        # input
        file_name = os.path.basename(path)
        file_dir = os.path.dirname(path)

        buoys = {}
        date_fmt = '%Y-%m-%d %H:%M:%S'
        date_input_fmt = '%Y-%m-%dT%H:%M:%SZ'
        with open(path, "rb") as csv_file:
            header = csv_file.readline()
            field_names = [field[:field.find('(')] if -1 < field.find('(') else field for field in header.split(',') ]
            field_names = map(str.strip, field_names)
            csv_reader = unicodecsv.DictReader( csv_file, skipinitialspace=True
                                              , delimiter=','
                                              , fieldnames=field_names)
            for row in csv_reader:
                buoy_id = '{0:d}'.format(int(row["PLATFORM"]))
                lon = float(row['LONGITUDE'])
                lon = lon if lon < 180.0 else lon - 360.0
                lat = float(row['LATITUDE'])
                d = datetime.strptime(row['DATE'], date_input_fmt)
                date_str = d.strftime(date_fmt)
                if buoy_id not in buoys:
                    buoys[buoy_id] = []

                item = {'lon': lon, 'lat': lat, 'date': date_str, 'fields': {}}
                missing_data = False
                u_value = float(row['EWCT LEVEL2'])
                v_value = float(row['NSCT LEVEL2'])
                if no_data_value in [u_value, v_value]:
                    # Skip row
                    missing_data = True
                    break;
                value = math.sqrt(u_value ** 2 + v_value ** 2)
                item['fields']['buoy'] = value

                if not missing_data:
                    buoys[buoy_id].append(item)

        return buoys
