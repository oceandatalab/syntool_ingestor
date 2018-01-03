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
import netCDF4
from datetime import datetime, timedelta
from syntool_ingestor.interfaces import IReaderPlugin

class GlobwaveNCReaderPlugin(IReaderPlugin):
    """ """

    def can_handle(self, source_format):
        """ """
        return 'globwave_netcdf' == source_format.lower()

    def extract_from_dataset(self, input_path, cfg):
        """ """
        filename = os.path.basename(input_path)
        dataset_name_base = os.path.splitext(filename)[0]
        product_name = cfg['input_options'].get('product', 'Globwave_spectral_density')
        dataset = netCDF4.Dataset(input_path, 'r')
        times = dataset.variables["time"][:].flatten()
        dataset.close()

        base_date = datetime.strptime("1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        time_offset = cfg['input_options'].get('time_offset', 347155200) # 1981/01/01
        time_offset = int(time_offset)
        base_date += timedelta(seconds=time_offset)

        l = len(times)
        for i in xrange(0, l):
            time = int(times[i])
            if 0 == i:
                min_time = (3 * time - int(times[1])) / 2
            else:
                min_time = (time + int(times[i-1])) / 2
            if i == len(times) - 1:
                max_time = (3 * time - int(times[-2])) / 2
            else:
                max_time = (time + int(times[i+1])) / 2
            start_date = base_date + timedelta(seconds=min_time)
            start_datetime = start_date.strftime("%Y-%m-%d %H:%M:%S")
            end_date = base_date + timedelta(seconds=max_time)
            end_datetime = end_date.strftime("%Y-%m-%d %H:%M:%S")

            dataset_name = '{}_{}'.format(dataset_name_base, i)

            metadata = { 'begin_datetime': start_datetime
                       , 'end_datetime': end_datetime
                       , 'product': product_name
                       , 'dataset': dataset_name
                       , 'time_index': i
                       , 'time_offset': time_offset
                       }
            yield(metadata, None)

