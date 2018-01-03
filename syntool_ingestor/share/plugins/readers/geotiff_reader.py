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

import contextlib
from osgeo import gdal
import syntool_ingestor.metadata
from syntool_ingestor.interfaces import IReaderPlugin

@contextlib.contextmanager
def load_dataset(geotiff_path):
    """ """
    try:
        geotiff_file = gdal.Open(geotiff_path)
        yield geotiff_file
    finally:
        geotiff_file = None
        del geotiff_file

class GeoTIFFReaderPlugin(IReaderPlugin):
    """ """

    def can_handle(self, source_format):
        """ """
        return source_format.lower() == 'geotiff'

    def extract_from_dataset(self, input_path, cfg):
        """ """
        with load_dataset(input_path) as dataset:
            meta = syntool_ingestor.metadata.extract_from_dataset(dataset, cfg)
        yield (meta, None)
