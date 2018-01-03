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

from yapsy.IPlugin import IPlugin

class IFormatterPlugin(IPlugin):
    """Interface that must be implemented by formatter plugins."""

    def can_handle(self, target_format):
        raise NotImplementedError()

    def get_output_id(self, cfg):
        raise NotImplementedError()

    def get_representation_type(self, cfg):
        raise NotImplementedError()

    def create_representation(self, input_path, input_data, workspace, cfg, src_meta):
        raise NotImplementedError()

class IReaderPlugin(IPlugin):
    """Interface that must be implemented by reader plugins."""

    def can_handle(self, source_format):
        raise NotImplementedError()

    def extract_from_dataset(self, input_path, cfg):
        raise NotImplementedError()


