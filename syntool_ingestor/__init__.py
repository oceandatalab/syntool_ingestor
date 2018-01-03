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
import json
import errno
import logging
import traceback
import contextlib
import pkg_resources
from osgeo import gdal, ogr
from yapsy.PluginManager import PluginManager
import syntool_ingestor.metadata
import syntool_ingestor.interfaces as ifaces


# Configure logging
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def load_dataset(geotiff_path):
    """ """
    try:
        geotiff_file = gdal.Open(geotiff_path)
        yield geotiff_file
    finally:
        geotiff_file = None
        del geotiff_file

def process_file(input_path, cfg):
    """ """

    source_format = cfg.get('input_format')
    target_format = cfg.get('output_format')

    # Initialize plugins
    readers_dir = pkg_resources.resource_filename( 'syntool_ingestor'
                                                 , 'share/plugins/readers')
    fmts_dir = pkg_resources.resource_filename( 'syntool_ingestor'
                                              , 'share/plugins/formatters')
    manager = PluginManager()
    manager.setCategoriesFilter({ 'formatters': ifaces.IFormatterPlugin
                                , 'readers': ifaces.IReaderPlugin
                                })
    manager.setPluginPlaces([fmts_dir, readers_dir])
    manager.collectPlugins()

    # Extract metadata and parse input data if needed
    plugin_found = False
    for plugin_wrapper in manager.getPluginsOfCategory('readers'):
        plugin = plugin_wrapper.plugin_object
        if plugin.can_handle(source_format):
            plugin_found = True
            break
    if not plugin_found:
        raise Exception('No plugin found for "{}"'.format(source_format))

    input_generator = plugin.extract_from_dataset(input_path, cfg)

    plugin_found = False
    # Loop round the plugins to find a suitable one
    for plugin_wrapper in manager.getPluginsOfCategory('formatters'):
        plugin = plugin_wrapper.plugin_object
        if plugin.can_handle(target_format):
            plugin_found = True
            break
    if not plugin_found:
        raise Exception('No plugin found for "{}"'.format(target_format))


    metadata_list = []
    for input_data in input_generator:
        src_meta, src_data = input_data
        src_meta['output_format'] = cfg['output_format']
        src_meta['output_options'] = cfg['output_options']
        produced_meta = {}

        # Build output dir
        workspace_root = cfg.get('workspace_root', os.getcwd())
        output_id = plugin.get_output_id(cfg)
        syntool_id = '{}_{}{}'.format( cfg['output_proj']
                                     , src_meta['product']
                                     , output_id)
        syntool_id = syntool_id.replace(' ', '_')
        src_meta['syntool_id'] = syntool_id
        if 'datagroup' in src_meta:
            workspace = os.path.join( workspace_root
                                    , syntool_id
                                    , src_meta['datagroup']
                                    , src_meta['dataset'])
        else:
            workspace = os.path.join( workspace_root
                                    , syntool_id
                                    , src_meta['dataset'])
        if not os.path.exists(workspace):
            try:
                os.makedirs(workspace)
            except OSError, e:
                if e.errno != errno.EEXIST:
                    raise

        # Produce representation
        produced_meta = plugin.create_representation(input_path, src_data, workspace, cfg, src_meta)

        produced_meta['output_type'] = plugin.get_representation_type(cfg)
        produced_meta['output_level'] = plugin_wrapper.details.get('Syntool', 'RepresentationLevel')
        produced_meta['output_timeline'] = plugin_wrapper.details.get('Syntool', 'TimelineBehavior')

        if 'output_timeline' in cfg:
            produced_meta['output_timeline'] = cfg['output_timeline']

        metadata = src_meta
        metadata.update(produced_meta)

        # Simplify shape given a tolerance (0. by default)
        if not cfg['no_shape']:
            shape_geom = ogr.CreateGeometryFromWkt(metadata['shape_str'])
            shape_tolerance = float(cfg['output_options'].get('shape-tolerance', 0.))
            simp_shape_geom = shape_geom.SimplifyPreserveTopology(shape_tolerance)
            shape_wkt = simp_shape_geom.ExportToWkt()
            shape_wkt = shape_wkt.replace('POLYGON (', 'POLYGON(')
            logger.debug('Shape simplification tolerance: {}'.format(shape_tolerance))
            logger.debug('Shape before simplification: {}'.format(metadata['shape_str']))
            logger.debug('Shape after simplification: {}'.format(shape_wkt))
            if simp_shape_geom.IsEmpty():
                raise Exception('Empty simplified shape.')
            if not simp_shape_geom.IsValid():
                raise Exception('Not valid simplified shape.')
            metadata['shape_str'] = shape_wkt

        if cfg['create_feature_metadata']:
            features_path = os.path.join(workspace, 'features')
            try:
                os.makedirs(features_path)
            except OSError, e:
                if e.errno != errno.EEXIST:
                    raise

            metadata_ini_path = os.path.join(features_path, 'metadata.ini')
            syntool_ingestor.metadata.write_ini(metadata_ini_path, metadata)
            if isinstance(metadata['output_path'], list):
                metadata['output_path'].append(features_path)
            else:
                metadata['output_path'] = [metadata['output_path'], features_path]

        # Save metadata
        metadata_path = os.path.join(workspace, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        metadata_list.append(metadata_path)


    return metadata_list
