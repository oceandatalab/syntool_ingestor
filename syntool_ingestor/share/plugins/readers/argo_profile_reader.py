# encoding: utf-8

"""
@author <sylvain.herledan@oceandatalab.com>
"""

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
import sys
import ogr
import netCDF4
import numpy
import numpy.ma
import logging
import pyproj
from datetime import datetime, timedelta
from syntool_ingestor.interfaces import IReaderPlugin

logger = logging.getLogger(__name__)


REF_TABLE_4 = { 'AO': 'AOML, USA'
              , 'BO': 'BODC, United Kingdom'
              , 'CI': 'Institute of Ocean Sciences, Canada'
              , 'CS': 'CSIRO, Australia'
              , 'GE': 'BSH, Germany'
              , 'GT': 'GTS : used for data coming from WMO GTS network'
              , 'HZ': 'CSIO, China Second Institute of Oceanography'
              , 'IF': 'Ifremer, France'
              , 'IN': 'INCOIS, India'
              , 'JA': 'JMA, Japan'
              , 'JM': 'Jamstec, Japan'
              , 'KM': 'KMA, Korea'
              , 'KO': 'KORDI, Korea'
              , 'ME': 'MEDS, Canada'
              , 'NA': 'NAVO, USA'
              , 'NM': 'NMDIS, China'
              , 'PM': 'PMEL, USA'
              , 'RU': 'Russia'
              , 'SI': 'SIO, Scripps, USA'
              , 'SP': 'Spain'
              , 'UW': 'University of Washington, USA'
              , 'VL': 'Far Eastern Regional Hydrometeorological Research Institute of Vladivostock, Russia'
              , 'WH': 'Woods Hole Oceanographic Institution, USA'
              }

REF_TABLE_8 = { '831': 'P-Alace float'
              , '840': 'Provor, no conductivity'
              , '841': 'Provor, Seabird conductivity sensor'
              , '842': 'Provor, FSI conductivity sensor'
              , '843': 'POPS ice Buoy/Float'
              , '844': 'Arvor, Seabird conductivity sensor'
              , '845': 'Webb Research, no conductivity'
              , '846': 'Webb Research, Seabird sensor'
              , '847': 'Webb Research, FSI sensor'
              , '850': 'Solo, no conductivity'
              , '851': 'Solo,  Seabird conductivity sensor'
              , '852': 'Solo, FSI conductivity sensor'
              , '853': 'Solo2, Seabird conductivity sensor'
              , '855': 'Ninja, no conductivity sensor'
              , '856': 'Ninja, SBE conductivity sensor'
              , '857': 'Ninja, FSI conductivity sensor'
              , '858': 'Ninja, TSK conductivity sensor'
              , '859': 'Profiling Float, NEMO, no conductivity'
              , '860': 'Profiling Float, NEMO, SBE conductivity sensor'
              , '861': 'Profiling Float, NEMO, FSI conductivity sensor'
              }


class CycleNumbersNotFound(Exception):
    pass

class ArgoProfileReaderPlugin(IReaderPlugin):
    """ """

    @staticmethod
    def _get_cycle_numbers(variables):
        """ """
        if 'CYCLE_NUMBER_INDEX_ADJUSTED' in variables \
        and not variables['CYCLE_NUMBER_INDEX_ADJUSTED'][:].all() is numpy.ma.masked:
            return variables['CYCLE_NUMBER_INDEX_ADJUSTED'][:]

        if 'CYCLE_NUMBER_INDEX' in variables \
        and not variables['CYCLE_NUMBER_INDEX'][:].all() is numpy.ma.masked:
            return variables['CYCLE_NUMBER_INDEX'][:]

        if 'CYCLE_NUMBER_ACTUAL' in variables \
        and not variables['CYCLE_NUMBER_ACTUAL'][:].all() is numpy.ma.masked:
            return variables['CYCLE_NUMBER_ACTUAL'][:]

        if 'CYCLE_NUMBER_ADJUSTED' in variables:
            cycle_numbers = numpy.unique(variables['CYCLE_NUMBER_ADJUSTED'][:])
            if not cycle_numbers.all() is numpy.ma.masked:
                return cycle_numbers

        if 'CYCLE_NUMBER' in variables:
            cycle_numbers = numpy.unique(variables['CYCLE_NUMBER'][:])
            if not cycle_numbers.all() is numpy.ma.masked:
                return cycle_numbers

        raise CycleNumbersNotFound()

    @staticmethod
    def _sanitize_coords(lats, lons, dates, qc):
        """ """
        _coords = zip(lats, lons, dates, qc)
        return filter(lambda x: x[0] is not numpy.ma.masked and x[1] is not numpy.ma.masked and x[2] is not None, _coords)

    @staticmethod
    def _sanitize_profile_data(pressure, temperature, salinity):
        """ """
        if not isinstance(pressure, numpy.ndarray) and not isinstance(pressure, numpy.ma.core.MaskedArray):
            pressure = [pressure]
        if not isinstance(temperature, numpy.ndarray) and not isinstance(temperature, numpy.ma.core.MaskedArray):
            temperature = [temperature]
        if not isinstance(salinity, numpy.ndarray) and not isinstance(salinity, numpy.ma.core.MaskedArray):
            salinity = [salinity]
        indices = {'pressure': {'index': 0, 'values': pressure}}
        idx = 1
        if temperature is not None:
            indices['temperature'] = {'index': idx, 'values': temperature}
            idx = idx + 1
        if salinity is not None:
            indices['salinity'] = {'index': idx, 'values': salinity}
            idx = idx +1

        _data = zip(*[v['values'] for _,v in indices.iteritems()])
        data = filter(lambda x: None not in x and numpy.ma.masked not in x, _data)
        if 0 >= len(data):
            return None, None, None
        values = zip(*data)

        _pressure = values[0]

        _temperature = None
        if 'temperature' in indices:
            _temperature = values[indices['temperature']['index']]

        _salinity = None
        if 'salinity' in indices:
            _salinity = values[indices['salinity']['index']]

        return _pressure, _temperature, _salinity

            

    @staticmethod
    def _get_profile_vars(handler, varnames, cycle_idx, indices):
        """ """
        for varname in varnames:
            if varname not in handler.variables:
                continue

            if indices is not None:
                values = numpy.take(handler.variables[varname][cycle_idx][:], indices)
            else:
                values = handler.variables[varname][cycle_idx][:]

            if 1 > len(values) or values.all() is numpy.ma.masked:
                continue

            return values

        return None

    def can_handle(self, source_format):
        """ """
        return 'argo_profile' == source_format.lower()

    def extract_from_dataset(self, input_path, cfg):
        """ """
        date_fmt = '%Y-%m-%d %H:%M:%S'
        geod =pyproj.Geod(ellps='WGS84')
        filedir = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        m = re.match('^([a-zA-Z0-9]+)_(Rtraj|prof)\.nc$', filename)
        if m is None:
            raise Exception('Input filename should be XXXX_Rtraj.nc or XXXX_prof.nc')

        identifier, _ = m.groups()
        traj_filepath = os.path.join(filedir, '{}_Rtraj.nc'.format(identifier))
        prof_filepath = os.path.join(filedir, '{}_prof.nc'.format(identifier))

        if not os.path.exists(traj_filepath):
            raise Exception('Missing trajectory file "{}"'.format(traj_filepath))

        if not os.path.exists(prof_filepath):
            raise Exception('Missing profile file "{}"'.format(prof_filepath))
        
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

        with netCDF4.Dataset(traj_filepath, 'r') as traj_file, \
             netCDF4.Dataset(prof_filepath, 'r') as prof_file:

            base_date_str = ''.join(traj_file.variables['REFERENCE_DATE_TIME'][:])
            base_date = datetime.strptime(base_date_str, "%Y%m%d%H%M%S")

            cycle_numbers = ArgoProfileReaderPlugin._get_cycle_numbers(traj_file.variables)
            
            for cycle_idx in xrange(0, len(cycle_numbers)):

                cycle_number = cycle_numbers[cycle_idx]
                if cycle_number not in prof_file.variables['CYCLE_NUMBER'][:]:
                    # Cycle does not contain a profile
                    continue

                # Extract trajectory
                indices = numpy.where(traj_file.variables['CYCLE_NUMBER'][:] == cycle_number)[0]
                lats = numpy.take(traj_file.variables['LATITUDE'][:], indices)
                lons = numpy.take(traj_file.variables['LONGITUDE'][:], indices)
                juld = numpy.take(traj_file.variables['JULD'][:], indices)
                qc = numpy.take(traj_file.variables['POSITION_QC'][:], indices)
                d = map(lambda x: base_date + timedelta(x) if not x is numpy.ma.masked else None, juld)

                coords = ArgoProfileReaderPlugin._sanitize_coords(lats, lons, d, qc)
                if 0 >= len(coords):
                    logger.error('Cycle {} contains no data'.format(cycle_number))
                    continue

                metadata = { 'begin_datetime': coords[0][2].strftime(date_fmt)
                           , 'end_datetime': coords[-1][2].strftime(date_fmt)
                           , 'product': 'ARGO profilers'
                           , 'dataset': '{}_{}'.format(identifier, cycle_number)
                           , 'extra': { 'Cycle number': int(cycle_number)
                                      , 'WMO ID': traj_file.variables['PLATFORM_NUMBER'][:].tostring().strip()
                                      , 'Positioning system': traj_file.variables['POSITIONING_SYSTEM'][:].tostring().strip()
                                      , 'Project': traj_file.variables['PROJECT_NAME'][:].tostring().strip()
                                      , 'Principal investigator': traj_file.variables['PI_NAME'][:].tostring().strip()
                                      , 'Data centre': REF_TABLE_4[traj_file.variables['DATA_CENTRE'][:].tostring().strip()]
                                      }
                           }
                if 'INST_REFERENCE' in traj_file.variables:
                    metadata['extra']['Instrument'] = traj_file.variables['INST_REFERENCE'][:].tostring().strip()

                if 'WMO_INST_TYPE' in traj_file.variables:
                    wmo_inst_type = traj_file.variables['WMO_INST_TYPE'][:].tostring().strip()
                    if wmo_inst_type in REF_TABLE_8:
                        metadata['extra']['WMO inst. type'] = REF_TABLE_8[wmo_inst_type]
                    else:
                        metadata['extra']['WMO inst. type'] = wmo_inst_type

                if 'GROUNDED' in traj_file.variables:
                    grounded_idx = cycle_idx
                    if 'N_CYCLE' in traj_file.dimensions and len(traj_file.dimensions['N_CYCLE']) <= max(cycle_numbers):
                        grounded_idx = cycle_idx - 1
                    grounded = traj_file.variables['GROUNDED'][cycle_idx]
                    if not isinstance(grounded, numpy.ma.MaskedArray):
                        metadata['extra']['Grounded'] = grounded

                data = []
                skip_count = 0
                last_lon = None
                last_lat = None
                last_d = None
                for lat, lon, _d, _qc in coords:
                    # 0: no QC
                    # 1: Good data
                    # 2: Probably good data
                    # 3: Correctable bad data
                    # 4: Bad data
                    # 5: Value changed
                    # 8: Interpolated value
                    # 9: Missing value
                    if _qc not in [numpy.ma.masked, '0', '1', '8']:
                        continue

                    # Insert a noop when the trajectory goes out of the bbox
                    if valid_extent is not None:
                        p = ogr.Geometry(ogr.wkbPoint)
                        p.AddPoint_2D(lons[i], lats[i])
                        if not p.Within(valid_extent):
                            data.append({'lat': None, 'lon': None})
                            skip_count = skip_count + 1
                            continue

                    speed = 0.0
                    if None not in [last_lon, last_lat, last_d]:
                        dt = (_d - last_d).total_seconds()

                        try:
                            _, _, dist = geod.inv(last_lon, last_lat, lon, lat)
                        except ValueError:
                            logger.error('Could not compute distance between {} {} and {} {}'.format(last_lon, last_lat, lon, lat))
                            continue
                        if dt > 0.0:
                            speed = dist / dt
                    last_lon = lon
                    last_lat = lat
                    last_d = _d

                    item = { 'lon': lon if lon < 180.0 else lon - 360.0
                           , 'lat': lat
                           , 'date': _d.strftime(date_fmt)
                           , 'fields': {'speed': speed}
                           }
                    data.append(item)
                if skip_count == len(data) or 0 >= len(data):
                    logger.error('No valid trajectory data for cycle "{}" of profiler "{}"'.format(cycle_number, identifier))
                    continue

                # Extract profile data
                prof_cycle_idx = numpy.where(prof_file.variables['CYCLE_NUMBER'][:] == cycle_number)[0][0]

                indices = None
                if 'PRES_ADJUSTED_ERROR' in prof_file.variables \
                and prof_file.variables['PRES_ADJUSTED_ERROR'][prof_cycle_idx][:].all() is not numpy.ma.masked:
                    # Only keep data with a pressure error inferior 
                    # to 20 decibars, as advised in:
                    # http://www.argodatamgt.org/Data-Mgt-Team/News/Pressure-Biases
                    indices = numpy.where(prof_file.variables['PRES_ADJUSTED_ERROR'][prof_cycle_idx][:] < 20)[0]

                pressure = ArgoProfileReaderPlugin._get_profile_vars( prof_file
                                                                    , ['PRES_ADJUSTED', 'PRES']
                                                                    , prof_cycle_idx
                                                                    , indices)
                if pressure is None:
                    logger.error('No pressure information for cycle {} of {}'.format(cycle_number, identifier))
                    # Without pressure information it is impossible to draw a profile => skip this cycle
                    continue

                temperature = ArgoProfileReaderPlugin._get_profile_vars( prof_file
                                                                       , ['TEMP_ADJUSTED', 'TEMP']
                                                                       , prof_cycle_idx
                                                                       , indices)
                salinity = ArgoProfileReaderPlugin._get_profile_vars( prof_file
                                                                    , ['PSAL_ADJUSTED', 'PSAL']
                                                                    , prof_cycle_idx
                                                                    , indices)

                pressure, temperature, salinity = ArgoProfileReaderPlugin._sanitize_profile_data(pressure, temperature, salinity)
                if pressure is None:
                    continue
                data[0]['fields']['pressure'] = pressure
                if temperature is not None:
                    data[0]['fields']['temperature'] = temperature
                if salinity is not None:
                    data[0]['fields']['salinity'] = salinity

                yield(metadata, data)
