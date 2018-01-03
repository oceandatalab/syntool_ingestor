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
import osr
import math
import json
import numpy
import errno
import logging
import ConfigParser
from datetime import datetime, timedelta
import netCDF4

from syntool_ingestor.interfaces import IFormatterPlugin
from syntool_ingestor.utils import NumpyAwareJSONEncoder

logger = logging.getLogger(__name__)

class BuoyDirSpectraPlugin(IFormatterPlugin):
    """ """

    def can_handle(self, target_format):
        """ """
        return 'buoy_dir_spectrum' == target_format

    def get_output_id(self, cfg):
        """ """
        return ''

    def get_representation_type(self, cfg):
        """ """
        return 'MOORED'

    def _create_metadata(self, output_path, bbox):
        """"""

        minX, minY, maxX, maxY = bbox
        topleft = '{} {}'.format(minX, maxY)
        topright = '{} {}'.format(maxX, maxY)
        bottomright = '{} {}'.format(maxX, minY)
        bottomleft = '{} {}'.format(minX, minY)
        extent = 'POLYGON(({},{},{},{},{}))'.format( topleft, topright
                                                   , bottomright, bottomleft
                                                   , topleft)

        metadata = { 'min_zoom_level': 0
                   , 'max_zoom_level': 0
                   , 'resolutions': []
                   , 'bbox_str': extent
                   , 'shape_str': extent
                   }

        return metadata

    def create_representation( self, input_path, input_data, workspace
                             , cfg, src_meta):
        """ """

        spectrum_data, bbox = self._create_spectrum(input_path, workspace, cfg, src_meta)

        # spectrum
        output_name = '{}.spectrum.json'.format(src_meta['dataset'])
        spectrum_file_name = os.path.join(workspace, output_name)
        with open(spectrum_file_name, "w") as outfile:
            json.dump( spectrum_data, outfile, cls=NumpyAwareJSONEncoder
                     , indent=4, sort_keys=True)

        # features
        features_dir = os.path.join(workspace, 'features')
        if not os.path.exists(features_dir):
            try:
                os.makedirs(features_dir)
            except OSError, e:
                if e.errno != errno.EEXIST:
                    raise

        metadata_list = [
            {
                "variable": "lon",
                "units": "°"
            },
            {
                "variable": "lat",
                "units": "°"
            },
            {
                "variable": "depth",
                "units": "m"
            },
            {
                "variable": "distance_to_shore",
                "units": "m"
            },
        ]

        feature_file_name = os.path.join(features_dir, 'spectrum_0.ini')
        config_parser = ConfigParser.ConfigParser()
        with open(feature_file_name, "w") as feature_file:
            config_parser.add_section("global")
            config_parser.set("global", "display_type", "SPECTRUM")
            config_parser.set("global", "feature_type", "Buoy")
            config_parser.add_section("metadata")
            for metadata in metadata_list:
                variable = metadata["variable"]
                value = ("%.2f" % spectrum_data[variable]) + metadata["units"]
                config_parser.set("metadata", variable, value)
            config_parser.set("metadata", "spectrum", "spectrum.json")
            config_parser.write(feature_file)

        # Metadata
        metadata = self._create_metadata(workspace, bbox)
        metadata['output_path'] = [ os.path.abspath(spectrum_file_name)
                                  , os.path.abspath(features_dir)
                                  ]
        return metadata

    def _create_spectrum(self, input_path, workspace, cfg, src_meta):
        """ """
        # Get settings
        opts = cfg['input_options']
        spectral_density_var = opts.get( 'spectral_density_var'
                                       , 'sea_surface_variance_spectral_density')
        central_var = opts.get( 'central_var', 'central_frequency')
        range_var = opts.get('range_var', None)

        srs4326 = osr.SpatialReference()
        srs4326.ImportFromEPSG(4326)
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(cfg['output_proj'])
        transform = osr.CoordinateTransformation(srs4326, target_srs)

        time_index = src_meta.get('time_index', None)
        if time_index is None:
            raise Exception('Time index not set in source metadata!')

        time_offset = src_meta.get('time_offset', None)
        if time_offset is None:
            raise('Time offset not set in source metadata!')

        # Set constants
        d2r = 0.01745329
        r2d = 1 / d2r
        dr = 0.0174533
        nbin = 24
        Kbin = 360 / nbin


        # Extract data
        dataset = netCDF4.Dataset(input_path, 'r')

        nb_elems = len(dataset.dimensions["frequency"])

        th1m = dataset.variables["theta1"][:].flatten()
        th2m = dataset.variables["theta2"][:].flatten()
        sth1m = dataset.variables["stheta1"][:].flatten()
        sth2m = dataset.variables["stheta2"][:].flatten()
        spectral_densities = dataset.variables[spectral_density_var][:].flatten()
        centrals = dataset.variables[central_var][:].flatten()
        times = dataset.variables["time"][:].flatten()

        if range_var is not None:
            ranges_ = dataset.variables[range_var][:].flatten()
        else:
            ranges = []

            ranges.append(centrals[1] - centrals[0])
            for i in range(1, nb_elems - 1):
                ranges.append((centrals[i + 1] - centrals[i - 1]) / 2)
            ranges.append(centrals[nb_elems - 1] - centrals[nb_elems - 2])

        e1 = [None, None, None, None, None]
        e2 = [None, None, None, None, None]
        ndir = [None, None, None, None, None]
        nn = 0
        for n in range(5, 3605):
            a = float(n) * .1 * dr
            e1.append(complex(math.cos(a), -math.sin(a)))
            e2.append(complex(math.cos(2*a), -math.sin(2*a)))
            angle = 1 + nn / 10
            if angle > 360:
                angle = angle - 360
            if angle < 1:
                angle = angle + 360
            ndir.append(angle)
            nn = nn + 1

        # Build formatted directional parameters
        m1 = map(lambda sth1m: abs(1 - 0.5 * (sth1m * d2r)**2) if None != sth1m else 0, sth1m)
        m1_ = zip(th1m, m1)
        a1_ = map(lambda v: v[1] * math.cos(v[0] * d2r) if None != v[0] else 0, m1_)
        b1_ = map(lambda v: v[1] * math.sin(v[0] * d2r) if None != v[0] else 0, m1_)

        m2 = map(lambda sth2m: abs(1 - 0.5 * (sth2m * d2r)**2) if None != sth2m else 0, sth2m)
        m2_ = zip(th2m, m2)
        a2_ = map(lambda v: v[1] * math.cos(v[0] * d2r) if None != v[0] else 0, m2_)
        b2_ = map(lambda v: v[1] * math.sin(v[0] * d2r) if None != v[0] else 0, m2_)

        logger.debug('a1s: {}'.format(len(a1_)))
        logger.debug('th1m: {}'.format(len(th1m)))

        # Get buoy coordinates
        lon = dataset.variables["lon"][0].item()
        lat = dataset.variables["lat"][0].item()
        x, y, _ = transform.TransformPoint(lon, lat)
        minX = x - 10000
        maxX = x + 10000
        minY = y - 10000
        maxY = y + 10000

        # start_i = 0 + nb_elems * min_time_index
        start_i = time_index * nb_elems
        end_i = start_i + nb_elems
        ordinate_values = spectral_densities[start_i:end_i]
        # centrals = centrals_[start_i:end_i]
        if range_var:
            ranges = ranges_[start_i:end_i]
        a1 = a1_[start_i:end_i]
        a2 = a2_[start_i:end_i]
        b1 = b1_[start_i:end_i]
        b2 = b2_[start_i:end_i]

        logger.debug('Count: {}'.format(nb_elems))
        logger.debug('Index: {}'.format(start_i))

        nfreq = nb_elems

        ds = numpy.zeros((nbin, nfreq))

        logger.debug('nfreq = {}'.format(nfreq))

        # Browse through central frequencies
        for j in range(0, nfreq):
            logger.debug('Compute spectrum for centrals[{}]'.format(j))

            spectral_density = ordinate_values[j]
            if spectral_density is None or 0 == spectral_density:
                for k in range(0, nbin):
                    ds[k, j] = 0
                continue

            # Compute spectrum
            d = numpy.zeros(361)
            chk = 0.

            # Method MEM
            # ----------

            # Switch to Lygre & Krogstad notation
            d1 = a1[j]
            d2 = b1[j]
            d3 = a2[j]
            d4 = b2[j]
            c1 = complex(d1, d2)
            c2 = complex(d3, d4)
            p1 = (c1 - c2 * c1.conjugate())/(1. - abs(c1)**2)
            p2 = c2 - c1 * p1
            x = 1. - p1 * c1.conjugate() - p2 * c2.conjugate()

            # Sum in .1 deg steps, get distribution with 1 degree resolution
            tot = 0
            for n in range(5, 3604 + 1):
                y = abs(complex(1., 0) - p1 * e1[n] - p2 * e2[n])**2
                d[ndir[n]] = d[ndir[n]] + abs(x / y) / 3600.
                tot = tot + abs(x / y)

            # Compute value for each angle index
            for k in range(Kbin, 360, Kbin):
                sum = 0
                for l in range(k - Kbin/2, 1+k+Kbin/2):
                    sum += d[l]
                ds[k / Kbin, j] = spectral_density * sum * nbin / (2 * 3.141592)
            sum = 0
            for l in range(360 - Kbin/2, 1+360):
                sum += d[l]
            for l in range(1, 1 + Kbin/2):
                sum += d[l]
            ds[0, j] = spectral_density * sum * nbin / (2 * 3.141592)

        a1s = numpy.zeros(nfreq)
        b1s = numpy.zeros(nfreq)
        thet = numpy.zeros(nfreq)
        etot = numpy.zeros(nfreq)
        efmax = float('-inf')
        ifp = -1
        hs = 0
        for n in range(0, nfreq):
            if None == ranges[n]:
                continue

            for k in range(0, nbin):
                a1s[n] += math.cos(k * Kbin * d2r) * ds[k][n]
                b1s[n] += math.sin(k * Kbin * d2r) * ds[k][n]
                etot[n] += ds[k][n]

            if etot[n] * Kbin * d2r > efmax:
                efmax = etot[n] * Kbin * d2r
                ifp = n

            a1s[n] *= Kbin * d2r
            b1s[n] *= Kbin * d2r
            logger.debug('{} of {}'.format(n, len(ranges)))
            etot[n] *= ranges[n] * Kbin * d2r
            hs += etot[n]
            thet[n] = math.atan2(b1s[n], a1s[n])

            """
            if thet[j] < 0:
                thet[j] += 2 * 3.141592
            """

        logger.debug('Efmax: {}'.format(efmax))
        logger.debug('A1S\n {}\n'.format(str(a1s)))
        logger.debug('B1S\n {}\n'.format(str(b1s)))
        logger.debug('THET\n {}\n'.format(str(thet * r2d)))

        hs = 4 * math.sqrt(hs)
        fp = centrals[ifp]
        meandir = (270 - thet[ifp] * r2d) % 360
        if efmax != 0:
            spread = math.sqrt(abs(2*(1 - (a1s[ifp]*math.cos(thet[ifp]) + b1s[ifp] * math.sin(thet[ifp])) / efmax))) * r2d
        else:
            spread = 0

        hsValue = hs
        pfValue = fp
        mdValue = meandir
        spValue = spread

        logger.debug('Hs: {}'.format(hs))
        logger.debug('Fp: {}'.format(fp))
        logger.debug('Mean dir: {}'.format(meandir))
        logger.debug('Spread: {}'.format(spread))

        flat = ds.flatten('C').tolist()
        #print >> sys.stderr, 'Number of flat items: %i' % len(flat)

        tValue = times[time_index] + time_offset
        abCountValue = nfreq
        yValue = nbin
        if 1 == nfreq:
            x_values = [centrals[0]]
            z_values = [ranges[0]]
        else:
            x_values = centrals[0:nb_elems]
            z_values = ranges[0:nb_elems]
        y_values = flat

        # Extrema
        xMin = min(y_values)
        xMax = max(y_values)
        aMin = min(x_values)
        aMax = max(x_values)
        bMin = min(z_values)
        bMax = max(z_values)

        data_conf = {}
        data_conf['t'] = tValue                  # Time
        data_conf['abCount'] = abCountValue      # Number of frequencies
        data_conf['y'] = yValue                  # Number of angles
        data_conf['a'] = x_values                  # Central frequencies
        data_conf['b'] = z_values                 # Frequency ranges
        data_conf['x'] = y_values                  # Spectral density
        data_conf['hs'] = hsValue
        data_conf['principal_freq'] = pfValue
        data_conf['mean_dir'] = mdValue
        data_conf['spread'] = spValue
        data_conf['metadata'] = {}
        data_conf['metadata']['model'] = 'directional_spectrum'
        data_conf['metadata']['x'] = {}
        data_conf['metadata']['x']['name'] = 'spectral_wave_density'
        data_conf['metadata']['x']['min'] = xMin
        data_conf['metadata']['x']['max'] = xMax
        data_conf['metadata']['t'] = {}
        data_conf['metadata']['t']['name'] = 'Time'
        data_conf['metadata']['a'] = {}
        data_conf['metadata']['a']['name'] = 'Central frequencies'
        data_conf['metadata']['a']['min'] = aMin
        data_conf['metadata']['a']['max'] = aMax
        data_conf['metadata']['b'] = {}
        data_conf['metadata']['b']['name'] = 'Frequency ranges'
        data_conf['metadata']['b']['min'] = bMin
        data_conf['metadata']['b']['max'] = bMax
        data_conf['metadata']['y'] = {}
        data_conf['metadata']['y']['name'] = 'Number of angles'
        data_conf['metadata']['abCount'] = {}
        data_conf['metadata']['abCount']['name'] = 'Number of frequencies'
        data_conf['metadata']['hs'] = {}
        data_conf['metadata']['hs']['name'] = 'Significant Height'
        data_conf['metadata']['principal_freq'] = {}
        data_conf['metadata']['principal_freq']['name'] = 'Principal Frequency'
        data_conf['metadata']['mean_dir'] = {}
        data_conf['metadata']['mean_dir']['name'] = 'Mean direction'
        data_conf['metadata']['spread'] = {}
        data_conf['metadata']['spread']['name'] = 'Spread'

        for variable in ['lon', 'lat', 'depth', 'distance_to_shore']:
            data_conf[variable] = dataset.variables[variable][0].item()

        return data_conf, [minX, minY, maxX, maxY]
