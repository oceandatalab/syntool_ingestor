# encoding: utf-8

"""
syntool_ingestor: Extraction of metadata and tiling from GeoTiff files.

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
from setuptools import setup
import subprocess

package_dir = os.path.dirname(__file__)
version_path = os.path.join(package_dir, 'VERSION.txt')

major_version = '0.1'
if os.path.exists('.git') and os.path.isdir('.git'):
    commits = subprocess.check_output([ '/usr/bin/git'
                                      , 'rev-list'
                                      , 'HEAD'
                                      , '--count']).decode('utf-8').strip()
    with open(version_path, 'w') as f:
        f.write('{}.{}\n'.format(major_version, commits))

with open(version_path, 'r') as f:
    version = f.read()

setup(
    zip_safe=False,
    name='syntool_ingestor',
    version=version,
    author=', '.join(('Sylvain Gérard <sylvain.gerard@oceandatalab.com>',
                      'Gilles Guitton <gilles.guitton@oceandatalab.com>',
                      'Sylvain Herlédan <sylvain.herledan@oceandatalab.com>')),
    author_email='syntool@oceandatalab.com',
    packages=[ 'syntool_ingestor'
             ],
    scripts=[
        'bin/syntool-ingestor',
        'bin/syntool-merger',
    ],
    license='AGPLv3',
    description='Extraction of metadata and tiling from GeoTiff files.',
    url='https://git.oceandatalab.com/syntool_odl/syntool_ingestor',
    long_description=open('README.txt').read(),
    install_requires=[ 'gdal'
                     , 'subprocess32'
                     , 'Pillow'
                     , 'numpy'
                     , 'pyproj'
                     , 'Yapsy'
                     , 'unicodecsv'
                     , 'netCDF4'
    ],
    package_data={'syntool_ingestor': [ 'share/tilers_tools/*.*'
                                      , 'share/cfg.ini.sample'
                                      , 'share/plugins/readers/*.*'
                                      , 'share/plugins/formatters/*.*'
                                      ]
    },
)
