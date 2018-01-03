# encoding: utf-8

"""
@author: <sylvain.herledan@oceandatalab.com>
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

import datetime

def _modifier_to_timedelta(value, unit):
    """ """
    if 'd' == unit:
        return datetime.timedelta(days=value)
    elif 'h' == unit:
        return datetime.timedelta(hours=value)
    elif 'm' == unit:
        return datetime.timedelta(minutes=value)
    elif 's' == unit:
        return datetime.timedelta(seconds=value)
    elif 'ms' == unit:
        return datetime.timedelta(milliseconds=value)

    raise Exception('Unit "{}" is not supported for time modifiers'.format(unit))

def get_range(initial_date, modifiers):
    """Expected format for modifiers is ±N{d,h,m,s,ms}, where N is an integer
    value."""
    if type(initial_date) == str:
        initial_date = datetime.datetime.strptime(initial_date, '%Y-%m-%dT%H:%M:%S')

    start_modifier, stop_modifier = map(lambda x: x.replace('"', ''), modifiers)

    start_modifier_sign = start_modifier[0]
    start_modifier_value = filter(lambda x: x.isdigit(), start_modifier)
    start_modifier_value = int('{}{}'.format(start_modifier_sign, start_modifier_value))
    start_modifier_unit = filter(lambda x: x.isalpha(), start_modifier)
    start_modifier = _modifier_to_timedelta(start_modifier_value, start_modifier_unit)
    start = initial_date + start_modifier
    start_str = datetime.datetime.strftime(start, '%Y-%m-%d %H:%M:%S')

    stop_modifier_sign = stop_modifier[0]
    stop_modifier_value = filter(lambda x: x.isdigit(), stop_modifier)
    stop_modifier_value = int('{}{}'.format(stop_modifier_sign, stop_modifier_value))
    stop_modifier_unit = filter(lambda x: x.isalpha(), stop_modifier)
    stop_modifier = _modifier_to_timedelta(stop_modifier_value, stop_modifier_unit)
    stop = initial_date + stop_modifier
    stop_str = datetime.datetime.strftime(stop, '%Y-%m-%d %H:%M:%S')


    return (start_str, stop_str)
