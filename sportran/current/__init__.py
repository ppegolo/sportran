# -*- coding: utf-8 -*-
"""Current and units registry.

This module handles the current types and units supported by the library.
The currently supported entries can be inspected with
:py:func:`build_currents_units_table`.

To add a new current type and/or unit:

1. Define a class derived from :py:class:`.current.Current` and set:

   - ``_current_type`` used in the user interfaces.
   - ``_input_parameters`` (usually ``{'DT_FS', 'UNITS', 'TEMPERATURE', 'VOLUME'}``).
   - ``_KAPPA_SI_UNITS`` describing the output units.

2. Define ``__init__`` as:

   .. code-block:: python

      def __init__(self, traj, **params):
          super().__init__(traj, **params)

3. Define ``_get_builder`` so an equivalent object can be rebuilt:

   .. code-block:: python

      CurrentType, builder = self._get_builder()
      new_ts = CurrentType(**builder)

4. Define unit conversion functions in :py:mod:`.current.units`:

   - Add a file named ``_current_type.py``.
   - Add functions named ``scale_kappa_*``; the suffix becomes the unit name in
     user interfaces.
"""

from .current import Current
from .electric import *
from .generic import *
from .heat import *
from .stress import *
from .thermoelectric import *

__all__ = [
    "GenericCurrent",
    "HeatCurrent",
    "ElectricCurrent",
    "StressCurrent",
    "ThermoElectricCurrent",
]

# define list of all classes with units defined
import inspect

_all = dir()


def _get_currents_with_units():
    """
    Inspect all the classes accessible from this module, and detect the ones that contains the attribute `_current_type`.
    Then call the `get_units` method to inspect the units implemented for each discovered class.
    :return: ( {'_current_type': (CurrentClass, ['unit_list'], ['parameter_list'])} )
    """
    currents_with_units = {}
    all_units = []
    all_parameters = []
    for k in _all:
        v = globals()[k]
        att = getattr(v, "_current_type", None)
        if att is not None:
            parameters = []
            units = []
            for unit, funct in v._get_units().items():
                param = list(inspect.signature(funct).parameters.keys())
                units.append(unit)
                parameters += param
            parameters = list(set(parameters))
            currents_with_units[att] = (v, units, parameters)
            all_units += units
            all_parameters += parameters
    all_units = list(set(all_units))
    all_parameters = list(set(all_parameters))
    return currents_with_units, all_units, all_parameters


def _list_of_currents_and_units(verbose=False):
    s = ""
    for k, v_ in all_currents.items():
        v = v_[0]
        s += f"'{k}': {v._input_parameters}\n"
        for u in v.get_units_list():
            s += f"   - '{u}'\n"
            if verbose:
                s += v._get_units()[u].__doc__
    return s


# list of currents classes, units implemented and parameters that are found dynamically when the module is imported
all_currents, all_units, all_parameters = _get_currents_with_units()


def build_currents_units_table(col=9):
    """Print a table with the Current classes and the units implemented for each class"""
    table = ""
    table += " " * col
    for u in all_units:
        table += u[:col].ljust(col)
    table += "\n"
    for k, v in all_currents.items():
        table += k[:col].ljust(col)
        for unit in all_units:
            if unit in v[1]:
                table += "X".ljust(col)
            else:
                table += " " * col
        table += "\n"
    return table
