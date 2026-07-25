# -*- coding: utf-8 -*-

from typing import Any

from . import Current

__all__ = ["ElectricCurrent"]


class ElectricCurrent(Current):
    """
    ElectricCurrent API for thermo-cepstral analysis. Defines an ElectricCurrent object
    with useful tools to perform analysis.
    """

    _current_type = "electric"
    _input_parameters = {"DT_FS", "UNITS", "TEMPERATURE", "VOLUME"}
    _KAPPA_SI_UNITS = "S/m"

    @property
    def _builder(self) -> dict[str, Any]:
        return dict(
            DT_FS=self.DT_FS,
            UNITS=self.UNITS,
            TEMPERATURE=self.TEMPERATURE,
            VOLUME=self.VOLUME,
            PSD_FILTER_W=self.PSD_FILTER_W_THZ,
            FREQ_UNITS="THz",
        )
