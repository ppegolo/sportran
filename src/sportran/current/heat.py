# -*- coding: utf-8 -*-

from typing import Any

from . import Current

__all__ = ["HeatCurrent"]


class HeatCurrent(Current):
    """
    HeatCurrent API for thermo-cepstral analysis. Defines a HeatCurrent object with
    useful tools to perform analysis.
    """

    _current_type = "heat"
    _input_parameters = {"DT_FS", "UNITS", "TEMPERATURE", "VOLUME"}
    _KAPPA_SI_UNITS = "W/m/K"

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
