# -*- coding: utf-8 -*-

from typing import Any

from . import Current

__all__ = ["StressCurrent"]


class StressCurrent(Current):
    """
    StressCurrent API for thermo-cepstral analysis. Defines a StressCurrent object with
    useful tools to perform analysis.
    """

    _current_type = "stress"
    _input_parameters = {"DT_FS", "UNITS", "TEMPERATURE", "VOLUME"}
    _KAPPA_SI_UNITS = "Pa*s"

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
