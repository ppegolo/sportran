# -*- coding: utf-8 -*-

from typing import Any

from . import Current

__all__ = ["GenericCurrent"]


class GenericCurrent(Current):
    """
    GenericCurrent API for thermo-cepstral analysis. Defines a HeatCurrent object with
    useful tools to perform analysis.
    """

    _current_type = None
    _input_parameters = {"DT_FS", "KAPPA_SCALE"}
    _KAPPA_SI_UNITS = ""

    @property
    def _builder(self) -> dict[str, Any]:
        return dict(
            DT_FS=self.DT_FS,
            KAPPA_SCALE=self.KAPPA_SCALE,
            PSD_FILTER_W=self.PSD_FILTER_W_THZ,
            FREQ_UNITS="THz",
        )
