# -*- coding: utf-8 -*-

from . import constants


def scale_kappa_real(TEMPERATURE: float, VOLUME: float) -> float:
    return constants.charge**2 / TEMPERATURE / constants.kB / VOLUME * 10000.0 * 1.0e6


def scale_kappa_metal(TEMPERATURE: float, VOLUME: float) -> float:
    return constants.charge**2 / TEMPERATURE / constants.kB / VOLUME * 10000.0


def scale_kappa_qepw(TEMPERATURE: float, VOLUME: float) -> float:
    return (
        constants.charge**2
        / TEMPERATURE
        / constants.kB
        / VOLUME
        * 10000.0
        * constants.J_PWtoMETAL**2
    )


def scale_kappa_gpumd(TEMPERATURE: float, VOLUME: float) -> float:
    return (
        constants.charge**3
        / TEMPERATURE
        / constants.massunit
        / constants.kB
        / VOLUME
        * 1.0e8
    )
