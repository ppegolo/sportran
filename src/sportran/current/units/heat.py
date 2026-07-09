# -*- coding: utf-8 -*-

from . import constants


def scale_kappa_real(TEMPERATURE: float, VOLUME: float) -> float:
    return (
        (constants.kcal / constants.NA / TEMPERATURE) ** 2
        / constants.kB
        / VOLUME
        * 100.0
    )


def scale_kappa_metal(TEMPERATURE: float, VOLUME: float) -> float:
    return (constants.charge / TEMPERATURE) ** 2 / constants.kB / VOLUME * 10000.0


def scale_kappa_qepw(TEMPERATURE: float, VOLUME: float) -> float:
    return (
        (constants.charge / TEMPERATURE) ** 2
        / constants.kB
        / VOLUME
        * 10000.0
        * (constants.Ry * constants.J_PWtoMETAL) ** 2
    )


def scale_kappa_gpumd(TEMPERATURE: float, VOLUME: float) -> float:
    return (
        (constants.charge) ** 3
        / (TEMPERATURE) ** 2
        / constants.massunit
        / constants.kB
        / VOLUME
        * 1.0e8
    )
