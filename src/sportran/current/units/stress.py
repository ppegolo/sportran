# -*- coding: utf-8 -*-

from . import constants


def scale_kappa_GPa(TEMPERATURE: float, VOLUME: float) -> float:
    return 1.0e-4 * VOLUME / TEMPERATURE / constants.kB


def scale_kappa_real(TEMPERATURE: float, VOLUME: float) -> float:
    return constants.atm**2 * 1.0e-12 * VOLUME / TEMPERATURE / constants.kB


def scale_kappa_metal(TEMPERATURE: float, VOLUME: float) -> float:
    return 1.0e-12 * VOLUME / TEMPERATURE / constants.kB


def scale_kappa_qepw(TEMPERATURE: float, VOLUME: float) -> float:
    return (
        (constants.charge * constants.Ry_per_bohr3) ** 2
        * VOLUME
        / TEMPERATURE
        / constants.kB
    )


def scale_kappa_gpumd(TEMPERATURE: float, VOLUME: float) -> float:
    return constants.charge**2 * VOLUME / TEMPERATURE / constants.kB
