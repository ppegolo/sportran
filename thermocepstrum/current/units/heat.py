# -*- coding: utf-8 -*-

from . import constants


def scale_kappa_real(TEMPERATURE, VOLUME):
    """
    Conversion factor for the thermal conductivity from REAL LAMMPS units to SI units.
    INPUT:
    TEMPERATURE [K]
    VOLUME      cell VOLUME [A^3]
    """
    return (constants.kcal / constants.NA / TEMPERATURE)**2 / constants.kB / VOLUME * 100.


def scale_kappa_metal(TEMPERATURE, VOLUME):
    """
    Conversion factor for the thermal conductivity from METAL LAMMPS units to SI units.
    INPUT:
    TEMPERATURE [K]
    VOLUME      cell VOLUME [A^3]
    """
    return (constants.charge / TEMPERATURE)**2 / constants.kB / VOLUME * 10000.


def scale_kappa_qepw(TEMPERATURE, VOLUME):
    """
    Conversion factor for the thermal conductivity from Quantum Espresso PW/HARTREE-HEATCURRENT units to SI units.
    INPUT:
    TEMPERATURE [K]
    VOLUME      cell VOLUME [A^3]
    """
    return (constants.charge / TEMPERATURE)**2 / constants.kB / VOLUME * 10000. * constants.J_PWtoMETAL**2


def scale_kappa_gpumd(TEMPERATURE, VOLUME):
    """
    Conversion factor for the thermal conductivity from GPUMD units to SI units.
    INPUT:
    TEMPERATURE [K]
    VOLUME      cell VOLUME [A^3]
    """
    return (constants.charge)**3 / (TEMPERATURE)**2 / constants.massunit / constants.kB / VOLUME * 1.0e8


def scale_kappa_dlpoly(TEMPERATURE, VOLUME):
    """
    Conversion factor for the thermal conductivity from DL_POLY units to SI units.
    INPUT:
    TEMPERATURE [K]
    VOLUME      cell VOLUME [A^3]
    """
    return (1.0 / constants.NA / TEMPERATURE)**2 / constants.kB / VOLUME * 1e10

def scale_kappa_lj(TEMPERATURE, VOLUME):
    """
    Conversion factor for the thermal conductivity in LJ LAMMPS units.
    INPUT:
    TEMPERATURE [eps/kB]
    VOLUME      cell VOLUME [sigma^3]
    """
    return (1 / TEMPERATURE)**2 / VOLUME

def scale_kappa_METALVIStoSI(TEMPERATURE, VOLUME):
    # To be deleted: compute shear viscosity using the HeatCurrent class
    kB = 1.3806504
    bartoPa=1
    """
    Conversion factor for viscosity from REAL units to CGS (Poise) units.
    INPUT:
    temp      =  temperature [ K ]
    volume    =  cell volume [ A^3 ]
    timestep  =  integration time step [ fs ]
    """
    return (VOLUME / kB / TEMPERATURE * bartoPa**2 * 1e-11)
