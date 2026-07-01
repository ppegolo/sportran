# -*- coding: utf-8 -*-
"""
Utilities to pretty-plot the results
"""

__all__ = [
    "plt",
    "use_plot_style",
    "Plotter",
    "MDSamplePlotter",
    "CurrentPlotter",
    "CLIPlotter",
    "PdfPages",
    "addPlotToPdf",
]

# matplotlib.use('Agg')  # if needed use force=True, warn=False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .cli import CLIPlotter
from .current import CurrentPlotter
from .mdsample import MDSamplePlotter
from .plotter import Plotter, addPlotToPdf
from .style import use_plot_style
