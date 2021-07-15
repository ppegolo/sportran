# -*- coding: utf-8 -*-

from .plotter import Plotter

class CurrentPlotter(Plotter):

    ## DA CAMBIARE QUI>......................

    def plot_periodogram(self, current, PSD_FILTER_W=None, freq_units='THz', freq_scale=1.0, axes=None, kappa_units=False,
                         FIGSIZE=None, mode='log', **plot_kwargs):   # yapf: disable
        return super().plot_periodogram(current=current, PSD_FILTER_W=PSD_FILTER_W, freq_units=freq_units,
                                        freq_scale=freq_scale, axes=axes, kappa_units=kappa_units, FIGSIZE=FIGSIZE,
                                        mode=mode, **plot_kwargs)

    def plot_ck(self, current, axes=None, label=None, FIGSIZE=None):
        return super().plot_ck(current, axes, label, FIGSIZE)

    def plot_L0_Pstar(self, current, axes=None, label=None, FIGSIZE=None):
        return super().plot_L0_Pstar(current, axes, label, FIGSIZE)

    def plot_kappa_Pstar(self, current, axes=None, label=None, FIGSIZE=None):
        return super().plot_kappa_Pstar(current, axes, label, FIGSIZE)

    def plot_cepstral_spectrum(self, current, freq_units='THz', freq_scale=1.0, axes=None, kappa_units=True, FIGSIZE=None, mode='log',
                               **plot_kwargs):   # yapf: disable
        return super().plot_cepstral_spectrum(current, freq_units, freq_scale, axes, kappa_units, FIGSIZE, mode,
                                              **plot_kwargs)

    def plot_fstar_analysis(self, current, xf, FSTAR_THZ_LIST, axes=None, FIGSIZE=None, **plot_kwargs):
        return super().plot_fstar_analysis(current, xf, FSTAR_THZ_LIST, axes, FIGSIZE, **plot_kwargs)

    def plt_resample(self, current, xf, axes=None, freq_units='THz', PSD_FILTER_W=None, FIGSIZE=None, mode='log'):
        return super().plt_resample(current, xf, axes, freq_units, PSD_FILTER_W, FIGSIZE, mode)
