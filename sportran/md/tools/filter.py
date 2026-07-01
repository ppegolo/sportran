# -*- coding: utf-8 -*-

import numpy as np


def runavefilter(X, WF):
    """Compute a running average over ``WF`` consecutive elements.

    If ``WF`` is even it is increased by one. The input array is assumed to be
    even (``X[-i] = X[i]``) and anti-periodic (``X[(N-1)+i] = X[(N-1)-i]``),
    as for a one-sided PSD.
    """

    if WF % 2 == 0:
        WF = WF + 1
    W = int(WF / 2)

    if W > X.shape[0] - 1:
        W = X.shape[0] - 1
        WF = 2 * W
        print("Warning: reducing filtering window")

    Y = np.concatenate((X[W:0:-1], X, X[-2 : -W - 2 : -1]))
    return np.convolve(Y, np.array([1.0 / WF] * WF), "valid")
