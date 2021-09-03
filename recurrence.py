from __future__ import division, print_function
import numpy as np
from scipy.spatial.distance import pdist, squareform

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def recurrence_plot(s, m=2, lag=1, theta=0.2):
    N = s.shape[0]
    assert N - ((m - 1) * lag) >= 1
    X = [s[st:((m - 1) * lag + st + 1):lag] for st in range(N - (m - 1) * lag)]
    X = np.array(X)
    X = normalization(X)
    d = pdist(X, metric="euclidean")
    d = np.where(d <= theta, 1, 0)
    RP = squareform(d)
    REC = np.sum(RP) / (N * (m - 1))
    return REC
