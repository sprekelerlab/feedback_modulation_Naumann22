"""
Main functions for analysing the trained models.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.linalg import lstsq


def get_signal_clarity(s, y, params, n_test=None):
    """
    Compute signal clarity for over several contexts.
    input:
     - s: true sources (time x #sources)
     - y: test signal to compute signal clarity on (time x #signal dim), e.g. the network output
     - params: dictionary of parameters. Should contain 'n_sample' (default 1000) and 'n_test' if not specified else
     - n_test: number of mixings (contexts) to compute signal clarity over
     output:
     - perf: signal clarity for every context (length: n_test)
     - corr: pairwise correlations for every context (shape: n_test x # source x # test signals)
    """
    n_sig = s.shape[1]
    if not n_test:
        n_test = params['n_test']
    corr = []
    for ntest in range(n_test):
        tts, tte = ntest * params['n_sample'], (ntest + 1) * params['n_sample']  # start and stop times of context
        corr.append(np.corrcoef(s[tts:tte].T, y[tts:tte].T)[:n_sig, n_sig:])
    corr = np.array(corr)
    abs_corr = np.abs(corr)
    perf = np.nanmean(np.abs(abs_corr[:, 0, :] - abs_corr[:, 1, :]), axis=1)

    return perf, corr


def linear_source_decoding(X, s, n_mix_train, n_sample=1000):
    """
    Linear decoding of the sources from a population of neurons/units.
    input:
     - X: population of neurons/units to decode from (shape: time x N)
     - s: sources (shape: time x n_sources)
     - n_mix_train: number of contexts to use for training of the decoder
     - n_sample: number of samples for each context
    output:
     - regression score (R2) for each test context
    """

    reg = LinearRegression()
    n_mix_tot = int(len(s)/n_sample)
    z_train, s_train = X[:n_mix_train * 1000], s[:n_mix_train * 1000]
    reg.fit(z_train, s_train)
    score_all = []
    for itest in range(n_mix_train, n_mix_tot):
        z_test = X[itest * n_sample:(itest + 1) * n_sample]
        s_test = s[itest * n_sample:(itest + 1) * n_sample]
        s_pred = reg.predict(z_test)
        score = r2_score(s_test, s_pred)
        score_all.append(score)

    return np.array(score_all)


def get_normal_vector(data):
    """
    Fit a plane to 3d data and return the corresponding normal vector.
    input:
     - data: input data of shape (time x 3)
    output:
     - normal vector describing the plane, normalised to norm 1
    """

    # fit linear plane to data
    X = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    C, _, _, _ = lstsq(X, data[:, 2])

    # choose some 3 points on plane
    xx = np.array([-0.4, -0.3, 0.4])
    yy = np.array([-0.7, 0.5, -0.3])
    zz = C[0] * xx + C[1] * yy + C[2]
    points = np.array([xx, yy, zz]).T

    # compute normal vector from the 3 points
    normal = np.cross((points[1] - points[0]), (points[2] - points[0]))

    return normal / np.linalg.norm(normal)


def compute_angle(nv1, nv2):
    """
    Compute the angle between two planes based on their normal vector.
    input
     - nv1: normal vector of plane 1
     - nv2: normal vector of plane 2
    output:
     - angle between the planes, circular measure such that maximum is 180 degrees
    """

    cross = np.dot(nv1, nv2) / (np.linalg.norm(nv1) * np.linalg.norm(nv2))
    angle_rad = np.arccos(cross)
    angle_deg = np.rad2deg(angle_rad)

    if not np.isfinite(angle_deg):
        angle_deg = 0

    return np.minimum(angle_deg, 180 - angle_deg)
