import numpy as np


def extract_poincare_feat(temp):
    x_bar = np.mean(np.array(temp))
    i_0 = np.array(temp[:-1])
    i_1 = np.array(temp[1:])
    d1_i = np.abs(i_0 - i_1) / np.power(2, 0.5)
    sd1 = np.sqrt(np.var(d1_i))
    d2_i = np.abs(i_0 + i_1 - (2 * x_bar)) / np.power(2, 0.5)
    sd2 = np.sqrt(np.var(d2_i))
    ratio = sd1 / sd2
    return ratio
