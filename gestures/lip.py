import numpy as np
from scipy.spatial import distance as dist

yawn_thresh = 35

def is_yawn(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = dist.euclidean(top_mean, low_mean)
    return distance > yawn_thresh

def get_lip(shape):
    n1 = shape[48:49]
    n2 = shape[54:55]
    return n1, n2