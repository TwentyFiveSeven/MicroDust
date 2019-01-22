import numpy as np

def normalization(data):
    min = np.amin(data, axis=0)
    max = np.amax(data, axis=0)

    normalizedData = (data - min)/(max - min)

    return normalizedData
