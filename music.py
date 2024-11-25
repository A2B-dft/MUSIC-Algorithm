import numpy as np
from steer import *

def MUSIC(ang_range,noise_subspace,n,space):
    spectrum = []
    for angle in ang_range:
        sv = steer(angle,n,space)
        spectrum.append(1/np.abs(sv.conj().T @ noise_subspace @ noise_subspace.conj().T @ sv))
    return np.array(spectrum)
