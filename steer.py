import numpy as np
#Parameters for the array antenna

def steer(angle,n,space):
    angle_rad = np.radians(angle)
    phase_shifts = np.arange(n)*2*np.pi*space*np.sin(angle_rad)
    return np.exp(1j*phase_shifts)
