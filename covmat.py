from steer import *
import numpy as np
from music import *
import matplotlib.pyplot as plt

#Steering Vectors for each signal
n = int(input('Enter the number of array elements: '))
spacing = float(input('Specify the element spacing: '))
nsig = int(input('Enter the number of incoming signals: '))

angles = np.random.uniform(0, 180, nsig)

A = np.array([steer(angle,n,spacing) for angle in angles]).T

#Generate random signals
np.random.seed(0)
sig_pwr = 10
sigs = np.sqrt(sig_pwr/2)*(np.random.randn(nsig,100)) + 1j*np.random.randn(nsig,100)
noise = (np.random.randn(n,100) + 1j*np.random.randn(n,100))

#Array output
X = A @ sigs + noise
#Covariance Matrix
R = np.cov(X)

#Eigenvalue decomposition
eigvals, eigvecs = np.linalg.eigh(R)
idx = eigvals.argsort()[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:,idx]

noise_subspace = eigvecs[:,nsig:]


#Construction of MUSIC pseudo spectrum
angle_range = np.linspace(0,360,5000)
spectrum = MUSIC(angle_range,noise_subspace,n,spacing)

estimated_angles = angle_range[np.argsort(spectrum)[-nsig:]]
print("Estimated DOAs:", estimated_angles)

#Plotting and locating the peaks
plt.plot(angle_range, 10*np.log10(spectrum/np.max(spectrum)), label='MUSIC')
for angle in estimated_angles:
    plt.axvline(x=angle,color='red',linestyle='--',label=f'Estimated DOA: {angle:.1f}°')

plt.title('MUSIC SPECTRUM')
plt.xlabel('Angle (degrees)')
plt.ylabel('Spatial Spectrum (dB)')
plt.legend()
plt.grid()
plt.show()