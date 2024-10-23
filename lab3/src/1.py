import numpy as np
import math 
import matplotlib.pyplot as plt

fm = np.zeros((8, 8), dtype = complex)

for l in range (8):
    for n in range(8):
        fm[l][n] = math.e ** (2 * np.pi * 1j * l * n / 8) * 1 / np.sqrt(8)

fig, axs = plt.subplots(8)
for n in range(8):
    real = [x.real for x in fm[n]]
    imag = [x.imag for x in fm[n]]
    axs[n].plot(real, label = "real")
    axs[n].plot(imag, label = "imaginar")
    axs[n].grid()

plt.legend()
plt.tight_layout()
plt.savefig('../plots/1.png', format='png', dpi=300) 
plt.savefig('../plots/1.pdf', format='pdf')

fm_inv = np.linalg.inv(fm)
print(np.allclose(np.eye(8), np.matmul(fm, fm_inv))) #True
