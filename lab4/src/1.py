import numpy as np
import matplotlib.pyplot as plt
import time
import math

def compute_dft(s, omega_max = 100):
   dft = [np.sum([s[i] * math.e ** (-2 * np.pi * 1j * i * omega / len(s)) for i in range (len(s))]) for omega in range(omega_max)]
   return [np.sqrt(np.real(x) ** 2 + np.imag(x) ** 2) for x in dft]

custom_dft = []
numpy_dft = []
n = [128, 256, 512, 1024, 2048, 4096, 8192]
for dim in n:
    t = np.linspace(0, 1, dim)
    s = np.sin(2 * np.pi * 85 * t)

    start = time.perf_counter()
    dft = compute_dft(s)
    end = time.perf_counter()
    custom_dft.append(end - start)

    start = time.perf_counter()
    dft = np.fft.fft(s)
    end = time.perf_counter()
    numpy_dft.append(end - start)

for i in range (len(custom_dft)):
    print(f'{n[i]} esantioane: dft -> {custom_dft[i]}, numpy_dft -> {numpy_dft[i]}')
plt.plot(n, np.exp(custom_dft), label = 'custom')
plt.plot(n, np.exp(numpy_dft), label = 'numpy')
plt.xlabel('nr esantioane')
plt.ylabel('timp executie')

plt.legend()
plt.grid()
plt.savefig('../plots/1.pdf')
plt.savefig('../plots/1.png')


