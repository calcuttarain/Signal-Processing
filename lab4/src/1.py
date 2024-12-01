import numpy as np
import matplotlib.pyplot as plt
import time
import math

custom_dft = []
numpy_fft = []
n = [128, 256, 512, 1024, 2048, 4096, 8192]

for dim in n:
    t = np.linspace(0, 1, dim)
    s = np.sin(2 * np.pi * 85 * t)

    start = time.perf_counter()
    dft = [np.sum([s[i] * math.e ** (-2 * np.pi * 1j * i * omega / dim) for i in range (dim)]) for omega in range(dim)]
    end = time.perf_counter()
    custom_dft.append(end - start)

    start = time.perf_counter()
    dft = np.fft.fft(s)
    end = time.perf_counter()
    numpy_fft.append(end - start)

for i in range (len(custom_dft)):
    print(f'{n[i]} esantioane: dft -> {custom_dft[i]}, numpy_fft -> {numpy_fft[i]}')

'''
128 esantioane: dft -> 0.028716792003251612, numpy_fft -> 0.0044430000125430524
256 esantioane: dft -> 0.11348383300355636, numpy_fft -> 5.441700341179967e-05
512 esantioane: dft -> 0.5091474170039874, numpy_fft -> 7.67909805290401e-05
1024 esantioane: dft -> 1.8278434579842724, numpy_fft -> 6.533399573527277e-05
2048 esantioane: dft -> 7.2255365420132875, numpy_fft -> 0.00011237498256377876
4096 esantioane: dft -> 28.753891625005053, numpy_fft -> 0.000166166020790115
8192 esantioane: dft -> 116.03217858300195, numpy_fft -> 0.0004845829971600324
'''

plt.figure(figsize=(16, 9))
plt.plot(n,custom_dft, label = 'custom dft')
plt.plot(n, numpy_fft, label = 'numpy fft')
plt.xlabel('nr esantioane')
plt.ylabel('timp executie')
plt.yscale('log')

plt.legend()
plt.grid()
plt.savefig('../plots/1.pdf')
plt.savefig('../plots/1.png')
