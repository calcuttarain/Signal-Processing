import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 0.3, 1/200000)

semnal_sinus = np.sin(2 * np.pi * 50 * t)
semnal_cosinus = np.cos(2 * np.pi * 50 * t - np.pi / 2)

plt.plot(t, semnal_sinus, color='b', label='Sinus')  
plt.plot(t, semnal_cosinus, color='r', label='Cosinus')

plt.grid(True)
plt.savefig("../plots/1.pdf")
