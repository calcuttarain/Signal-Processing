import numpy as np
import matplotlib.pyplot as plt

def sin_signal(t, f):
    return np.sin(2 * np.pi * f * t)

def sawtooth_signal(t, f):
    return 2 * (t * f % 1) - 1

t = np.linspace(0, 3, 3 * 44100)
f = 400
signals = [sin_signal(t, f), sawtooth_signal(t, f), sin_signal(t, f) + sawtooth_signal(t, f)]

fig, axs = plt.subplots(3)
for i in range (3):
    axs[i].plot(t, signals[i])
    axs[i].set_xlim(0, 0.04)
    axs[i].grid(True)

plt.tight_layout()
plt.savefig("../plots/4.pdf")

