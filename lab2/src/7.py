import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1000)
s = np.sin(2 * np.pi * 100 * t)

fig, axs = plt.subplots(3)

axs[0].plot(t, s)
axs[1].plot(t[::4], s[::4])
axs[2].plot(t[1::4], s[1::4])
for ax in axs.flat:
    ax.grid(True)

plt.savefig("../plots/7.pdf")
