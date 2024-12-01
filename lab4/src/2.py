import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 16000)
f = 10
esantioane = t[::2000]

fig, axs = plt.subplots(3, figsize=(16, 9))
for i in range (3):
    s = np.sin(2 * np.pi * t * (f + 8 * i))
    s_esan = s[::2000]

    axs[i].scatter(esantioane, s_esan, color = 'r')
    axs[i].plot(t, s)

    axs[i].set_title(f"f = {f + 7 * i}")
    axs[i].grid()

plt.suptitle("Frecventa sub-Nyquist: fs = 8")
plt.tight_layout()
plt.savefig('../plots/2.pdf')
plt.savefig('../plots/2.png')
