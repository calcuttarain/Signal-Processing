import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 16000)
f = 10
esantioane = t[::334]
print(len(esantioane))

fig, axs = plt.subplots(3, figsize = (16, 9))
for i in range (3):
    s = np.sin(2 * np.pi * t * (f + 8 * i))
    s_esan = s[::334]

    axs[i].scatter(esantioane, s_esan, color = 'r')
    axs[i].plot(t, s)

    axs[i].set_title(f"f = {f + 7 * i}")
    axs[i].grid()

plt.suptitle("Frecventa Nyquist: fs = 48")
plt.tight_layout()
plt.savefig('../plots/3.pdf')
plt.savefig('../plots/3.png')
