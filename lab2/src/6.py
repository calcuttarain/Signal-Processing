import numpy as np
import matplotlib.pyplot as plt

fs = 100
t = np.linspace(0, 1, fs)

f_1 = fs / 2
f_2 = fs / 4

fig, axs = plt.subplots(3)

axs[0].plot(t, np.sin(2 * np.pi * f_1 * t))
axs[1].plot(t, np.sin(2 * np.pi * f_2 * t))
axs[2].plot(t, np.zeros_like(t))

plt.tight_layout()
plt.savefig("../plots/6.pdf")

#semnalul cu f_2 este mai bine esantionat decat cel cu f_1, deoarece esantionarea e mai "deasa" daca ne raportam la frecventa primului semnal care e de doua ori mai mare decat frecventa celui de-al doilea
#pentru frecventa 0, se obtine oscilatia constanta 0
