import matplotlib.pyplot as plt
import numpy as np
import math

def compute_dft(s, t, omega_max = 100):
    dft = [np.sum([s[i] * math.e ** (-2 * np.pi * 1j * i * omega / len(s)) for i in range (len(s))]) for omega in range(omega_max)]
    modul_dft = [np.sqrt(np.real(x) ** 2 + np.imag(x) ** 2) for x in dft]

    fig, axs = plt.subplots(1, 2, figsize = (16, 9))

    axs[0].plot(t, s, linewidth=1)
    axs[1].stem(np.arange(100), modul_dft)

    axs[0].grid()
    axs[0].set_ylabel('amplitudine')
    axs[0].set_xlabel('timp')

    axs[1].grid()
    axs[1].set_ylabel("|X(ω)|")
    axs[1].set_xlabel("frecventa")
    axs[0].set_box_aspect(1)
    axs[1].set_box_aspect(1)

    plt.suptitle("Discrete Fourier Transform")
    plt.tight_layout()
    plt.savefig('../plots/3.png', format='png', dpi=300) 
    plt.savefig('../plots/3.pdf', format='pdf')

t = np.linspace(0, 1, 10000)
s = np.sin(2 * np.pi * t * 5) + 3 * np.sin(2 * np.pi * t * 25) + 0.5 * np.sin(2 * np.pi * t * 70)
compute_dft(s, t)
