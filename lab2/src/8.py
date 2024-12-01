import numpy as np
import matplotlib.pyplot as plt

def aproximari(alfa):
    plt.figure(figsize = (16, 9))
    plt.plot(alfa, (alfa - 7 * alfa ** 3 / 60) / (1 + alfa ** 2 / 20), label = "Pade")
    plt.plot(alfa, alfa, label = "Taylor")
    plt.plot(alfa, np.sin(alfa), label = "sin(x)")
    plt.grid()
    plt.title('Aproximarile Taylor si Pade')
    plt.legend()
    plt.savefig("../plots/8_aproximari.pdf")

def eroare_aproximari(alfa):
    fig, axs = plt.subplots(2, figsize=(16, 9)) 
    axs[0].plot(alfa, np.abs(np.sin(alfa) - alfa))
    axs[0].grid()
    axs[0].set_title("Eroarea aproximarii Taylor")
    axs[0].set_xlabel("alfa")
    axs[0].set_ylabel("eroarea")
    axs[0].set_yscale("log")

    axs[1].plot(alfa, np.abs(np.abs(np.sin(alfa) - (alfa - 7 * alfa ** 3 / 60) / (1 + alfa ** 2 / 20))))
    axs[1].grid()
    axs[1].set_title("Eroarea aproximarii Pade")
    axs[1].set_xlabel("alfa")
    axs[1].set_ylabel("eroarea")
    axs[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig("../plots/8_erori.pdf")

alfa = np.linspace(-np.pi, np.pi, 4000)

aproximari(alfa)
eroare_aproximari(alfa)
