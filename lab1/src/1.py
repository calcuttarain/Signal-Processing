import numpy as np
import matplotlib.pyplot as plt

def semnal_x(t):
    return np.cos(520 * np.pi * t + np.pi / 3)


def semnal_y(t):
    return np.cos(280 * np.pi * t - np.pi / 3)


def semnal_z(t):
    return np.cos(120 * np.pi * t + np.pi / 3)


def b():
    t = np.arange(0, 0.03, 0.0005)

    fig, axs = plt.subplots(3)
    fig.suptitle("Semnalele x, y, z")

    axs[0].plot(t, semnal_x(t))
    axs[0].set_title("Semnal x")

    axs[1].plot(t, semnal_y(t))
    axs[1].set_title("Semnal y")

    axs[2].plot(t, semnal_z(t))
    axs[2].set_title("Semnal z")

    for ax in axs.flat:
        ax.set_xlabel("Timp")
        ax.set_ylabel("Amplitudine") 
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('../plots/1_b.pdf')

def c():
    fs = 200
    n = np.arange(0, 0.03, 1 / fs)

    fig, axs = plt.subplots(3)
    fig.suptitle("Semnalele esantionate x, y, z")

    axs[0].stem(n, semnal_x(n))
    axs[0].set_title("Semnal eșantionat x")

    axs[1].stem(n, semnal_y(n))
    axs[1].set_title("Semnal eșantionat y")

    axs[2].stem(n, semnal_z(n))
    axs[2].set_title("Semnal eșantionat z")

    for ax in axs.flat:
        ax.set_xlabel("Timp")
        ax.set_ylabel("Amplitudine") 
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("../plots/1_c.pdf")


b()

c()
