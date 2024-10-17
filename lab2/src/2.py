import numpy as np
import matplotlib.pyplot as plt

def compute_noise(snr, signal, noise):
    gamma = np.sqrt(np.linalg.norm(signal) ** 2 / (np.linalg.norm(noise) ** 2 * snr))
    return signal + gamma * noise 

def a():
    t = np.arange(0, 0.05, 1/200000)

    semnal_sinus_1 = np.sin(2 * np.pi * 50 * t + 0.5)
    semnal_sinus_2 = np.sin(2 * np.pi * 50 * t + 1.5)
    semnal_sinus_3 = np.sin(2 * np.pi * 50 * t - np.pi)
    semnal_sinus_4 = np.sin(2 * np.pi * 50 * t + np.pi/2)
 
    plt.plot(t, semnal_sinus_1, color='b', label = "faza = 0.5")
    plt.plot(t, semnal_sinus_2, color='c', label = "faza = 1.5")  
    plt.plot(t, semnal_sinus_3, color='m', label = "faza = -pi")  
    plt.plot(t, semnal_sinus_4, color='g', label = "faza = pi/2")  

    plt.legend()
    plt.savefig("../plots/2_a.pdf")

def b():
    fig, axs = plt.subplots(5)

    t = np.linspace(0, 0.05, 100)
    s = np.sin(2 * np.pi * 50 * t + 0.5)
    axs[0].plot(t, s)
    axs[0].grid(True)
    axs[0].set_title("Semnal sinusoidal")

    noise = np.random.normal(loc=0.0, scale=1.0, size=100)
    SNR = [0.1, 1, 10, 100]

    for i in range (len(SNR)):
        axs[i + 1].plot(t, compute_noise(SNR[i], s, noise) + s)
        axs[i + 1].set_title(f"SNR = {SNR[i]}")
        axs[i + 1].grid(True)

    plt.tight_layout()
    plt.savefig("../plots/2_b.pdf")
    
a()

b()
