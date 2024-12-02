import numpy as np
import matplotlib.pyplot as plt

def save_plots(time, signal1, signal2, label1, label2, title, filename):
    plt.figure(figsize=(16, 9))
    plt.plot(time, signal1, label = label1, color = 'purple')
    plt.plot(time, signal2, label = label2, color = "gold")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('../plots/' + filename + '.pdf')
    plt.clf()

def exponential_smoothing(signal, alpha):
    smoothed_signal = []
    smoothed_signal.append(signal[0])

    for i in range(1, len(signal)):
        s = alpha * signal[i] + (1 - alpha) * smoothed_signal[i - 1]
        smoothed_signal.append(s)

    return smoothed_signal


# 1)
polinom = {'a': 3, 'b': 5, 'c': 4}
freq = [12, 18]
time = np.linspace(0, 1, 1000)

trend = polinom['a'] * time ** 2 + polinom['b'] * time + polinom['c']
seasonal = np.sin(2 * np.pi * freq[0] * time) + np.sin(2 * np.pi * freq[1] * time)
white_noise = np.random.normal(0, 0.3, 1000)

signal = trend + seasonal + white_noise


# 2)
alpha = 0.1
smoothed_signal = exponential_smoothing(signal, alpha)

save_plots(time, signal, smoothed_signal, 'signal', 'smoothed_signal', f'Exponential Smoothing with chosen $\\alpha = {alpha}$', '2_1')

a = np.array([signal[t] - smoothed_signal[t - 1] for t in range (1, len(signal) - 1)])
b = np.array([signal[t + 1] - smoothed_signal[t - 1] for t in range (1, len(signal) - 1)])

best_alpha = (a.T @ b) / np.dot(a, a)
best_alpha_smoothed_signal = exponential_smoothing(signal, best_alpha)

save_plots(time, signal, best_alpha_smoothed_signal, 'signal', 'smoothed_signal', f'Exponential Smoothing with calculated $\\alpha = {best_alpha:.5f}$', '2_2')


# 3)
