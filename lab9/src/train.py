import numpy as np
import matplotlib.pyplot as plt

def save_plots(time1, time2, signal1, signal2, label1, label2, title, filename):
    plt.figure(figsize=(16, 9))
    plt.plot(time1, signal1, label = label1, color = 'purple')
    plt.plot(time2, signal2, label = label2, color = "gold")
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

# moving average model
def ma_model(signal, errors, q):
    n = len(signal)
    X = np.zeros((n - q, q))
    y = signal[q:]

    for t in range(q, n):
        X[t - q, :] = errors[t - q:t][::-1]

    theta = np.linalg.pinv(X.T @ X) @ X.T @ y

    return theta

def ma_predict(signal, errors, q):
    miu = np.mean(signal)
    predictions = []
    thetas = np.concatenate(([1], ma_model(signal, errors, q), [1]))
    for index in range(q + 1, len(signal)):
        time_horizon = errors[index: index - q - 1: -1]
        time_horizon = np.append(time_horizon, miu)
        prediction = np.dot(thetas, time_horizon.T)
        predictions.append(prediction)

    return np.array(predictions)

# autoregressive model
def ar_model(signal, q, p):
    Y = np.array([signal[-p-i:len(signal)-i] for i in range(q)])
    y = signal[-q:]
    
    x = np.linalg.inv(Y.T @ Y) @ Y.T @ y

    return x

def ar_predict(signal, q, p):
    start = q + p 
    predictions = []

    for i in range (start, len(signal)):
        samples = signal[:i]
        x_star = ar_model(samples, q, p)
        prediction = x_star.T @ samples[-p:]
        predictions.append(prediction)

    return np.array(predictions)

# autoregressive moving average model
def arma_predict(signal, errors, q, p):
    ar_predictions = ar_predict(signal, q, p)
    ma_predictions = ma_predict(signal, errors, q)

    padding = np.zeros(p - 1)
    ar_predictions = np.concatenate((padding, ar_predictions))

    arma_predictions = ar_predictions + ma_predictions
    return arma_predictions

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

save_plots(time, time, signal, smoothed_signal, 'signal', 'smoothed_signal', f'Exponential Smoothing with chosen $\\alpha = {alpha}$', '2_1')

a = np.array([signal[t] - smoothed_signal[t - 1] for t in range (1, len(signal) - 1)])
b = np.array([signal[t + 1] - smoothed_signal[t - 1] for t in range (1, len(signal) - 1)])

best_alpha = (a.T @ b) / np.dot(a, a)
best_alpha_smoothed_signal = exponential_smoothing(signal, best_alpha)

save_plots(time, time, signal, best_alpha_smoothed_signal, 'signal', 'smoothed_signal', f'Exponential Smoothing with calculated $\\alpha = {best_alpha:.5f}$', '2_2')


# 3)
miu = np.mean(signal)
q = 15

thetas = ma_model(signal, white_noise, q)

predictions = ma_predict(signal, white_noise, q) 

save_plots(time, time[q + 1:], signal, predictions, 'signal', 'predictions', f'Moving Average Model Predictions for q = {q}', '3')


# 4)
p = 15
q = 20

predictions = arma_predict(signal, white_noise, q, p)

save_plots(time, time[q + 1:], signal, predictions, 'signal', 'predictions', f'Autoregressive Moving Average Model Predictions for q = {q}, p = {p}', '4_1')

