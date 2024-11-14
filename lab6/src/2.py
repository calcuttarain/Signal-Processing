import numpy as np
import time

def polynomial_multiplication(p, q, N):
    res = [0] * (2 * N - 1)

    for i in range(N):
        for j in range(N):
            res[i + j] += p[i] * q[j]
    return res

def fft_convolution(p, q, N):
    l = 2 * N - 1

    if l & (l - 1) != 0:
        l = 1 << l.bit_length()

    p = np.concatenate((p, np.zeros(l - N)))
    q = np.concatenate((q, np.zeros(l - N)))

    p_fft = np.fft.fft(p)
    q_fft = np.fft.fft(q)

    res = np.round(np.fft.ifft(p_fft * q_fft).real) 

    return [int(c) for c in res[:2 * N - 1]]

N = 10000 

p = np.random.randint(-50, 50, size = N)
q = np.random.randint(-50, 50, size = N)

start = time.time()
res = polynomial_multiplication(p, q, N) 
end = time.time()
print(end - start) # 23.431788206100464

start = time.time()
res = fft_convolution(p, q, N)
end = time.time()
print(end - start) # 0.006051778793334961
