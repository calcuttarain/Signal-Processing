import numpy as np
import matplotlib.pyplot as plt

def save_img_spec(X, Y, title, filename):
    freq_db = 20 * np.log10(abs(Y))

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    axes[0].imshow(X, cmap="cividis")
    axes[0].set_title("Image")

    im = axes[1].imshow(freq_db, cmap="cividis")
    axes[1].set_title("Spectrum")

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.04).set_label("Intensity (dB)")

    plt.suptitle(title, fontsize = 15)

    plt.savefig('../plots/' + filename + '.png', dpi=300)
    plt.savefig('../plots/' + filename + '.pdf')
    plt.clf()

N = 20
n1 = np.linspace(-1, 1, 100)
n2 = np.linspace(-1, 1, 100)
n1, n2 = np.meshgrid(n1, n2)

# 1
X = np.sin(2 * np.pi * n1 + 3 * np.pi * n2)
Y = np.fft.fft2(X)

save_img_spec(X, Y, r'$x_{n_1, n_2} = \sin(2 \pi n_1 + 3 \pi n_2)$', '1_1')


# 2
X = np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)
Y = np.fft.fft2(X)

save_img_spec(X, Y, r'$x_{n_1, n_2} = \sin(4 \pi n_1) + \cos(6 \pi n_2)$', '1_2')


# 3
Y = np.zeros((N, N), dtype = 'complex')
Y[0, 5] = 1
Y[0, N - 5] = 1
X = np.fft.ifft2(Y)
X = np.real(X)

save_img_spec(X, Y, r'$Y_{0,5} = Y_{0,N-5} = 1\text{, altfel }Y_{m_1,m_2} = 0,\ \forall m_1, m_2$', '1_3')


# 4
Y = np.zeros((N, N), dtype = 'complex')
Y[5, 0] = 1j  
Y[N - 5, 0] = 1j
X = np.fft.ifft2(Y)
X = np.imag(X)

save_img_spec(X, Y, r'$Y_{5,0} = Y_{N-5,0} = 1\text{, altfel }Y_{m_1,m_2} = 0,\ \forall m_1, m_2$', '1_4')


# 5
Y = np.zeros((N, N), dtype = 'complex')
Y[5, 5] = 1
Y[N - 5, N - 5] = 1
X = np.fft.ifft2(Y)
X = np.real(X)

save_img_spec(X, Y, r'$Y_{5,5} = Y_{N-5,N-5} = 1\text{, altfel }Y_{m_1,m_2} = 0,\ \forall m_1, m_2$', '1_5')
