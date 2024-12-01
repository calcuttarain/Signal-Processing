import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage

def save_img_spec(X, title, filename):
    Y = np.fft.fft2(X)
    freq_db = 20 * np.log10(abs(Y))

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    axes[0].imshow(X, cmap=plt.cm.gray)
    axes[0].set_title("Image")

    im = axes[1].imshow(freq_db, cmap="cividis")
    axes[1].set_title("Spectrum")

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.04).set_label("Intensity (dB)")

    plt.suptitle(title, fontsize = 15)

    plt.savefig('../plots/' + filename + '.png', dpi=300)
    plt.savefig('../plots/' + filename + '.pdf')
    plt.clf()

def computeSNR(original_signal, noisy_signal):
    original_signal_fft = np.fft.fft2(original_signal)
    p_original = np.sum(np.abs(original_signal_fft ** 2))

    noisy_signal_fft = np.fft.fft2(noisy_signal)
    p_noisy = np.sum(np.abs(noisy_signal_fft ** 2))

    return p_original / p_noisy

def noising(X):
    pixel_noise = 200

    noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
    return X + noise

def low_pass_filtering(X, sigma):
    rows, cols = X.shape
    Y = np.fft.fftshift(np.fft.fft2(X))
    center_x, center_y = rows // 2, cols // 2
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if distance >= sigma:
                Y[i][j] = 0
    
    X_filtered = np.fft.ifft2(np.fft.ifftshift(Y))
    
    return np.real(X_filtered)


x = misc.face(gray=True)
x_noisy = noising(x)
save_img_spec(x_noisy, f"Noisy Image: SNR = {computeSNR(x, x_noisy)}", "noisy_image")

filtered_x = low_pass_filtering(x_noisy, 40)
save_img_spec(filtered_x, f"Filtered Image: SNR = {computeSNR(x, filtered_x)}", 'filtered_image')
