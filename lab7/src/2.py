from scipy import misc
import numpy as np
import matplotlib.pyplot as plt

'''
Stabilind un SNR, de fapt consider zgomot frecventele care au o amplitudine mai mica decat SNR.
'''

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

def compress_image(X, snr):
    Y = np.fft.fft2(X)
    freq_db = 20*np.log10(abs(Y))

    Y_cutoff = Y.copy()
    Y_cutoff[freq_db < snr] = 0
    X_cutoff = np.fft.ifft2(Y_cutoff)
    X_cutoff = np.real(X_cutoff)
    
    save_img_spec(X_cutoff, f'SNR = {snr}', f'2_{snr}')
    
X = misc.face(gray=True)
save_img_spec(X, 'Imagine originala', 'imagine_originala')

SNR = [90, 100, 110, 120]
for snr in SNR:
    compress_image(X, snr)
