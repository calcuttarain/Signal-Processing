import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.fft import dctn, idctn
from constants import Q_JPEG

X = misc.ascent()
h, w = X.shape

X_jpeg = np.zeros_like(X, dtype=np.float32)

for i in range(0, h, 8):
    for j in range(0, w, 8):
        block = X[i:i+8, j:j+8]
        dct_block = dctn(block, norm='ortho')
        
        quantized_block = Q_JPEG * np.round(dct_block / Q_JPEG)
        
        idct_block = idctn(quantized_block, norm='ortho')
        X_jpeg[i:i+8, j:j+8] = idct_block

fig, axes = plt.subplots(1, 2, figsize=(16, 9))

axes[0].imshow(X, cmap=plt.cm.gray)
axes[0].set_title('Imagine Originala')

axes[1].imshow(X_jpeg, cmap=plt.cm.gray)
axes[1].set_title('Imagine Reconstruita dupa Compresie')

plt.savefig("../images/1.png", dpi = 300)
plt.imsave('../images/1_original_image.png', X, cmap=plt.cm.gray)
plt.imsave('../images/1_jpeg_reconstructed_image.png', X_jpeg, cmap=plt.cm.gray)
