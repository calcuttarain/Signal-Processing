import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.fft import dctn, idctn
import cv2
from constants import *


# RGB -> YCbCr conversion
def rgb_to_ycbcr(X_rgb):
    reshaped_image = X_rgb.reshape(-1, 3) 
    ycbcr_image = np.dot(reshaped_image, RGB_TO_YCBCR.T) 
    return ycbcr_image.reshape(X_rgb.shape)

def ycbcr_to_rgb(X_ycbcr):
    reshaped_image = X_ycbcr.reshape(-1, 3)
    rgb_image = np.dot(reshaped_image, YCBCR_TO_RGB.T)
    return rgb_image.reshape(X_ycbcr.shape)


# downsample / upsample
def block_average(channel, factor):
    h, w = channel.shape
    h_reduced, w_reduced = h // factor, w // factor
    reduced = np.zeros((h_reduced, w_reduced), dtype=np.float32)

    for i in range(h_reduced):
        for j in range(w_reduced):
            block = channel[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            reduced[i, j] = np.mean(block)

    return reduced

def downsample(ycbcr_image, factor = 2):
    Y = ycbcr_image[:, :, 0]
    Cb = ycbcr_image[:, :, 1]
    Cr = ycbcr_image[:, :, 2]

    Cb_downsampled = block_average(Cb, factor)
    Cr_downsampled = block_average(Cr, factor)

    return Y, Cb_downsampled, Cr_downsampled

def upsample(Y, Cb, Cr, factor):
    Cb_upsampled = np.zeros((Y.shape[0], Y.shape[1]), dtype=np.float32)
    Cr_upsampled = np.zeros((Y.shape[0], Y.shape[1]), dtype=np.float32)

    for i in range(Cb.shape[0]):
        for j in range(Cb.shape[1]):
            block_value_cb = Cb[i, j]
            block_value_cr = Cr[i, j]
            Cb_upsampled[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor] = block_value_cb
            Cr_upsampled[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor] = block_value_cr

    ycbcr_image = np.stack([Y, Cb_upsampled, Cr_upsampled], axis=-1)

    return ycbcr_image


# quantization / dequantization
def quantize(channel, Q):
    h, w = channel.shape
    quantized_channel = np.zeros_like(channel)
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i+8, j:j+8]
            dct_block = dctn(block, norm='ortho')
            
            quantized_block = Q * np.round(dct_block / Q)
            quantized_channel[i:i+8, j:j+8] = quantized_block
            
    return quantized_channel

def dequantize(channel, Q):
    h, w = channel.shape
    dequantized_channel = np.zeros_like(channel)
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i+8, j:j+8]
            idct_block = idctn(block, norm='ortho')

            dequantized_channel[i:i+8, j:j+8] = idct_block

    return dequantized_channel


# compresion / decompresion
def compress(image, downsampling_factor = 2, Q_factor = 1):
    ycbcr_image = rgb_to_ycbcr(image)

    Y, Cb_downsampled, Cr_downsampled = downsample(ycbcr_image, downsampling_factor)

    Y_quantized = quantize(Y, Q_LUMA * Q_factor)
    Cb_quantized = quantize(Cb_downsampled, Q_CHROMA)
    Cr_quantized = quantize(Cr_downsampled, Q_CHROMA)

    save_ycbcr_components(ycbcr_image)
    save_downsampled_components(Y, Cb_downsampled, Cr_downsampled, downsampling_factor)

    return Y_quantized, Cb_quantized, Cr_quantized


def decompress(Y, Cb, Cr, downsampling_factor = 2, target_mse = 28):
    Y_dequantized = dequantize(Y, Q_LUMA)
    Cb_dequantized = dequantize(Cb, Q_CHROMA)
    Cr_dequantized = dequantize(Cr, Q_CHROMA)

    decompressed_ycbcr_image = upsample(Y_dequantized, Cb_dequantized, Cr_dequantized, downsampling_factor)
    decompressed_rgb_image = ycbcr_to_rgb(decompressed_ycbcr_image)

    decompressed_image = np.clip(decompressed_rgb_image, 0, 255).astype(np.uint8)

    return decompressed_image


# finding the best Q matrix factor for desired mse
def calculate_mse(original, decompressed):
    return np.mean((original - decompressed) ** 2)

def find_best_scaling_factor(image, target_mse, downsampling_factor=2, scale_range=(0.5, 2.5), tolerance=1e-3):
    low, high = scale_range  
    
    best_factor = None
    best_mse = float('inf')
    
    while high - low > tolerance:
        mid = (low + high) / 2
        
        Y, Cb, Cr = compress(image, downsampling_factor, mid)
        decompressed_image = decompress(Y, Cb, Cr, downsampling_factor)
        
        mse = calculate_mse(image, decompressed_image)
        print(mse)

        if int(mse) == int(target_mse):
            return mid
        
        if mse < best_mse:
            best_mse = mse
            best_factor = mid
        
        if mse < target_mse:
            low = mid
        else:
            high = mid
    
    return best_factor


# video
def process_video(input_video_path, output_video_path, downsampling_factor=2, Q_factor=1):
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        raise IOError("Error")

    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = video.read()  
        if not ret:
            break  

        frame_count += 1
        print(f"frame_count: {frame_count}...")

        Y, Cb, Cr = compress(frame, downsampling_factor, Q_factor)
        decompressed_frame = decompress(Y, Cb, Cr, downsampling_factor)

        decompressed_frame_bgr = cv2.cvtColor(decompressed_frame, cv2.COLOR_RGB2BGR)
        out.write(decompressed_frame_bgr)  

    video.release()
    out.release()

# save images
def save_ycbcr_components(ycbcr_image):
    Y = ycbcr_image[:, :, 0]
    Cb = ycbcr_image[:, :, 1]
    Cr = ycbcr_image[:, :, 2]

    ycbcr_image_normalized = ycbcr_image.astype(np.uint8)
    Cb_normalized = ((Cb - np.min(Cb)) / (np.max(Cb) - np.min(Cb)) * 255).astype(np.uint8)
    Cr_normalized = ((Cr - np.min(Cr)) / (np.max(Cr) - np.min(Cr)) * 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    axes[0][0].imshow(ycbcr_image.astype(np.uint8), cmap='coolwarm')
    axes[0][0].set_title("YCbCr Image")

    axes[0][1].imshow(Y, cmap='gray')
    axes[0][1].set_title("Y Component")
    
    axes[1][0].imshow(Cb_normalized, cmap='coolwarm')
    axes[1][0].set_title("Cb Component")
    
    axes[1][1].imshow(Cr_normalized, cmap='coolwarm')
    axes[1][1].set_title("Cr Component")

    plt.tight_layout()
    plt.suptitle("Y'CbCr components")
    plt.savefig("../images/2_ycbcr_components.png", dpi = 300)
    plt.clf()

def save_downsampled_components(Y, Cb_down, Cr_down, downsampling_factor):
    Cb_normalized = ((Cb_down - np.min(Cb_down)) / (np.max(Cb_down) - np.min(Cb_down)) * 255).astype(np.uint8)
    Cr_normalized = ((Cr_down - np.min(Cr_down)) / (np.max(Cr_down) - np.min(Cr_down)) * 255).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(Y, cmap='gray')
    axes[0].set_title("Y Component (Luminance)")
    axes[0].axis("off")
    
    axes[1].imshow(Cb_normalized, cmap='coolwarm')
    axes[1].set_title("Cb Component (Downsampled)")
    axes[1].axis("off")
    
    axes[2].imshow(Cr_normalized, cmap='coolwarm')
    axes[2].set_title("Cr Component (Downsampled)")
    axes[2].axis("off")
    
    plt.suptitle(f"Downsampled Cb and Cr components with a factor of {downsampling_factor}")
    plt.tight_layout()
    plt.savefig("../images/2_downsampled_components.png", dpi=300)

def save_original_vs_decompressed(image, decompressed_image):

    mse = calculate_mse(image, decompressed_image)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off") 

    axes[1].imshow(decompressed_image)
    axes[1].set_title("Decompressed Image")
    axes[1].axis("off")

    plt.suptitle(f"Original Image vs Decompressed Image, MSE = {mse}")
    plt.tight_layout()
    plt.savefig("../images/original_vs_decompressed.png", dpi=300)


# cerintele 2 si 3
image = misc.face()
ycbr_image = rgb_to_ycbcr(image)
downsampling_factor = 2
target_mse = 50

Q_factor = find_best_scaling_factor(image, target_mse)

Y, Cb, Cr = compress(image, downsampling_factor, Q_factor)
decompressed_image = decompress(Y, Cb, Cr)

print(f"MSE: {calculate_mse(image, decompressed_image)}") # MSE: 50.321791330973305

save_original_vs_decompressed(image, decompressed_image)


# cerinta 4
input_video_path = "../videos/input_video.mp4" 
output_video_path = "../videos/output_video.mp4"
process_video(input_video_path, output_video_path, downsampling_factor=2, Q_factor=1)
