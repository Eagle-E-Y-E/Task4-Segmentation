import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# ----------------------------
# Smooth histogram function
# ----------------------------
def smooth_histogram(hist, sigma=2):
    return gaussian_filter1d(hist, sigma)

# ----------------------------
# Function to restore original color to regions
# ----------------------------
def color_regions(original_img, seg_map):
    """
    For each region label in seg_map, compute the average color from original_img,
    then create an output image where every pixel in the region is filled with that color.
    """
    colored_output = np.zeros_like(original_img)
    labels = np.unique(seg_map)
    for label in labels:
        if label == 0:
            continue  # skip background/unlabeled
        mask = seg_map == label
        mean_color = cv2.mean(original_img, mask=(mask.astype(np.uint8) * 255))[:3]
        colored_output[mask] = np.array(mean_color, dtype=np.uint8)
    return colored_output

def region_growing(img, seed, tol, mode='gray', convert_to_lab=True):
    """
    Integrated region growing function that works for both grayscale and color images.

    Parameters:
      img            - Input image.
                       For 'gray' mode, this should be a 2D array.
                       For 'color' mode, this should be a BGR image if convert_to_lab is True,
                       or an image already in Lab color space if convert_to_lab is False.
      seed           - Tuple (x, y) indicating the starting pixel.
      tol            - Tolerance threshold.
                       In grayscale mode, this is the maximum allowed intensity difference.
                       In color mode, this is the maximum allowed Euclidean distance between Lab vectors.
      mode           - 'gray' to perform region growing on a grayscale image,
                       'color' to perform region growing in Lab space.
      convert_to_lab - Applicable only when mode='color'. If True, the BGR input image is converted
                       to Lab color space before processing.
                       
    Returns:
      seg - A binary mask (of the same spatial dimensions as the processed image) with 255 for pixels in
            the grown region.
    """
    if mode == 'gray':
        # For grayscale images: assume img is a (rows, cols) array.
        rows, cols = img.shape
        seg = np.zeros_like(img, dtype=np.uint8)
        seg[seed] = 255
        seed_intensity = int(img[seed])
        stack = [seed]
        
        while stack:
            x, y = stack.pop()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Skip the seed pixel itself
                    xn, yn = x + dx, y + dy
                    if 0 <= xn < rows and 0 <= yn < cols:
                        if seg[xn, yn] == 0 and abs(int(img[xn, yn]) - seed_intensity) <= tol:
                            seg[xn, yn] = 255
                            stack.append((xn, yn))
        return seg

    elif mode == 'color':
        # For color images: optionally convert BGR to Lab.
        if convert_to_lab:
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        else:
            img_lab = img
        rows, cols, _ = img_lab.shape
        seg = np.zeros((rows, cols), dtype=np.uint8)
        x, y = seed
        seg[x, y] = 255
        seed_color = img_lab[x, y].astype(np.float32)
        stack = [seed]
        
        while stack:
            x, y = stack.pop()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    xn, yn = x + dx, y + dy
                    if 0 <= xn < rows and 0 <= yn < cols:
                        if seg[xn, yn] == 0:
                            pix_color = img_lab[xn, yn].astype(np.float32)
                            if np.linalg.norm(pix_color - seed_color) <= tol:
                                seg[xn, yn] = 255
                                stack.append((xn, yn))
        return seg

    else:
        raise ValueError("Unsupported mode. Use either 'gray' or 'color'.")


def unsupervised_region_segmentation(img, peaks, tol=15, peak_tol=2, mode='gray'):
    """
    Perform unsupervised segmentation using histogram peak seeds.
    
    Parameters:
      img      - Input image. For grayscale mode, it should be a single-channel image;
                 for color mode, it should be a BGR image.
      peaks    - Peaks detected from the histogram in the corresponding intensity channel.
      tol      - Tolerance for region growing.
      peak_tol - Allowed deviation from the peak for candidate seed selection.
      mode     - 'gray' to perform grayscale segmentation, 'color' for color segmentation.
      
    Returns:
      seg_map - Integer label map for the segmented regions.
    """
    if mode == 'gray':
        # Assume img is already a grayscale (2D) image.
        rows, cols = img.shape
        seg_map = np.zeros((rows, cols), dtype=np.int32)
        region_label = 1

        for peak in peaks:
            # Find candidate seeds within a narrow band around the peak.
            candidates = np.argwhere((np.abs(img - peak) <= peak_tol) & (seg_map == 0))
            for (x, y) in candidates:
                if seg_map[x, y] == 0:
                    region = region_growing(img, (x, y), tol, mode='gray')
                    seg_map[region == 255] = region_label
                    region_label += 1
        return seg_map

    elif mode == 'color':
        # Convert image to Lab color space.
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L_channel = img_lab[:, :, 0]
        rows, cols = L_channel.shape
        seg_map = np.zeros((rows, cols), dtype=np.int32)
        region_label = 1

        for peak in peaks:
            # Use the L-channel for candidate seed selection.
            candidates = np.argwhere((np.abs(L_channel - peak) <= peak_tol) & (seg_map == 0))
            for (x, y) in candidates:
                if seg_map[x, y] == 0:
                    region = region_growing(img_lab, (x, y), tol, mode='color')
                    seg_map[region == 255] = region_label
                    region_label += 1
        return seg_map

    else:
        raise ValueError("Unsupported mode. Choose 'gray' or 'color'.")


# ----------------------------
# Unified segmentation function
# ----------------------------
def segment_image(img, tol=15, peak_tol=2, prominence=10, distance=10):
    if img is None:
        raise ValueError(f"Error: Could not read the image.")
    
    # Check if the image is effectively grayscale.
    process_as_color = True
    if len(img.shape) == 3 and img.shape[2] == 3:
        if np.allclose(img[..., 0], img[..., 1], atol=1) and np.allclose(img[..., 0], img[..., 2], atol=1):
            print("Detected as Grayscale image (stored in 3 channels).")
            img = img[..., 0]
            process_as_color = False
        else:
            print("Detected as Color image.")
    elif len(img.shape) == 2:
        print("Detected as Grayscale image.")
        process_as_color = False
    else:
        raise ValueError("Unsupported image shape.")
    
    # Perform histogram analysis and peak detection based on image type.
    if not process_as_color:
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        smoothed_hist = smooth_histogram(hist, sigma=2)
        global_max_peak = np.argmax(smoothed_hist)
        peaks, _ = find_peaks(smoothed_hist, prominence=prominence, distance=distance)
        peaks = np.append(peaks, global_max_peak)
        peaks = np.unique(peaks)
        print("Detected grayscale peaks (intensity):", peaks)
        # (Optional) display the histogram.
        plt.figure(figsize=(10, 4))
        plt.plot(smoothed_hist, label="Smoothed Histogram")
        plt.plot(peaks, smoothed_hist[peaks], "x", label="Peaks")
        plt.title("Grayscale Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
        seg_map = unsupervised_region_segmentation(img, peaks, tol, peak_tol, mode='gray')
        # For display purposes, normalize.
        seg_out = cv2.normalize(seg_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return seg_out  # for grayscale, return the normalized segmentation map.
    else:
        # For color segmentation, perform analysis using the L-channel.
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L_channel = img_lab[:, :, 0]
        hist = cv2.calcHist([L_channel], [0], None, [256], [0, 256]).flatten()
        smoothed_hist = smooth_histogram(hist, sigma=2)
        global_max_peak = np.argmax(smoothed_hist)
        peaks, _ = find_peaks(smoothed_hist, prominence=prominence, distance=distance)
        peaks = np.append(peaks, global_max_peak)
        peaks = np.unique(peaks)
        print("Detected peaks in L-channel:", peaks)
        # (Optional) display the histogram.
        plt.figure(figsize=(10, 4))
        plt.plot(smoothed_hist, label="Smoothed L-channel Histogram")
        plt.plot(peaks, smoothed_hist[peaks], "x", label="Peaks")
        plt.title("L-channel Histogram")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
        seg_map = unsupervised_region_segmentation(img, peaks, tol, peak_tol, mode='color')
        # Restore original color in each region.
        seg_out = color_regions(img, seg_map)
        return seg_out