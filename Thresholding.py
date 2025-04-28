import numpy as np
import cv2


class Thresholder:
    def __init__(self, max_iterations=100):
        self.max_iterations = max_iterations

    def thresholding(self, image):
        threshold = np.mean(image)
        for _ in range(self.max_iterations):

            image_class1 = image[image > threshold]
            image_class2 = image[image <= threshold]

            mean1 = np.mean(image_class1) if image_class1.size > 0 else 0
            mean2 = np.mean(image_class2) if image_class2.size > 0 else 0
            new_threshold = (mean1 + mean2) / 2

            if np.abs(new_threshold - threshold) < 0.5:
                break

            threshold = new_threshold

        output_image = np.zeros_like(image, dtype=np.uint8)
        output_image[image > threshold] = 255
        return output_image, threshold

    def otsu_thresholding(self, image):
        histogram, _ = np.histogram(image, bins=256, range=(0, 256))
        histogram = histogram / histogram.sum()  # normalize the histogram to get probabilities
        max_variance = 0
        optimal_threshold = 0

        for t in range(256):
            q1 = np.sum(histogram[:t + 1])  # Weight of Class 1
            q2 = np.sum(histogram[t + 1:])  # Weight of Class 2
            if q1 == 0 or q2 == 0:
                continue
            # Mean Calculation
            mean_1 = np.sum(np.arange(t + 1) * histogram[:t + 1]) / q1 if q1 > 0 else 0
            mean_2 = np.sum(np.arange(t + 1, 256) * histogram[t + 1:]) / q2 if q2 > 0 else 0

            # Maximizing the Between-class variance
            variance = q1 * q2 * (mean_1 - mean_2) ** 2
            if variance > max_variance:
                max_variance = variance
                optimal_threshold = t

        output_image = np.zeros_like(image, dtype=np.uint8)
        output_image[image > optimal_threshold] = 255
        return output_image, optimal_threshold

    def spectral_thresholding(self, image, num_modes=5, smooth_window=50):
        # Compute and normalize histogram
        histogram, _ = np.histogram(image, bins=256, range=(0, 256))
        histogram = histogram / histogram.sum()

        kernel = np.ones(smooth_window) / smooth_window
        smoothed_histogram = np.convolve(histogram, kernel, mode='same')

        peaks_indices = []
        for i in range(1, len(smoothed_histogram) - 1):
            if smoothed_histogram[i] > smoothed_histogram[i - 1] and smoothed_histogram[i] > smoothed_histogram[i + 1]:
                peaks_indices.append(i)

        if len(peaks_indices) < num_modes:
            print(f"Warning: Found only {len(peaks_indices)} peaks, expected {num_modes}.")
            num_modes = len(peaks_indices)
        if num_modes < 2:
            print("Error: Need at least 2 peaks for spectral thresholding. Returning binary image.")
            return self.otsu_thresholding(image)

        peak_indices = sorted(peaks_indices, key=lambda x: smoothed_histogram[x], reverse=True)[:num_modes]
        peak_indices = sorted(peak_indices)

        thresholds = []
        for i in range(len(peak_indices) - 1):
            valley_idx = np.argmin(smoothed_histogram[peak_indices[i]:peak_indices[i + 1]]) + peak_indices[i]
            thresholds.append(valley_idx)

        output_image = np.zeros_like(image, dtype=np.uint8)
        levels = [0] + thresholds + [255]
        print(levels)
        for i in range(len(levels) - 1):
            mask = (image > levels[i]) & (image <= levels[i + 1])
            output_image[mask] = i * (255 // (len(levels) - 1))

        return output_image, thresholds

    def local_thresholding(self, image, block_size=200, C=10):
        if block_size % 2 == 0:
            block_size += 1
            print(f"Warning: block_size must be odd. Adjusted to {block_size}.")
        pad = block_size // 2
        padded = np.pad(image, pad, mode='reflect')
        height, width = image.shape[:2]

        output_image = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                region = padded[i:i + block_size, j:j + block_size]
                local_threshold = np.mean(region) - C
                output_image[i, j] = 255 if image[i, j] >= local_threshold else 0

        return output_image, []

    def update_output(self,image, method, num_modes_, smooth_window_, block_size_, C):
        num_modes = max(2, num_modes_)
        smooth_window = max(1, smooth_window_)
        block_size = max(3, block_size_)
        if block_size % 2 == 0:
            block_size += 1

        if method == 'Iterative':
            output_image, thresholds = self.thresholding(image)
            title = f'Iterative (Threshold: {thresholds:.2f})'
        elif method == 'Otsu':
            output_image, thresholds = self.otsu_thresholding(image)
            title = f'Otsu (Threshold: {thresholds})'
        elif method == 'Spectral':
            output_image, thresholds = self.spectral_thresholding(image, num_modes, smooth_window)
            title = f'Spectral (Thresholds: {thresholds})'
        else:
            output_image, thresholds = self.local_thresholding(image, block_size, C)
            title = f'Local (Block Size: {block_size}, C: {C})'

        return output_image, thresholds


# Main script

# # image = cv2.imread('Data/Thresholding/chainsaw.png', cv2.IMREAD_GRAYSCALE)
# sp = Thresholder()
#
# output_image, thresholds = sp.local_thresholding(image)
# print(thresholds)
# cv2.imshow('Spectral Thresholding Output', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
