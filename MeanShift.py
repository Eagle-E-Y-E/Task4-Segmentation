import numpy as np
import cv2

def mean_shift_filter(image, spatial_radius=10, color_radius=20, max_level=1, max_iter=4, epsilon=1):
    """
    Mean-Shift filtering similar to OpenCV's pyrMeanShiftFiltering.
    
    Args:
        image: Input image (BGR or RGB).
        spatial_radius: Spatial bandwidth (sp).
        color_radius: Color bandwidth (sr).
        max_level: Maximum pyramid level.
        max_iter: Maximum mean-shift iterations.
        epsilon: Minimum shift to declare convergence.

    Returns:
        Segmented image.
    """
    original_shape = image.shape[:2]
    
    # Build pyramid
    pyramid = [image.copy()]
    for _ in range(max_level):
        down = cv2.pyrDown(pyramid[-1])
        pyramid.append(down)

    # Mean-shift at coarsest level
    small_img = pyramid[-1]
    result = mean_shift_core(small_img, spatial_radius, color_radius, max_iter, epsilon)

    # Upsample and refine
    for lvl in reversed(range(max_level)):
        result = cv2.pyrUp(result)
        if result.shape[:2] != pyramid[lvl].shape[:2]:
            result = cv2.resize(result, (pyramid[lvl].shape[1], pyramid[lvl].shape[0]))
        # Light refinement (optional): blend slightly with higher-res image
        # result = mean_shift_core(result, spatial_radius // 2, color_radius // 2, max_iter=2, epsilon=epsilon)

    return result

def mean_shift_core(img, spatial_radius, color_radius, max_iter=10, epsilon=1):
    """
    Core mean-shift operation using Gaussian spatial + color weights.
    """
    h, w, c = img.shape
    img = img.astype(np.float32)
    shifted = img.copy()

    # Precompute spatial kernel (only depends on window size)
    window_size = 2 * spatial_radius + 1
    y, x = np.mgrid[-spatial_radius:spatial_radius+1, -spatial_radius:spatial_radius+1]
    spatial_kernel = np.exp(-(x**2 + y**2) / (2 * (spatial_radius**2)))

    for iteration in range(max_iter):
        shifted_new = shifted.copy()
        total_shift = 0

        for i in range(h):
            for j in range(w):
                # Extract local patch
                x_min = max(j - spatial_radius, 0)
                x_max = min(j + spatial_radius + 1, w)
                y_min = max(i - spatial_radius, 0)
                y_max = min(i + spatial_radius + 1, h)

                patch = shifted[y_min:y_max, x_min:x_max]

                # Crop spatial kernel accordingly
                spatial_crop = spatial_kernel[
                    (y_min - i + spatial_radius):(y_max - i + spatial_radius),
                    (x_min - j + spatial_radius):(x_max - j + spatial_radius)
                ]

                center_color = shifted[i, j]
                color_diff = patch - center_color
                color_distance_squared = np.sum(color_diff**2, axis=2)

                # Compute color kernel
                color_kernel = np.exp(-color_distance_squared / (2 * (color_radius**2)))

                # Total weight = spatial * color
                total_kernel = spatial_crop * color_kernel

                # Weighted mean calculation
                total_weight = np.sum(total_kernel) + 1e-8  # prevent division by zero
                weighted_sum = np.sum(patch * total_kernel[:, :, np.newaxis], axis=(0, 1))

                new_value = weighted_sum / total_weight

                shifted_new[i, j] = new_value
                total_shift += np.linalg.norm(new_value - center_color)

        shifted = shifted_new

        if total_shift < epsilon * h * w:
            print(f"Converged at iteration {iteration}")
            break

    return np.clip(shifted, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    img = cv2.imread('data/flowers.jpeg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    segmented = mean_shift_filter(img_rgb, spatial_radius=8, color_radius=16, max_level=1)

    cv2.imshow('Segmented', cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
