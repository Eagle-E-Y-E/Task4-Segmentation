import cv2
import numpy as np
import os

# Global variables to hold clustering information and configuration
clusters_list = []
cluster = {}    # mapping from pixel (as tuple) to cluster index
centers = {}
n_channels = None  # number of channels the image has (1 for grayscale, 3 for color)
initial_k = None   # initial number of clusters to start with
clusters_number = None  # final number of clusters

def calculate_distance(x1, x2):
    """Compute the Euclidean distance between two vectors."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def clusters_average_distance(cluster1, cluster2):
    """
    Compute the distance between two clusters based on the Euclidean distance between their centers.
    Centers are computed by taking the average along axis 0 so that the color/intensity dimensions are preserved.
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return calculate_distance(cluster1_center, cluster2_center)

def initial_clusters(image_clusters):
    """
    Initialize clusters by a simple quantization of pixel intensities (for grayscale)
    or colors (for color images). The pixel values get assigned to the nearest pre-determined group.
    """
    global initial_k, n_channels
    groups = {}
    cluster_color = int(256 / initial_k)    # defines the step size between quantized values.
    
    if n_channels == 1:
        # For grayscale images, keys are 1-element tuples.
        for cluster_index in range(initial_k):
            val = cluster_index * cluster_color
            groups[(val,)] = []
        for pixel in image_clusters:
            p_tuple = tuple(pixel)
            best_key = min(groups.keys(), key=lambda c: np.linalg.norm(np.array(p_tuple) - np.array(c)))    # find the closest key
            groups[best_key].append(pixel)
    else:
        # For color images, we build keys as tuples with three identical values.
        for cluster_index in range(initial_k):
            color = cluster_index * cluster_color
            groups[(color, color, color)] = []
        for pixel in image_clusters:
            p_list = pixel.tolist()
            best_key = min(groups.keys(), key=lambda c: np.linalg.norm(np.array(p_list) - np.array(c)))
            groups[best_key].append(pixel)
    return [group for group in groups.values() if len(group) > 0]

def get_cluster_center(point):
    """
    Retrieve the center of the cluster to which a given pixel belongs.
    The point is converted to a tuple form (with one element if grayscale) for the dictionary lookup.
    """
    global cluster, centers, n_channels
    point_tuple = tuple(point) if n_channels > 1 else (point[0],)
    point_cluster_num = cluster[point_tuple]
    center = centers[point_cluster_num]
    return center

def get_clusters(image_clusters):
    """
    Agglomerate (merge) clusters until the number of clusters drops to the desired clusters_number.
    The merging is done based on the minimum Euclidean distance between cluster centers.
    """
    global clusters_list, cluster, centers, clusters_number
    clusters_list = initial_clusters(image_clusters)

    # Merge clusters until we have the desired number of clusters.
    while len(clusters_list) > clusters_number:
        # Find the two clusters with the smallest distance between their centers.
        cluster1, cluster2 = min(
            [(c1, c2) for i, c1 in enumerate(clusters_list) for c2 in clusters_list[:i]],
            key=lambda c: clusters_average_distance(c[0], c[1])
        )
        # Remove the two clusters that will be merged.
        clusters_list = [cl for cl in clusters_list if cl is not cluster1 and cl is not cluster2]
        merged_cluster = cluster1 + cluster2
        clusters_list.append(merged_cluster)

    # Map each pixel (as a tuple) to its corresponding cluster number.
    cluster = {}
    for cl_num, cl in enumerate(clusters_list):
        for point in cl:
            point_tuple = tuple(point) if n_channels > 1 else (point[0],)
            cluster[point_tuple] = cl_num

    # Compute the center of each cluster.
    centers = {}
    for cl_num, cl in enumerate(clusters_list):
        centers[cl_num] = np.average(cl, axis=0)

def apply_agglomerative_clustering(image, number_of_clusters, initial_number_of_clusters):
    """
    Process the supplied image using the agglomerative segmentation method.
    
    Parameters:
    - image: Input image as a NumPy array. This can be a grayscale (2D) or color (3D) image.
    - number_of_clusters: Final (desired) number of clusters for segmentation.
    - initial_number_of_clusters: A larger number of clusters to start with before merging.
    
    The function resizes the image to a fixed resolution (256x256) for uniform segmentation,
    processes the clustering, and then displays the output segmentation.
    """
    global clusters_number, initial_k, n_channels

    # Determine if the image is grayscale or color.
    if len(image.shape) == 2:
        n_channels = 1
    else:
        n_channels = image.shape[2]

    # Resize image to a fixed size (256,256).
    resized_image = cv2.resize(image, (256, 256))
    clusters_number = number_of_clusters
    initial_k = initial_number_of_clusters

    # Reshape the image pixels into a flat list.
    if n_channels == 1:
        flattened_image = np.copy(resized_image.reshape((-1, 1)))
    else:
        flattened_image = np.copy(resized_image.reshape((-1, n_channels)))

    # Cluster the pixels.
    get_clusters(flattened_image)

    # Generate the output image by replacing each pixel with the center of its cluster.
    output_image = []
    if n_channels == 1:
        for row in resized_image:
            rows = []
            for col in row:
                center = get_cluster_center([col])
                rows.append(center[0])
            output_image.append(rows)
        # output_image = np.array(output_image, dtype=np.uint8)
    else:
        for row in resized_image:
            rows = []
            for col in row:
                center = get_cluster_center(list(col))
                rows.append(center)
            output_image.append(rows)
    output_image = np.array(output_image, dtype=np.uint8)

    return output_image
