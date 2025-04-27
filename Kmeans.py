import numpy as np
import matplotlib.pyplot as plt
import cv2 

def initialize_centroids(X, k):
    """Randomly initialize k centroids """
    random_indices = np.random.choice(len(X), size=k, replace=False)
    centroids = X[random_indices]
    return centroids

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    cluster_labels = np.argmin(distances, axis=1) # assign to nearest centroid
    return cluster_labels

def update_centroids(X, cluster_labels, k):
    new_centroids = []
    for i in range(k):
        points = X[cluster_labels == i]
        if len(points) == 0:
            # a random point to avoid empty cluster
            new_centroids.append(X[np.random.choice(len(X))])
        else:
            new_centroids.append(points.mean(axis=0))
    return np.array(new_centroids)

def kmeans(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        old_centroids = centroids.copy()
        
        cluster_labels = assign_clusters(X, centroids)
        centroids = update_centroids(X, cluster_labels, k)
        
        # Check for convergence
        if np.allclose(centroids, old_centroids):
            break
                
    return centroids, cluster_labels


def kmeans_segment_image(image, k=3, max_iters=100):
    w, h, d = image.shape
    print(f"Image loaded: {w}x{h} pixels, {d} channels")

    # RGB img--> reshape to 2D array where each row is a pixel and each column is a color channel
    image_array = np.reshape(image, (w * h, d))
    image_array = image_array.astype(float) / 255.0  # Normalize

    # Apply k-means
    centroids, labels = kmeans(image_array, k, max_iters=max_iters)

    # full_labels after assigning clusters
    full_labels = assign_clusters(image_array, centroids)

    segmented_img = centroids[full_labels] # replace each pixel with its centroid
    segmented_img = segmented_img.reshape(w, h, d)

    segmented_img = (segmented_img * 255).astype(np.uint8)
    return segmented_img

if __name__ == "__main__":
    segmented = kmeans_segment_image('data/orange.jpg', k=5)
    
    plt.imshow(segmented)
    plt.axis('off')
    plt.title("K-means Image Segmentation")
    plt.show()
