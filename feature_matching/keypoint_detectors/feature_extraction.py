"""Feature Extraction using SIFT and Visual Vocabulary.

Functions:
    - detect_sift_keypoints_descriptors
    - custom_match_descriptors
    - plot_keypoint_matches
    - visualize_data
    - extract_sift_features
    - build_visual_vocabulary
    - build_histograms
    - adjust_frequency_vector
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import ConnectionPatch
from PIL import Image
from scipy.sparse._matrix import spmatrix
from scipy.spatial.distance import cdist
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import SIFT
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm


__all__ = [
    "adjust_frequency_vector",
    "build_histograms",
    "build_visual_vocabulary",
    "custom_match_descriptors",
    "detect_sift_keypoints_descriptors",
    "extract_sift_features",
    "plot_keypoint_matches",
    "visualize_data",
]
__author__ = "Gavin Meyer"


def detect_sift_keypoints_descriptors(
    dst_img: np.ndarray, src_img: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect SIFT keypoints and descriptors from two images.

    Args:
        dst_img: The destination image.
        src_img: The source image.

    Returns:
        keypoints1: Keypoints from the destination image.
        descriptors1: Descriptors from the destination image.
        keypoints2: Keypoints from the source image.
        descriptors2: Descriptors from the source image.
    """
    detector1 = SIFT()
    detector2 = SIFT()
    detector1.detect_and_extract(dst_img)
    detector2.detect_and_extract(src_img)
    keypoints1 = detector1.keypoints
    descriptors1 = detector1.descriptors
    keypoints2 = detector2.keypoints
    descriptors2 = detector2.descriptors

    return keypoints1, descriptors1, keypoints2, descriptors2


def custom_match_descriptors(
    descriptors1: np.ndarray, descriptors2: np.ndarray, cross_check: bool = True
) -> np.ndarray:
    """Match descriptors between two images.

    Args:
        descriptors1: Descriptors from the destination image.
        descriptors2: Descriptors from the source image.
        cross_check: Whether to use cross-checking to find mutual matches. (default: True)

    Returns:
        matches: The indices of the matched descriptors.
    """
    # Compute the Euclidean distance between each pair of descriptors
    distances = cdist(descriptors1, descriptors2, "euclidean")

    # Find the nearest neighbor for each descriptor in descriptors1
    nearest_neighbor_12 = np.argmin(distances, axis=1)

    # For cross-checking, find the nearest neighbor for each descriptor in descriptors2
    if cross_check:
        nearest_neighbor_21 = np.argmin(distances, axis=0)

        mutual_matches = [
            i for i, j in enumerate(nearest_neighbor_12) if nearest_neighbor_21[j] == i
        ]
        return np.array([[i, j] for i, j in enumerate(nearest_neighbor_12) if i in mutual_matches])
    return np.column_stack([np.arange(descriptors1.shape[0]), nearest_neighbor_12])


def plot_keypoint_matches(
    dst_img: np.ndarray,
    keypoints1: np.ndarray,
    src_img: np.ndarray,
    keypoints2: np.ndarray,
    matches: np.ndarray,
) -> None:
    """Plot the matched keypoints between two images.

    Args:
        dst_img: The destination image.
        keypoints1: Keypoints from the destination image.
        src_img: The source image.
        keypoints2: Keypoints from the source image.
        matches: The indices of the matched keypoints.
    """
    # Allocate matching keypoints
    dst = keypoints1[matches[:, 0]]
    src = keypoints2[matches[:, 1]]

    # Plot the matched keypoints
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(dst_img, cmap="gray")
    ax2.imshow(src_img, cmap="gray")

    for i in range(src.shape[0]):
        coordB = (dst[i, 1], dst[i, 0])  # Coordinates in the destination image
        coordA = (src[i, 1], src[i, 0])  # Coordinates in the source image
        con = ConnectionPatch(
            xyA=coordA,
            xyB=coordB,
            coordsA="data",
            coordsB="data",
            axesA=ax2,
            axesB=ax1,
            color="red",
        )
        ax2.add_artist(con)
        ax1.plot(dst[i, 1], dst[i, 0], "ro")
        ax2.plot(src[i, 1], src[i, 0], "ro")

    plt.show()


def visualize_data(data: np.ndarray) -> None:
    """Visualize the first 10 images in the dataset.

    Args:
        data: The input images.
    """
    _, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes):
        ax.imshow(data[i], cmap="gray")
        ax.axis("off")
    plt.show()


def extract_sift_features(X_data: np.ndarray, y_data: np.ndarray) -> tuple[list, list]:
    """Extract SIFT features from the images.

    Args:
        X_data: The input images.
        y_data: The image labels.

    Returns:
        descriptors_list: A list of SIFT descriptors for each image.
        y_features: The image labels.
    """
    sift = SIFT()
    descriptors_list = []
    y_features = []

    for img in tqdm(range(X_data.shape[0]), desc="Processing images"):
        try:
            sift.detect_and_extract(X_data[img])
            descriptors_list.append(sift.descriptors)
            y_features.append(y_data[img])
        except:
            pass

    return descriptors_list, y_features


def build_visual_vocabulary(descriptor_list: list, vocab_size: int) -> KMeans:
    """Build a visual vocabulary using KMeans clustering.

    Args:
        descriptor_list: A list of SIFT descriptors for each image.
        vocab_size: The number of visual words to use.

    Returns:
        kmeans: The KMeans model trained on the descriptors.
    """
    descriptor_np = np.concatenate(descriptor_list)
    kmeans = KMeans(n_clusters=vocab_size, random_state=42)
    kmeans.fit(descriptor_np)

    return kmeans


def build_histograms(descriptor_list: list, kmeans: KMeans, vocab_size: int) -> np.ndarray:
    """Build histograms of visual words for each image.

    Args:
        descriptor_list: A list of SIFT descriptors for each image.
        kmeans: The KMeans model trained on the descriptors.
        vocab_size: The number of visual words to use.

    Returns:
        histograms: An array of histograms of visual words for each image.
    """
    histograms = []

    for descriptors in tqdm(descriptor_list, desc="Building histograms"):
        # Predict the closest cluster for each feature
        clusters = kmeans.predict(descriptors)
        # Build a histogram of the clusters
        histogram, _ = np.histogram(clusters, bins=vocab_size, range=(0, vocab_size))
        histograms.append(histogram)

    return np.array(histograms)


def adjust_frequency_vector(histograms: np.ndarray) -> spmatrix:
    """Adjust the frequency of visual words using TF-IDF.

    Args:
        histograms: An array of histograms of visual words for each image.

    Returns:
        The histogram data transformed using TF-IDF.
    """
    # Create a TfidfTransformer
    tfidf = TfidfTransformer()

    # Fit the TfidfTransformer to the histogram data
    tfidf.fit(histograms)

    # Transform the histogram data using the trained TfidfTransformer
    return tfidf.transform(histograms)


def main() -> None:
    """Main function to demonstrate the feature extraction process."""
    # Load the images
    dst_img_rgb = np.asarray(Image.open("./data/Rainier1.png"))
    src_img_rgb = np.asarray(Image.open("./data/Rainier2.png"))

    # Convert RGBA to RGB if necessary
    if dst_img_rgb.shape[2] == 4:
        dst_img_rgb = rgba2rgb(dst_img_rgb)
    if src_img_rgb.shape[2] == 4:
        src_img_rgb = rgba2rgb(src_img_rgb)

    # Convert images to grayscale
    dst_img = rgb2gray(dst_img_rgb)
    src_img = rgb2gray(src_img_rgb)

    # Detect SIFT keypoints and descriptors
    print("Detecting SIFT keypoints and descriptors...")
    keypoints1, descriptors1, keypoints2, descriptors2 = detect_sift_keypoints_descriptors(
        dst_img, src_img
    )

    print("Matching descriptors between images...")
    matches = custom_match_descriptors(descriptors1, descriptors2, cross_check=True)

    print(f"Number of matches: {len(matches)}")
    print("Plotting the matched keypoints...")

    plot_keypoint_matches(dst_img, keypoints1, src_img, keypoints2, matches)

    print("Beginning feature extraction process...")

    # Load the split data
    data = np.load("./cifar10.npz", allow_pickle=True)

    # Allocate the training and testing data
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Convert the input data to a numpy array
    X_train_rgb = np.array(X_train, dtype="uint8")
    X_test_rgb = np.array(X_test, dtype="uint8")

    # Reshape the data to (num_images, height, width, num_channels)
    X_train_rgb = X_train_rgb.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test_rgb = X_test_rgb.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Convert the images to grayscale
    X_train_gray = rgb2gray(X_train_rgb)
    X_test_gray = rgb2gray(X_test_rgb)

    print("Visualizing the first 10 images of training set...")
    visualize_data(X_train_gray)
    print("Visualizing the first 10 images of test set...")
    visualize_data(X_test_gray)

    print("Extracting SIFT features from the training data...")
    X_train_descriptors, y_train_features = extract_sift_features(X_train_gray, y_train)
    print(f"Number of training features: {len(X_train_descriptors)}")

    print("Extracting SIFT features from the testing data...")
    X_test_descriptors, y_test_features = extract_sift_features(X_test_gray, y_test)
    print(f"Number of testing features: {len(X_test_descriptors)}")

    vocab_size = 50
    print(f"Building visual vocabulary with {vocab_size} words for training set...")
    X_train_vocabulary = build_visual_vocabulary(X_train_descriptors, vocab_size)
    print(f"Building visual vocabulary with {vocab_size} words for testing set...")
    X_test_vocabulary = build_visual_vocabulary(X_test_descriptors, vocab_size)

    print("Building histograms for training set...")
    X_train_hist = build_histograms(X_train_descriptors, X_train_vocabulary, vocab_size)
    print("Building histograms for testing set...")
    X_test_hist = build_histograms(X_test_descriptors, X_test_vocabulary, vocab_size)

    print("Adjusting frequency using TF-IDF for each set...")
    X_train_hist_tfidf = adjust_frequency_vector(X_train_hist)
    X_test_hist_tfidf = adjust_frequency_vector(X_test_hist)

    print("Saving processed data as 'processed_cifar10_sift.npz'...")
    sift_processed_data = {
        "X_train": X_train_hist_tfidf.toarray(),
        "X_test": X_test_hist_tfidf.toarray(),
        "y_train": y_train_features,
        "y_test": y_test_features,
    }
    np.savez("processed_cifar10_sift.npz", **sift_processed_data)

    print("Feature extraction complete!")
    print("Evaluate the processed data using 'evaluate_sift.py'.")


if __name__ == "__main__":
    main()
