import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import gaussian_filter, sobel
from typing import List, Tuple

from core.utils import resize_image

### Preprocess Image
def preprocess_image(image_bgr: np.ndarray, target_height: int = 256) -> np.ndarray:
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_gray = resize_image(image=image_gray, target_height=target_height)
    # image_gray = cv2.resize(image_gray, (target_height, target_height))
    return image_gray



### Local Descriptor
def local_ternary_pattern(image, radius=1, neighbors=8, threshold=5):
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale.")

    height, width = image.shape
    ltp_image = np.zeros((height, width), dtype=np.int8)

    y, x = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    circular_mask = x**2 + y**2 <= radius**2

    samples = np.column_stack(np.where(circular_mask))
    center = np.array([radius, radius])
    samples = samples - center
    for row in range(radius, height - radius):
        for col in range(radius, width - radius):
            center_value = image[row, col]
            values = []
            for dy, dx in samples:
                neighbor_value = image[row + dy, col + dx]
                diff = neighbor_value.astype(np.float64) - center_value
                if diff > threshold:
                    values.append(1)
                elif diff < -threshold:
                    values.append(-1)
                else:
                    values.append(0)

            ltp_image[row, col] = sum((3**i) * val for i, val in enumerate(values))

    return ltp_image


def compute_ltp_histogram(image, radius=1, neighbors=8, threshold=5, grid_size=(8, 8)):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    grid_h, grid_w = h // grid_size[0], w // grid_size[1]
    feature_vector = []

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            cell = image[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
            ltp_cell = local_ternary_pattern(cell, radius, neighbors, threshold)
            hist, _ = np.histogram(ltp_cell.ravel(), bins=np.arange(-1, 3), range=(-1, 1))
            feature_vector.extend(hist)

    return np.array(feature_vector)


def monogenic_binary_coding(image, scales=[1, 2, 4]):
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale.")

    mbc_features = []
    for scale in scales:
        smoothed = gaussian_filter(image, sigma=scale)

        gx = sobel(smoothed, axis=1)
        gy = sobel(smoothed, axis=0)

        magnitude = np.sqrt(gx**2 + gy**2)
        phase = np.arctan2(gy, gx)
        phase_binary = (phase > 0).astype(np.uint8)

        combined_mbc = magnitude * phase_binary
        mbc_features.append(combined_mbc)

    return np.dstack(mbc_features)


def compute_mbc_histogram(image, grid_size=(8, 8), scales=[1, 2, 4]):
    mbc_features = monogenic_binary_coding(image, scales)
    h, w = image.shape
    grid_h, grid_w = h // grid_size[0], w // grid_size[1]
    mbc_histogram = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            cell = mbc_features[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
            hist, _ = np.histogram(cell.ravel(), bins=16, range=(0, 256))  # Adjust bins and range as needed
            mbc_histogram.extend(hist)
    return mbc_histogram



# Fusion
def extract_hist_features(image, grid_size=(8, 8), radius=1, neighbors=8, threshold=5, scales=[1, 2, 4]):
    ltp_histogram = compute_ltp_histogram(image, radius, neighbors, threshold, grid_size)
    mbc_histogram = compute_mbc_histogram(image, grid_size=grid_size, scales=scales)
    combined_features = np.concatenate([ltp_histogram, mbc_histogram])
    return combined_features


### Recognizer
class FaceRecognizerModel:
    def __init__(self, image_dataset: List[Tuple[np.ndarray, int]]):
        self.X, self.y = [], []
        for image, y in image_dataset:
            image_processed = preprocess_image(image)
            self.X.append(extract_hist_features(image=image_processed))
            self.y.append(y)
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def svm(self, kernel='linear', C=1.0):
        svm = SVC(kernel=kernel, C=C)
        svm.fit(self.X, self.y)
        return svm

    def randomforest(self, n_estimators=100, max_depth=10, random_state=42):
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        rf.fit(self.X, self.y)
        return rf
    
    def knn(self, n_neighbors=3):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(self.X, self.y)
        return knn

def face_recognition(image_bgr: np.ndarray, MODEL, CLASSES) -> List[str]:
    image_processed = preprocess_image(image_bgr=image_bgr)
    hist_feature = extract_hist_features(image=image_processed)
    pred = MODEL.predict(hist_feature.reshape(1, -1))[0]
    return [CLASSES[pred]]
