import cv2
import numpy as np
from typing import Dict, List
from core.utils import resize_image, LocalDescriptorResult, Person


### Preprocess Image
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def _gamma_estimation(image_bgr: np.ndarray):  # 
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, _, val = cv2.split(hsv)  # split into hue, sat, val [only use val]
    mid = .5
    mean = np.mean(val)
    gamma = np.log(mid * 255) / np.log(mean)
    return gamma

def _adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_image(image_bgr: np.ndarray, target_height: int = 256) -> np.ndarray:
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
    
    image_gray = clahe.apply(image_gray)

    approx_gamma = _gamma_estimation(image_bgr=image_bgr)
    image_gray = _adjust_gamma(image_gray, gamma=approx_gamma)
    
    image_gray = resize_image(image_gray, target_height)
    return image_gray


### Local Descriptor
sift = cv2.SIFT_create()
def sift_descriptor_detect(image: np.ndarray) -> LocalDescriptorResult:
    keypoints, descriptors = sift.detectAndCompute(image, None)
    descriptors = np.float32(descriptors)
    return keypoints, descriptors

akaze = cv2.AKAZE_create()
def akaze_descriptor_detect(image: np.ndarray) -> LocalDescriptorResult:
    keypoints, descriptors = akaze.detectAndCompute(image, None)
    descriptors = np.float32(descriptors)
    return keypoints, descriptors


### Recognizer
flann_sift = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=3), 
    dict(checks=5)
)
flann_akaze = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=3),
    dict(checks=5)
)

def face_recognition(image_bgr: np.ndarray, FACEBANK: Dict[str, Person], *, MIN_MATCHES: int = 3) -> List[str]:
    bestScore = 0
    bestPerson = []

    resized_image_bgr = resize_image(image_bgr, target_height=256)
    image_processed = preprocess_image(resized_image_bgr)

    _, sift_desc = sift_descriptor_detect(image_processed)
    _, akaze_desc = akaze_descriptor_detect(image_processed)

    for name, person in FACEBANK.items():
        person_desc = person['descriptors']
        all_sift_desc = np.vstack([desc for _, desc in person_desc['sift']])
        all_akaze_desc = np.vstack([desc for _, desc in person_desc['akaze']])

        siftClusterMatches = flann_sift.knnMatch(sift_desc, all_sift_desc, 2)
        akazeClusterMatches = flann_akaze.knnMatch(akaze_desc,  all_akaze_desc, 2)

        currSiftMatches = [(fm, sm) for fm, sm in siftClusterMatches if fm.distance < 0.7 * sm.distance]
        currAkazeMatches = [(fm, sm) for fm, sm in akazeClusterMatches if fm.distance < 0.7 * sm.distance]

        sift_score = len(currSiftMatches)
        akaze_score = len(currAkazeMatches)

        sift_weight = 0.6
        akaze_weight = 0.4

        currScore = (
            sift_weight * sift_score +
            akaze_weight * akaze_score
        )

        if currScore < MIN_MATCHES:
            continue

        if bestScore <= currScore:
            bestScore = currScore
            if bestScore != currScore:
                bestPerson = []
            bestPerson.append(name)

    if not bestPerson:
        bestPerson = ["Unknown"]
    return bestPerson