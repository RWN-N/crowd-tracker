import numpy as np
import cv2
from typing import NamedTuple, TypedDict, List

class Coordinate(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int

class TrackedCoordinate(NamedTuple):
    id: int
    coordinate: Coordinate

class Person(TypedDict):
    name: str
    images: List[np.ndarray]


def resize_image(image: np.ndarray, target_height: int = 256) -> np.ndarray:
    h, w = image.shape[:2]
    scale = target_height / h
    new_width = int(w * scale)
    resized_image = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_image