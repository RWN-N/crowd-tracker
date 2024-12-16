import cv2
import numpy as np
import torch
from typing import NamedTuple, List, Optional, Tuple
from ultralytics import YOLO

from libs.sort.sort import Sort
from libs.yoloface.face_detector import YoloDetector
from core.utils import Coordinate, TrackedCoordinate


class DetectPersonResult(NamedTuple):
    frame: np.ndarray
    tracked: List[TrackedCoordinate]

class DetectFaceResult(NamedTuple):
    frame: np.ndarray
    coordinates: List[Coordinate]


class PersonFaceDetection:
    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        self.model = YOLO('yolov5su.pt', verbose=False)
        self.face_model = YoloDetector(target_size=720, device=f"{'cuda' if torch.cuda.is_available() else 'cpu'}:0", min_face=90)
        self.person_tracker = Sort(max_age=5, min_hits=3, iou_threshold=.3)

    def detect_person(self, image_bgr: np.ndarray, *, frame: Optional[np.ndarray] = None, CONFIDENCE_THRESHOLD: float = .5) -> DetectPersonResult:
        if frame is None:
            frame = np.zeros([*image_bgr.shape[:2], 4], dtype=np.uint8)

        results = self.model.predict(image_bgr, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []  # Extract YOLO detections into [x1, y1, x2, y2, confidence]
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                label = self.model.names[int(cls)]
                if label != "person":
                    continue
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2, conf.item()])

        detections = np.array(detections)
        tracked_persons = self.person_tracker.update(detections)
        tracked_coordinates = []
        for person in tracked_persons:
            x1, y1, x2, y2, person_id = map(int, person)
            tracked_coordinates.append((person_id, (x1, y1, x2, y2)))
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        return frame, tracked_coordinates


    def detect_face(self, image_bgr: np.ndarray, *, frame: Optional[np.ndarray] = None, CONFIDENCE_THRESHOLD: float = .5) -> DetectFaceResult:
        if frame is None:
            frame = np.zeros([*image_bgr.shape[:2], 4], dtype=np.uint8)

        bboxes, _points = self.face_model.predict(image_bgr, conf_thres=CONFIDENCE_THRESHOLD)
        bboxes, _points = bboxes[0], _points[0]
        for x1, y1, x2, y2 in bboxes:
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (122, 202, 0), 3)

        return frame, bboxes
    
    @staticmethod
    def image_coordinate_padding(x1: int, y1: int, x2: int, y2: int, max_yx: Tuple[int, int], pad_mod: int = 32) -> Coordinate:
        x1, y1, x2, y2 = (max(i, 0) for i in (x1, y1, x2, y2))  # coords for image shouldn't go negative

        x_diff, y_diff = x2 - x1, y2 - y1
        x_mod, y_mod = x_diff % pad_mod, y_diff % pad_mod
        x2 += pad_mod - x_mod
        y2 += pad_mod - y_mod

        max_y, max_x = max_yx
        if x2 > max_x:
            x1 -= x2 - max_x
            x2 = max_x
        if y2 > max_y:
            y1 -= y2 - max_y
            y2 = max_y

        x1, y1, x2, y2 = (max(i, 0) for i in (x1, y1, x2, y2))  # make sure everything don't go negative, and if both 1st 2nd coors max out, the max_yx still obey the pad_mod
        return x1, y1, x2, y2 