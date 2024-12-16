import os
import cv2
from typing import Dict

from core.utils import resize_image, Person

FACEBANK_PATH = "facebank/"  # base project

facebank: Dict[str, Person] = {}
for name in os.listdir(FACEBANK_PATH):
    namepath = os.path.join(FACEBANK_PATH, name)
    if os.path.isdir(namepath):
        for imgfn in os.listdir(namepath):
            _, ext = os.path.splitext(imgfn)
            if ext not in ('.jpeg', '.png', '.jpg'):
                continue

            imgpath = os.path.join(namepath, imgfn)
            img = cv2.imread(imgpath, None)
            img = resize_image(img, target_height=512)
            if name not in facebank:
                facebank[name] = Person(name=name, images=[], face_images=[], local_descriptors=[])

            facebank[name]['images'].append(img)

            # img = preprocess_image(img)
            # keypoints, descriptor = local_descriptor_detect(img)
            # facebank[name]['local_descriptors'].append((keypoints, descriptor))