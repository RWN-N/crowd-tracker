import os
import random
import cv2
from typing import Dict

from core.recognizer import sift_akaze_flann
from core.recognizer.sift_akaze_flann import sift_descriptor_detect, akaze_descriptor_detect
from core.utils import resize_image, Person

FACEBANK_PATH = "facebank/"  # base project

facebank: Dict[str, Person] = {}
for name in os.listdir(FACEBANK_PATH):
    if name.startswith("_"):
        continue
    namepath = os.path.join(FACEBANK_PATH, name)
    if os.path.isdir(namepath):
        for imgfn in os.listdir(namepath):
            _, ext = os.path.splitext(imgfn)
            if ext not in ('.jpeg', '.png', '.jpg'):
                continue

            imgpath = os.path.join(namepath, imgfn)
            img = cv2.imread(imgpath, None)
            img = resize_image(img, target_height=256)
            if name not in facebank:
                facebank[name] = Person(name=name, images=[], descriptors={})

            facebank[name]['images'].append(img)

            curr_desc = facebank[name]['descriptors']
            # SIFT + AKAZE
            if 'sift' not in curr_desc:
                curr_desc['sift'] = []
            if 'akaze' not in curr_desc:
                curr_desc['akaze'] = []

            sift_akaze_img = sift_akaze_flann.preprocess_image(image_bgr=img)
            sift_key, sift_desc = sift_descriptor_detect(sift_akaze_img)
            akaze_key, akaze_desc = akaze_descriptor_detect(sift_akaze_img)
            curr_desc['sift'].append((sift_key, sift_desc))
            curr_desc['akaze'].append((akaze_key, akaze_desc))


classes = []
image_dataset = []
for n, (name, person) in enumerate(facebank.items()):
    classes.append(name)
    for image in person['images']:
        image_dataset.append((image, n))

random.shuffle(image_dataset)