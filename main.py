import cv2
from typing import Dict
from collections import Counter
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from utils import bytes_to_image, image_to_bytes
from facebank import facebank

from core.detector import PersonFaceDetection
from core.tracker import PersonTracker
from core.recognizer import *

detector = PersonFaceDetection()
CONFIDENCE_THRESHOLD = .5
person_tracker: Dict[int, PersonTracker] = {}

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(req: Request):
    return templates.TemplateResponse("index.html", {"request": req})


class FrameRequest(BaseModel):
    image: str

class FrameResponse(BaseModel):
    overlay: str

@app.post("/process_frame", response_model=FrameResponse)
async def process_frame(frame: FrameRequest) -> FrameResponse:
    image_bgr = bytes_to_image(frame.image)
    
    overlay_frame, tracked_persons = detector.detect_person(image_bgr=image_bgr, CONFIDENCE_THRESHOLD=CONFIDENCE_THRESHOLD)
    for person_id, (x1, y1, x2, y2) in tracked_persons:
        x1, y1, x2, y2 = detector.image_coordinate_padding(x1, y1, x2, y2, max_yx=image_bgr.shape[:2], pad_mod=32)
        person_image = image_bgr[y1:y2, x1:x2]
        if min(person_image.shape) > 0:
            overlay_frame[y1:y2, x1:x2], tracked_faces = detector.detect_face(image_bgr=person_image, frame=overlay_frame[y1:y2, x1:x2])

        person = person_tracker.get(person_id)
        if person is None:
            person = PersonTracker(id=person_id)
            person_tracker[person_id] = person

        # remove True in production
        if tracked_faces and (True or not person.is_converge() or person.is_expired()):
            fx1, fy1, fx2, fy2 = tracked_faces[0]
            person_face_image = person_image[fy1:fy2, fx1:fx2]

            # # SIFT/ORB and FLANN
            # person_names = face_recognition_concat(person_face_image, FACEBANK=facebank)

            # # LBHP
            # person_name, _ = face_recognition_lbph(person_face_image, CLASSES=classes)
            # person_names = [person_name]

            # # HOG + SVM
            # person_name = face_recognition_hogsvm(person_face_image, CLASSES=classes)
            # person_names = [person_name]

            # # KNN
            # person_name = face_recognition_knn(person_face_image, CLASSES=classes)
            # person_names = [person_name]
            
            person_names = ["RECOGNITION IS NOT INSERTED"]
            person.add_persons(Counter(person_names))

        overlay_frame = cv2.putText(overlay_frame, f"ID: {person_id}, Name: {person.best_person()}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    overlay_frame[:,:,3] = (overlay_frame.max(axis=2) > 0).astype(int) * 255  # ensure conversion suitable
    overlay_frame = image_to_bytes(overlay_frame)
    return FrameResponse(overlay=overlay_frame)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8000, workers=1)