import cv2
from typing import Dict, Literal, List
from collections import Counter
from datetime import datetime
from fastapi import FastAPI, Request, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from utils import bytes_to_image, image_to_bytes
from facebank import facebank, image_dataset, classes

from core.detector import PersonFaceDetection
from core.tracker import PersonTracker
from core.recognizer import mbc_ltp_model, sift_akaze_flann

detector = PersonFaceDetection()
CONFIDENCE_THRESHOLD = .5
person_tracker: Dict[int, PersonTracker] = {}

faceRecognizerModel = mbc_ltp_model.FaceRecognizerModel(image_dataset=image_dataset)
rf_model = faceRecognizerModel.randomforest()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(req: Request):
    return templates.TemplateResponse("index.html", {"request": req})


class FrameRequest(BaseModel):
    image: str
    recognizer: Literal["sift_akaze_flann", "mbc_ltp_rf"]

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

            selected_recognizer = frame.recognizer
            if selected_recognizer == "sift_akaze_flann":
                person_names = sift_akaze_flann.face_recognition(image_bgr=person_face_image, FACEBANK=facebank)
            elif selected_recognizer == "mbc_ltp_rf":
                person_names = mbc_ltp_model.face_recognition(image_bgr=person_face_image, MODEL=rf_model, CLASSES=classes)
            person.add_persons(Counter(person_names))

        overlay_frame = cv2.putText(overlay_frame, f"ID: {person_id}, Name: {person.best_person()}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    overlay_frame[:,:,3] = (overlay_frame.max(axis=2) > 0).astype(int) * 255  # ensure conversion suitable
    overlay_frame = image_to_bytes(overlay_frame)
    return FrameResponse(overlay=overlay_frame)


class TrackerLogResponse(BaseModel):
    id: int
    name: str
    last_seen: datetime
    confidence: float
    last_recognized: datetime

@app.get("/tracker_logging", response_model=List[TrackerLogResponse])
def tracker_logging():
    tracker_logs: List[TrackerLogResponse] = []
    for person_id, tracked_person in person_tracker.items():
        best_person_name = tracked_person.best_person()
        tracker_log = TrackerLogResponse(
            id=person_id,
            name=best_person_name,
            last_seen=tracked_person.last_seen,
            confidence=tracked_person.confidence()[best_person_name],
            last_recognized=tracked_person.last_updated,
        )
        tracker_logs.append(tracker_log)
    return tracker_logs



@app.delete("/reset_tracker", response_class=JSONResponse, status_code=status.HTTP_200_OK)
def reset_tracker():
    person_tracker.clear()
    return JSONResponse(
        content={"message": "Person tracker has been reset successfully."},
        status_code=status.HTTP_200_OK
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8000, workers=1)