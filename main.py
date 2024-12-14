import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from utils import bytes_to_image, image_to_bytes

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
    
    overlay = np.zeros([*image_bgr.shape[:2], 4], dtype=np.uint8)
    overlay = cv2.putText(overlay, "Hello World", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    overlay[:,:,3] = (overlay.max(axis=2) > 0).astype(int) * 255  # ensure conversion suitable
    overlay = image_to_bytes(overlay)
    return FrameResponse(overlay=overlay)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8000, workers=1)