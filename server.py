"""DR-Defer web server. Run:  python server.py   then open http://127.0.0.1:8000

Images are processed in memory and never written to disk.
Locally it binds to localhost only. When the PORT environment variable is set
(hosting service) it listens on all interfaces.
"""
import io
import os
import threading
import webbrowser

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

import engine

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_BYTES = 20 * 1024 * 1024

app = FastAPI(title="DR-Defer", docs_url=None, redoc_url=None)


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 20 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file is not a readable image.")
    return engine.analyse(img)


if __name__ == "__main__":
    # On a hosting service (PORT is set) listen on all interfaces; locally stay on localhost.
    hosted = bool(os.environ.get("PORT"))
    host = "0.0.0.0" if hosted else "127.0.0.1"
    port = int(os.environ.get("PORT", 8000))
    if not hosted:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
