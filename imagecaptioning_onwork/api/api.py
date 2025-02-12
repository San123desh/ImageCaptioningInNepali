import os

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from imagecaptioning_onwork.service.caption_generator import CaptionGenerator

app = FastAPI()
MODEL_PATH = "./models"
caption_generator = CaptionGenerator(model_dir=MODEL_PATH)


@app.get("/")
def read_root():
    return {"message": "App is running!"}


@app.post("/caption/generate")
async def generate_caption(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    file_location = "temp_image.jpg"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    caption = caption_generator.generate_caption(file_location)
    os.remove(file_location)
    return JSONResponse(content={"caption": caption})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
