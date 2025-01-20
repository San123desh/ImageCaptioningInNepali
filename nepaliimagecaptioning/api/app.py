import io
import sys
import os
from datetime import datetime

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException

from nepaliimagecaptioning.service.image_caption_generator import CaptionGenerator

# Initialize the FastAPI app
app = FastAPI(
    title="Nepali Image Captioning API",
    description="Generate captions for images in Nepali language.",
    version="1.0.0",
)

model = CaptionGenerator()


@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    """
    Endpoint to generate a caption for an uploaded image.
    """
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Generate a caption
    try:
        caption = model.generate_caption(image)
        return {"filename": file.filename, "caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating caption")


# Endpoint for health check
@app.get("/")
async def health_check():
    return {"status": "running", "timestamp": datetime.now()}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
