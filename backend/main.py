import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

from config import UPLOAD_FOLDER, OUTPUT_FOLDER
from encoder import Encoder
from decoder import decode_image

app = FastAPI(title="Robust Digital Watermarking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "Running",
        "project": "Robust Watermarking System"
    }


# --------------------------------------------------
# Encode
# --------------------------------------------------

@app.post("/encode")
async def encode(
    image: UploadFile = File(...),
    secret_message: str = Form(...)
):
    extension = Path(image.filename).suffix

    unique_name = f"{uuid.uuid4()}{extension}"

    upload_path = UPLOAD_FOLDER / unique_name

    with open(upload_path, "wb") as buffer:
        buffer.write(await image.read())

    output_name = f"encoded_{unique_name}"

    output_path = OUTPUT_FOLDER / output_name

    encoder = Encoder()

    result = encoder.encode(
        str(upload_path),
        secret_message,
        str(output_path)
    )

    return {
        "success": True,
        "download": f"/download/{output_name}",
        "details": result
    }


# --------------------------------------------------
# Decode
# --------------------------------------------------

@app.post("/decode")
async def decode(
    image: UploadFile = File(...),
    total_bits: int = Form(...)
):
    extension = Path(image.filename).suffix

    unique_name = f"{uuid.uuid4()}{extension}"

    upload_path = UPLOAD_FOLDER / unique_name

    with open(upload_path, "wb") as buffer:
        buffer.write(await image.read())

    try:
        result = decode_image(
            str(upload_path),
            total_bits
        )

        return result
    except Exception as e:
        traceback.print_exc()

        return JSONResponse(
            status_code=400,
            content={
                "error": str(e),
                "type": type(e).__name__
            }
        )

# --------------------------------------------------
# Download
# --------------------------------------------------

@app.get("/download/{filename}")
def download(filename: str):

    path = OUTPUT_FOLDER / filename

    return FileResponse(
        path,
        media_type="image/png",
        filename=filename
    )