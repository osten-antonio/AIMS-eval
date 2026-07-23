from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import VisionEncoderDecoderModel, AutoProcessor
from pix2tex.cli import LatexOCR
import torch
from PIL import Image
import base64
import traceback
from io import BytesIO


MODELS = {
    "nougat-base": {
        "repo": "facebook/nougat-base",
        "model": None,
        "processor": None,
    },
    "nougat-latex-base": {
        "repo": "Norm/nougat-latex-base",
        "model": None,
        "processor": None,
    },
}

pix2tex_model = None


class InferenceRequest(BaseModel):
    image: str  # base64 encoded image


def load_model(name: str):
    print(f"Loading {MODELS[name]['repo']}...")
    processor = AutoProcessor.from_pretrained(MODELS[name]["repo"])
    model = VisionEncoderDecoderModel.from_pretrained(MODELS[name]["repo"])
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    MODELS[name]["model"] = model
    MODELS[name]["processor"] = processor
    print(f"{MODELS[name]['repo']} loaded.")


@asynccontextmanager
async def lifespan(app):
    global pix2tex_model
    for name in MODELS:
        load_model(name)
    print("Loading pix2tex...")
    pix2tex_model = LatexOCR()
    print("pix2tex loaded.")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/nougat-base/inference")
async def nougat_base_inference(request: InferenceRequest):
    return await run_inference("nougat-base", request.image)


@app.post("/nougat-latex-base/inference")
async def nougat_latex_base_inference(request: InferenceRequest):
    return await run_inference("nougat-latex-base", request.image)


@app.post("/pix2tex/inference")
async def pix2tex_inference(request: InferenceRequest):
    try:
        image_bytes = base64.b64decode(request.image)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        output = pix2tex_model(image)
        return {"model": "pix2tex", "output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def run_inference(model_name: str, image_b64: str):
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        processor = MODELS[model_name]["processor"]
        model = MODELS[model_name]["model"]

        pixel_values = processor(images=image, return_tensors="pt", do_crop_margin=False).pixel_values
        if torch.cuda.is_available():
            pixel_values = pixel_values.to("cuda")

        with torch.no_grad():
            outputs = model.generate(pixel_values, max_length=2048)

        text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return {"model": model_name, "output": text}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
