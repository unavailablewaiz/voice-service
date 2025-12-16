from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import tempfile, os

app = FastAPI()

asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=-1
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(await file.read())
        path = f.name

    text = asr(path)["text"]
    os.remove(path)

    return {"text": text}
