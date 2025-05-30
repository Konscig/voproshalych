from fastapi import FastAPI, UploadFile, File, HTTPException
from am import model, transcribe_audio
import os

app = FastAPI()

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        text = transcribe_audio(model, temp_path)
        print(f"Полученный текст: {text}")

        os.remove(temp_path)

        return {"status": "success", "transcription": text}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "hello, world!"}
