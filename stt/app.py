from fastapi import FastAPI, UploadFile, File, HTTPException
from inference import model, transcribe_audio
import os

app = FastAPI()

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):  # принимаем файл напрямую
    try:
        # создаем временный файл для обработки
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())  
        
        # обрабатываем файл
        text = transcribe_audio(model, temp_path)
        print(f"Полученный текст: {text}")
        
        # удаляем временный файл
        os.remove(temp_path)
        
        return {"status": "success", "transcription": text}
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        # удаляем временный файл в случае ошибки (если он был создан)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
async def read_root():
    return {"message": "hello, world!"}
