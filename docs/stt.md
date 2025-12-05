# Микросервис STT

## Описание
Микросервис, предоставляющий API для преобразования аудиофайлов в текст с использованием модели GigaAM-V2 RNNT.

## [app](../stt/app.py)

### `transcribe(file: UploadFile) -> dict`
Преобразует аудиофайл в текст.

    Args:
        file (UploadFile): аудиофайл для преобразования

    Returns:
        dict: словарь с результатом преобразования
            {
                "status": "success",
                "transcription": "Распознанный текст"
            }

    Raises:
        HTTPException: при ошибках обработки файла

### `read_root() -> dict`
Эндпоинт проверки работоспособности.

    Returns:
        dict: словарь с приветственным сообщением
            {
                "message": "hello, world!"
            }

## [am](../stt/am.py)

### `transcribe_audio(model, wav_path: str) -> str`
Преобразует аудиофайл в текст с использованием модели GigaAM.

    Args:
        model: модель GigaAM
        wav_path (str): путь к аудиофайлу

    Returns:
        str: распознанный текст
