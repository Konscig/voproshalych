import warnings
import os
from dotenv import load_dotenv
import gigaam

load_dotenv()

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

hf_token = os.getenv("HF_TOKEN", "")
if not hf_token:
    print("ВНИМАНИЕ: HF_TOKEN не найден в переменных окружения")
else:
    print(f"HF_TOKEN найден: {hf_token[:4]}...{hf_token[-4:]}")

os.environ["HF_TOKEN"] = hf_token

model_type = os.getenv("GIGAAM_MODEL_TYPE", "rnnt")
device = os.getenv("GIGAAM_DEVICE", "cpu")

model = gigaam.load_model(
    model_type,     # Модель GigaAM-V2 RNNT
    device=device,  # Инференс на CPU
)

def transcribe_audio(model, wav_path: str) -> str:
    """
    Транскрибирует аудиофайл с временными метками для длинных записей.

    Аргументы:
        model: Экземпляр модели GigaAM
        wav_path: Путь к аудиофайлу

    Возвращает:
        str: Отформатированная транскрипция с временными метками
    """
    try:
        print(f"Начало транскрибации файла: {wav_path}")
        recognition_result = model.transcribe_longform(wav_path)

        if not recognition_result:
            print("Получен пустой результат транскрибации")
            return ""

        formatted_transcription = []
        for utterance in recognition_result:
            transcription = utterance["transcription"]
            start, end = utterance["boundaries"]
            formatted_transcription.append(
                f"[{gigaam.format_time(start)} - {gigaam.format_time(end)}]: {transcription}"
            )

        result = "\n".join(formatted_transcription)
        print(f"Успешно транскрибировано: {len(formatted_transcription)} фраз")
        return result
    except Exception as e:
        print(f"Ошибка при транскрибации: {str(e)}")
        print(f"Текущий HF_TOKEN: {os.getenv('HF_TOKEN', 'не установлен')}")
        return ""
