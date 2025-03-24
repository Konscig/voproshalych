import warnings
import gigaam

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

model = gigaam.load_model(
    "rnnt",  # GigaAM-V2 RNNT model
    device="cpu",  # CPU-inference
)

# print(model.transcribe("example.wav"))

# функция принимает путь к .wav файлу и возвращает расшифрованный текст

def transcribe_audio(model, wav_path: str) -> str:
    return model.transcribe(wav_path)



