import warnings
import os
from dotenv import load_dotenv
import gigaam

# Load environment variables from .env file
load_dotenv()

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Get model parameters from environment variables with defaults
model_type = os.getenv("GIGAAM_MODEL_TYPE", "rnnt")
device = os.getenv("GIGAAM_DEVICE", "cpu")

model = gigaam.load_model(
    model_type,     # GigaAM-V2 RNNT model
    device=device,  # CPU-inference
)

def transcribe_audio(model, wav_path: str) -> str:
    return model.transcribe(wav_path)
