import os
import gigaam
from dotenv import load_dotenv

load_dotenv()

os.environ["HUGGINGFACE_HUB_CACHE"] = "/huggingface_cache"
os.environ["HF_HOME"] = "/huggingface_cache"

hf_token = os.getenv("HF_TOKEN", "")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

model_type = os.getenv("GIGAAM_MODEL_TYPE", "rnnt")
device = "cpu" # always use cpu for downloading

print(f"Downloading model {model_type} to {os.environ['HUGGINGFACE_HUB_CACHE']}...")

gigaam.load_model(
    model_type,
    device=device,
)

print("Model downloaded.")
