import os
import gigaam
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

print("---- Загрузка моделей ----")

# --- 1. Загрузка embedding-модели (логика наставника) ---
print("Шаг 1/2: Скачивание embedding-модели nizamovtimur/multilingual-e5-large-wikiutmn...")
# Скачиваем модель с mirror
save_path = snapshot_download(
    repo_id="nizamovtimur/multilingual-e5-large-wikiutmn",
    local_dir="saved_models/multilingual-e5-large-wikiutmn",
    endpoint="https://hf-mirror.com",
)
# Загружаем модель из локальной директории
model = SentenceTransformer(save_path)
print("Embedding-модель успешно скачана и инициализирована.")


# --- 2. Загрузка речевой модели (gigaam) ---
print("\nШаг 2/2: Скачивание речевой модели gigaam...")
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("ВНИМАНИЕ: HF_TOKEN не найден. Скачивание gigaam может не удасться.")
else:
    os.environ["HF_TOKEN"] = hf_token

gigaam_model_type = os.getenv("GIGAAM_MODEL_TYPE", "rnnt")
gigaam.load_model(
    gigaam_model_type,
    device=os.getenv("GIGAAM_DEVICE", "cpu"),
)
print("Речевая модель gigaam успешно скачана и кэширована.")
print("\n---- Все модели успешно загружены. ----")