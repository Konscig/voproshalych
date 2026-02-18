from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

# Скачиваем модель с mirror
save_path = snapshot_download(
    repo_id="nizamovtimur/multilingual-e5-large-wikiutmn",
    local_dir="saved_models/multilingual-e5-large-wikiutmn",
    endpoint="https://hf-mirror.com",
)

# Загружаем модель из локальной директории
model = SentenceTransformer(save_path)
