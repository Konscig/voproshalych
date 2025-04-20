## uv

```sh
uv init
uv venv --seed

uv add dotenv ffmpeg-python requests aiogram httpx
uv pip install git+https://github.com/salute-developers/GigaAM.git
```

```sh
# fastAPI
uv init --app

# uv add fastapi --extra standard
# uv pip install "fastapi[standard]"
uv run fastapi dev

# test the app by opening http://127.0.0.1:8000/?token=jessica in a web browser ;
```

```sh
# curl для тестирования эндпойнта /transcribe

curl -X 'POST' \
  'http://127.0.0.1:8000/transcribe/' \
  -H 'accept: application/json' \
  -F 'file=@/file_path'
```

## docker

```sh
docker build -t stt-service:0.1.0 .
docker run --name stt -it --rm -p 8000:8000 stt-service:0.1.0 uv run fastapi dev --host 0.0.0.0
```

