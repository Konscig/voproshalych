```sh
uv init    
uv venv --seed  

uv pip install dotenv ffmpeg-python requests aiogram
uv pip install git+https://github.com/salute-developers/GigaAM.git
```

```sh
# fastAPI
# this creates an project with an application layout and a pyproject.toml file
uv init --app

uv add fastapi --extra standard  
uv pip install "fastapi[standard]"
uv run fastapi dev

# uv run will automatically resolve and lock the project dependencies 
# (i.e., create a uv.lock alongside the pyproject.toml), create a virtual environment, 
# and run the command in that environment ;

# test the app by opening http://127.0.0.1:8000/?token=jessica in a web browser ;
```

```sh
# curl для тестирования эндпойнта /transcribe

curl -X 'POST' \
  'http://127.0.0.1:8000/transcribe/' \
  -H 'accept: application/json' \
  -F 'file=@/Users/masha/src/kitty/utmn/virtassist/voicehandler/example.wav'
```

## для тестирования :

```sh
uv run fastapi dev 
uv run telegram-bot.py
```

- отправить телеграм боту голосовое сообщение до 30 секунд ;
- в чат с ботом и консоль должна вернуться транскрипция голосового сообщения ;
  