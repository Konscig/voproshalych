# Chatbot

## test

```sh
cd ..
docker compose build chatbot
docker compose down --volumes
docker compose up -d  && docker compose logs -f

docker compose run --rm -it chatbot
```
