# Chatbot

## bootstrap DB

```sh
cd ..
docker compose down --volumes
# docker compose up -d  && docker compose logs -f

docker compose up -d db # --no-deps
docker exec -i virtassist-db psql -U postgres -c "DROP DATABASE IF EXISTS virtassist;"
docker exec -i virtassist-db psql -U postgres -c "CREATE DATABASE virtassist;"
docker exec -i virtassist-db psql -U postgres -d virtassist < tmp/virtassist.dump

docker exec -it virtassist-db psql -U postgres -d virtassist
```

## test

```sh
docker compose build chatbot && docker compose run --rm -it chatbot
```
