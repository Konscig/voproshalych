# Микросервис Chatbot

## Инициализация базы данных

### Остановка и перезапуск контейнеров
```sh
cd ..
docker compose down --volumes
docker compose up -d  && docker compose logs -f
```

### Подготовка базы данных
```sh
docker compose up -d db
docker exec -i virtassist-db psql -U postgres -c "DROP DATABASE IF EXISTS virtassist;"
docker exec -i virtassist-db psql -U postgres -c "CREATE DATABASE virtassist;"
docker exec -i virtassist-db psql -U postgres -d virtassist < tmp/virtassist.dump
```

### Подключение к базе данных
```sh
docker exec -it virtassist-db psql -U postgres -d virtassist
```

## Тестирование
```sh
docker compose build chatbot && docker compose run --rm -it chatbot
```
