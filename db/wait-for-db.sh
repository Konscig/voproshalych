#!/bin/sh
# Wait for database to be healthy and then run migrations
echo "Waiting for database to be healthy..."

while ! nc -z voproshalych_chatbot-conn 5432; do
    sleep 1
done

echo "Database is healthy, running migrations..."
exec "$@"
