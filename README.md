## Запуск модуля ИИ
Рекомендуется проводить запуск путем инициализации docker-контейнера. Всё необходимое окружение контейнера готово, необходимо только его собрать и запустить:

```
docker build -t pos-ai .
docker run -d -p 5000:5000 pos-ai
```

## Запуск venv для модуля ИИ
Введите в консоли команду `.venv\Scripts\activate`