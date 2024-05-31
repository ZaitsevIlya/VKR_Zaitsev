# Указываем базовый образ
FROM python:3.10

# Копируем содержимое текущего каталога в /app в контейнере
COPY . /app

# Рабочая директория - /app
WORKDIR /app

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исключая .venv
COPY . .

# Открываем порт, если необходимо
EXPOSE 5000

# Команда для запуска вашего приложения
CMD ["python", "server.py"]
