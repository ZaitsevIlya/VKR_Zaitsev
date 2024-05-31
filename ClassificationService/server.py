from flask import Flask, request, jsonify, render_template
import os
import psycopg2
from pars_etalon_to_db import parsZip
from ranging import ranging
from datetime import datetime


"""
Веб-сервер на Flask
"""
app = Flask(__name__)

# # Получение параметров подключения из переменных среды или файла конфигурации
# db_host = os.getenv('DB_HOST')
# db_user = os.getenv('DB_USER')
# db_password = os.getenv('DB_PASSWORD')
# db_name = os.getenv('DB_NAME')
#
# connection = psycopg2.connect(
#     host=db_host,
#     user=db_user,
#     password=db_password,
#     dbname=db_name
# )

# Подключение к БД
connection = psycopg2.connect(
    host="localhost",
    user="postgres",
    password="ubuntu1024",
    dbname="postgres"
)

cursor = connection.cursor()

# Основная страница
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

# Сохранение в БД истории запросов
@app.after_request
def save_request(response):
    cursor.execute(
        "INSERT INTO request_history (datetime, request_method, response_code, request_path) VALUES (%s, %s, %s, %s) RETURNING id",
        (datetime.now(), request.method, response.status_code, request.path,)
    )
    connection.commit()

    return response

# Получение списка эталонов
@app.route('/get_etalons')
def get_etalons():
    cursor.execute(f"SELECT id, name FROM etalons")
    res = cursor.fetchall()

    data = [{"id": row[0], "name": row[1], "type": "etalons"} for row in res]

    return jsonify(data)

# Получение списка моделей
@app.route('/get_models')
def get_models():
    cursor.execute(f"SELECT id, name FROM models")
    res = cursor.fetchall()

    data = [{"id": row[0], "name": row[1], "type": "models"} for row in res]

    return jsonify(data)

# Добавление эталона
@app.route('/add_ethalon', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'Файл не был выбран для загрузки', 400

    file = request.files['file']
    if file.filename == '':
        return 'Файл не был выбран для загрузки', 400

    if not file.filename.endswith('.zip'):
        return 'Файл должен быть формата .zip', 400

    filename = file.filename
    file_path = os.path.join('etalons', filename)
    file.save(file_path)
    parsZip(file_path, connection, cursor)

    return 'Файл успешно загружен', 200

# Получение топ для введеного списка данных
@app.route('/get_top_etalons', methods=['POST'])
def get_top_etalons():
    # Получение данных из формы
    directory = request.form['directory']
    etalon_id = request.form['etalonID']
    model_id = request.form['vectorizationModelID']
    metrics = request.form.getlist('metric')
    n = int(request.form['topN'])

    rang = ranging(directory, etalon_id, model_id, metrics, n, cursor)

    return jsonify(rang)

# Получение описание эталона/модели
@app.route('/get_row_info')
def get_row_info():
    id = int(request.args.get('id'))
    type_info = request.args.get('type').replace("id-", "")

    cursor.execute(f"SELECT id, description FROM {type_info}")
    res = cursor.fetchall()

    data = {row[0]: {"description": row[1]} for row in res}

    row_info = data.get(id)["description"]  # Получаем информацию о строке по id
    if row_info:
        return jsonify(row_info)  # Возвращаем информацию о строке в формате JSON
    else:
        return jsonify({'error': 'Строка не найдена'}), 404  # Если строка не найдена, возвращаем ошибку 404

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
