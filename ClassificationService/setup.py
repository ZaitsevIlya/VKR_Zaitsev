from pars_etalon_to_db import parsZip
import json


"""
Функция инициализации БД на Postgres

входные параметры:
connection - объект БД
cursor - курсор БД
"""
def dbInit(connection, cursor):
    cursor.execute("DROP TABLE IF EXISTS request_history;")
    cursor.execute("DROP TABLE IF EXISTS models;")
    cursor.execute("DROP TABLE IF EXISTS etalons_gen;")
    cursor.execute("DROP TABLE IF EXISTS etalons_list;")
    cursor.execute("DROP TABLE IF EXISTS etalons;")

    cursor.execute("CREATE TABLE etalons ("
                   "id SERIAL PRIMARY KEY, "
                   "name VARCHAR(50), "
                   "description TEXT, "
                   "path VARCHAR(100))"
    )
    cursor.execute("CREATE TABLE etalons_list ("
                   "id SERIAL PRIMARY KEY, "
                   "etalon TEXT, "
                   "etalons_id INTEGER REFERENCES etalons(id))"
    )
    cursor.execute("CREATE TABLE etalons_gen ("
                   "id SERIAL PRIMARY KEY, "
                   "etalon_id INTEGER REFERENCES etalons_list(id), "
                   "etalon_gen TEXT, "
                   "sbert_large DOUBLE PRECISION[])"
    )
    cursor.execute("CREATE TABLE models ("
                   "id SERIAL PRIMARY KEY, "
                   "name VARCHAR(50), "
                   "description TEXT, "
                   "path VARCHAR(100));"
    )
    cursor.execute("CREATE TABLE request_history  ("
                   "id SERIAL PRIMARY KEY, "
                   "datetime TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                   "request_method VARCHAR(10) NOT NULL, "
                   "response_code INTEGER NOT NULL, "
                   "request_path TEXT NOT NULL);"
    )

    cursor.execute("INSERT INTO models (name, description, path) VALUES ("
                   "'sbert_large', "
                   "'Большая модель BERT (без корпуса) для встраивания предложений на русском языке', "
                   "'models/sbert_large/');"
    )
    connection.commit()

    # добавление Алтайского эталона
    parsZip("etalons/etalon_1713595323.zip", connection, cursor)

    # добавление эмбеддингов Алтайского эталона
    with open('embeddings/etalon_1713595323.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    cursor.execute(
        f"SELECT etalons_list.id, etalon FROM etalons_list INNER JOIN etalons ON etalons_list.etalons_id=etalons.id and etalons.id=1")
    res = cursor.fetchall()
    for i in res:
        cursor.execute(
            f"SELECT etalons_gen.id, etalon_gen FROM etalons_gen INNER JOIN etalons_list ON etalons_gen.etalon_id=etalons_list.id and etalons_list.id={i[0]}")
        res2 = cursor.fetchall()
        for j in res2:
            cursor.execute("UPDATE etalons_gen SET sbert_large=%s WHERE id=%s", (json_data[i[1]][j[1]], j[0]))
            connection.commit()