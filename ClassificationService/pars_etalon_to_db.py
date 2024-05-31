import zipfile
import xml.etree.ElementTree as ET


"""
Функция добавления эталона (zip - архива) в БД
"""
def parsZip(file_in, connection, cursor):
    with zipfile.ZipFile(file_in, 'r') as zip_ref:
        with zip_ref.open("manifest.xml") as file:
            data = file.read()
            root = ET.fromstring(data)
            name = root.find("name").text
            description = root.find("description").text

            cursor.execute(
                "INSERT INTO etalons (name, description, path) VALUES (%s, %s, %s) RETURNING id",
                (name, description, file_in)
            )
            etalons_id = cursor.fetchone()[0]
            connection.commit()

        with zip_ref.open("etalon.xml") as file:
            data = file.read()
            root = ET.fromstring(data)

            for elem in root.findall('etalon'):
                etalon = elem.get("name")
                cursor.execute(
                    "INSERT INTO etalons_list (etalon, etalons_id) VALUES (%s, %s) RETURNING id",
                    (etalon, etalons_id)
                )

                etalon_id = cursor.fetchone()[0]

                cursor.execute(
                    "INSERT INTO etalons_gen (etalon_gen, etalon_id) VALUES (%s, %s)",
                    (etalon, etalon_id)
                )

                connection.commit()

                for e in elem:
                    etalon_gen = e.text
                    cursor.execute(
                        "INSERT INTO etalons_gen (etalon_gen, etalon_id) VALUES (%s, %s)",
                        (etalon_gen, etalon_id)
                    )
                    connection.commit()