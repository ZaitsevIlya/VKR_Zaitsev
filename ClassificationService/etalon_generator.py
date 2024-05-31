import pandas as pd
from openai import OpenAI, OpenAIError
import os
import json
import xml.etree.ElementTree as ET
import time
import zipfile


"""
Функция обогащения эталона с помощью GPT

в функцию задаются параметры:
path - путь до файла формата .xlsx с единственной колонкой
header - True/False обозначающий наличие заголовка колонки
N - количество генераций на элемент
Name - название эталона
Description - описание эталона

На выходе будет сформировани .zip архив содержащий:
etalon.xml - файл обогащенного эталона
manifest.xml - описание обогащенного эталона
"""
def parsGPT(path, header=True, N=10, Name="Название", Description="Описание"):
    def genXML(root, tag, text):
        el = ET.Element(tag)
        el.text = text
        return root.append(el)

    try:
        if header:
            df = pd.read_excel(path)
        else:
            df = pd.read_excel(path, header=None)
    except FileNotFoundError as e:
        print("File not found:", e)
    except pd.errors.ParserError as e:
        print("Error parsing Excel file:", e)

    etalons = [df.to_dict()[d] for d in df.to_dict()][0]

    del df

    os.environ["OPENAI_API_KEY"] = "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"
    client = OpenAI()

    ETALON_XML = "etalon.xml"
    MANIFRST_XML = "manifest.xml"

    root = ET.Element('manifest')
    genXML(root, 'name', Name)
    genXML(root, 'description', Description)
    genXML(root, 'type', "Должности")
    genXML(root, 'number-in', str(len(etalons)))
    genXML(root, 'number-gen', str(N))
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(MANIFRST_XML, encoding='utf-8', xml_declaration=True, method='xml')

    root = ET.Element('etalon-generator')
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(ETALON_XML, encoding='utf-8', xml_declaration=True, method='xml')

    for et in etalons:
        sys_cont = "Дай ответ в формате JSON, где ключом является исходная должность, а значением - массив сгенерированных должностей."
        u_cont = f"Для должностей '{etalons[et]}' сгенерируй для каждой {N} точно таких же должностей, но другими словами. Используй сокращения, перестановку слов, аббревиатуры и т.п. Не придумывай несуществующие слова и выражения, не повторяйся, если у тебя закончились варианты, то не генерируй ничего."

        try:
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": sys_cont},
                    {"role": "user",
                     "content": u_cont}
                ]
            )
        except OpenAIError as e:
            print(e)
        else:
            ans = json.loads(response.choices[0].message.content)

            tree = ET.parse(ETALON_XML)
            root = tree.getroot()

            for a in ans:
                etalon_el = ET.Element('etalon', name=a)

                for el in ans[a]:
                    genXML(etalon_el, 'generated-name', el)

                root.append(etalon_el)

            tree = ET.ElementTree(root)
            ET.indent(tree)
            tree.write(ETALON_XML, encoding='utf-8', xml_declaration=True, method='xml')


    with zipfile.ZipFile(f"etalon_{int(time.time())}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in [MANIFRST_XML, ETALON_XML]:
            zipf.write(file)