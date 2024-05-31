import json
import xml.etree.ElementTree as ET


"""
Конвертация эталона из XML в JSON
"""
ETALON_XML = "etalon_gpt4o.xml"

tree = ET.parse(ETALON_XML)
root = tree.getroot()

json_data = {}
for child in root:
    ar = []
    for c in child:
        ar.append(c.text)
    json_data[child.attrib["name"]] = ar

with open(f"etalon_sprav.json", 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)
