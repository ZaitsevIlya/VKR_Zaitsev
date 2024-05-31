import json
import xml.etree.ElementTree as ET


"""
Конвертация эталона из JSON в XML
"""
def genXML(root, tag, text):
    el = ET.Element(tag)
    el.text = text
    return root.append(el)

ETALON_XML = "etalon.xml"
root = ET.Element('etalon-generator')
tree = ET.ElementTree(root)

with open('Altai_full.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

for data in json_data:
    etalon_el = ET.Element('etalon', name=data)

    for el in json_data[data]:
        genXML(etalon_el, 'generated-name', el)

    root.append(etalon_el)

tree = ET.ElementTree(root)
ET.indent(tree)
tree.write(ETALON_XML, encoding='utf-8', xml_declaration=True, method='xml')