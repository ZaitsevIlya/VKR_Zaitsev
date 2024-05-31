from openai import OpenAI, OpenAIError
import os
import json


"""
Получение эмбеддингов через GPT
"""
def create_emb(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        ).data[0].embedding

        return response
    except OpenAIError as e:
        print(e)

os.environ["OPENAI_API_KEY"] = "sk-proj-e0kPRPJ0KD4xFb2x5zIGT3BlbkFJnCeiHZIhSsY0dZTUY9d8"
client = OpenAI()

with open("etalon_sprav.json", encoding='utf-8') as f:
    standarts = json.load(f)

with open("emb_sprav_openai_1.json", encoding='utf-8') as f:
    out_json = json.load(f)

for standart in standarts:
    print(standart)
    name_group = standart
    group = standarts[standart]
    group.append(name_group)
    group_dict = {}
    for sd in group:
        group_dict[sd] = create_emb(sd)

    out_json[name_group] = group_dict

    with open("emb_sprav_openai.json", 'w', encoding='utf-8') as f:
        json.dump(out_json, f, ensure_ascii=False, indent=4)