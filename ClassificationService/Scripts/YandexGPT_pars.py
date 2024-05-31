import requests
import pandas as pd
import random
import time
import json
import os


"""
Обогащение эталона с помощью YandexGPT API

выходные параметры:
folder_id - ID папки в yandex cloud
IAM - токен
model - используемая модель
file - путь до файла эталонов
N - количество эталонов для генерации, 0 - весь файл
K - количество генераций на эталон
M - количество эталонов используемых в генераций за один запрос к API
"""
def parsGPT(folder_id, IAM, model, file, N=0, K=1, M=1):
    if N < M:
        return ("N не может быть меньше M.")

    df = pd.read_excel(file)

    data = [df.to_dict()[d] for d in df.to_dict()][0]
    # print(data)

    if N == 0:
        N = len(data)

    keys = list(data.keys())
    random.shuffle(keys)

    data_N = []
    i = 0
    while len(data_N) < N:
        data_N.append(data[keys[i]])
        i += 1
    print(len(data_N))

    out_file = f'yandex out N={N} K={K} M={M}.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({'Время': 0, 'Токены input': 0, 'Токены output': 0, 'Токенов всего': 0}, f, ensure_ascii=False, indent=4)

    start_time = time.time()

    while data_N != []:
        sd_data = data_N[:M]
        sys_cont = (f'Ты - генератор должностей. Сгенерируй {K} точно таких же должностей, но другими словами. '
                    'Используй сокращения, перестановку слов, аббревиатуры и т.п. '
                    'Не придумывай несуществующие слова и выражения, не повторяйся, если у тебя закончились варианты, то не генерируй ничего. '
                    'Ответ должен быть в формате JSON, например: {"исходная должность": ["должность1", "должность2", "должность3"]}')
        u_cont = f"{sd_data[0]}"

        prompt = {
            "modelUri": f"gpt://{folder_id}/{model}",
            "completionOptions": {
                "stream": False,
                "temperature": 0.5,
                "maxTokens": "2000"
            },
            "messages": [
                {
                    "role": "system",
                    "text": sys_cont
                },
                {
                    "role": "user",
                    "text": u_cont
                },
            ]
        }

        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {IAM}"
        }

        response = requests.post(url, headers=headers, json=prompt)
        result = json.loads(response.text)

        if int(result["result"]["usage"]["completionTokens"]) < 2000:
            with open(out_file, encoding='utf-8') as f:
                out_data = json.load(f)

            out_data["Время"] += time.time() - start_time
            start_time = time.time()
            out_data["Токены input"] += int(result["result"]["usage"]["inputTextTokens"])
            out_data["Токены output"] += int(result["result"]["usage"]["completionTokens"])
            out_data["Токенов всего"] += int(result["result"]["usage"]["totalTokens"])

            print(result["result"]["alternatives"][0]["message"]["text"])
            print("-" * 30)
            ans = json.loads(result["result"]["alternatives"][0]["message"]["text"])
            # print(ans)
            # print("#" * 30)
            for key in ans:
                print(key)
                out_data[key] = ans[key]
            # out_data[sd_data[0]] = result["result"]["alternatives"][0]["message"]["text"]

            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(out_data, f, ensure_ascii=False, indent=4)

            data_N = data_N[M:]
        else:
            M -= 1
            print(f"M={M}")

        time.sleep(1)

start_time_f = time.time()

folder_id = "указать свое"

raw_IAM_token = os.popen('''curl -d "{\\"yandexPassportOauthToken\\":\\"указать свое\\"}" "https://iam.api.cloud.yandex.net/iam/v1/tokens"''').read()
IAM_token = json.loads(raw_IAM_token)["iamToken"]

modelGPT = "yandexgpt"

parsGPT(folder_id, IAM_token, modelGPT, "data/Алтай_Эталоны.xlsx", 50, 5, 1)

print("Время выполнения: %f.2 сек." % (time.time() - start_time_f))