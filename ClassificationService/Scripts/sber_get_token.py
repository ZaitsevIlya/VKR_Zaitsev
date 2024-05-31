import requests
import time
import json


"""
Обогащение с использованием GigaChat API
"""


# Получение токена
url_token = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

payload='scope=GIGACHAT_API_PERS'
headers = {
  'Content-Type': 'application/x-www-form-urlencoded',
  'Accept': 'application/json',
  'RqUID': 'указать свой',
  'Authorization': 'Basic указать свой'
}

response = requests.request("POST", url_token, headers=headers, data=payload, verify=False)
token = response.json()['access_token']
# print(token)

# Запрос на генерацию
start_time = time.time()

url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

payload = json.dumps({
  "model": "GigaChat",
  "messages": [
    {
      "role": "system",
      "content": 'Сгенерируй от 5 до 20 синонивов для должности "водитель". '
                 'Придерживайся следующих правил: синонимы должны быть уникальны, '
                 'не придумывай несуществующие слова и выражения, используй сокращения или аббревиатура, если у тебя закончились варианты, '
                 'то не генерируй ничего. Результат верни в формате JSON-массива без каких-либо пояснений, например, '
                 '{"название должности": ["должность1", "должность2"]}.'
    },
  ],
  "temperature": 1.0,
  "top_p": 0.1,
  "n": 1,
  "stream": False,
  "max_tokens": 512,
  "repetition_penalty": 1
})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'Authorization': f'Bearer {token}'
}

response = requests.request("POST", url, headers=headers, data=payload, verify=False)

print(response.json())
print(response.json()["choices"])
print(response.json()["choices"][0]["message"])
print(response.json()["choices"][0]["message"]["content"].replace("],", "]"))

out_file = 'gigachat.json'
# with open(out_file, 'w', encoding='utf-8') as f:
#     json.dump({'Время': 0, 'Токены input': 0, 'Токены output': 0}, f, ensure_ascii=False, indent=4)

with open(out_file, encoding='utf-8') as f:
  out_data = json.load(f)

out_data["Время"] += time.time() - start_time
out_data["Токены input"] += response.json()["usage"]["prompt_tokens"] + response.json()["usage"]["system_tokens"]
out_data["Токены output"] += response.json()["usage"]["completion_tokens"]

ans = json.loads(response.json()["choices"][0]["message"]["content"].replace("],", "]"))
for key in ans:
    out_data[key] = ans[key]

with open(out_file, 'w', encoding='utf-8') as f:
  json.dump(out_data, f, ensure_ascii=False, indent=4)

# print(type(response.json()["choices"][0]["message"]["content"]))