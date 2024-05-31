from transformers import AutoTokenizer, AutoModel
import torch
import string
import re
import scipy.spatial.distance as ds


"""
Функция поиска топ - N эталонов

входные параметры:
list - список (через запятую) для ранжирования
etalonID - ID эталона из БД по которому производится ранжирование
modelID - ID модели из БД с помощью которой производится ранжирование
metrics - метрики по которым производится ранжирование (min, mean, max)
N - количество эталонов на выходе (топ)
crs - объект БД
"""
def ranging(list, etalonID, modelID, metrics, N, crs):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    crs.execute(f"SELECT name, path FROM models WHERE id={modelID}")
    model_name, model_path = crs.fetchone()

    etalons_dict = {}
    crs.execute(f"SELECT id, etalon FROM etalons_list WHERE etalons_id={etalonID}")
    etalons = crs.fetchall()
    for etalon in etalons:
        etalons_dict[etalon[1]] = {}
        crs.execute(f"SELECT etalon_gen, {model_name}  FROM etalons_gen WHERE etalon_id={etalon[0]}")
        etalons_gen = crs.fetchall()
        for el in etalons_gen:
            etalons_dict[etalon[1]][el[0]] = el[1]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    positions = list.split(",")

    # num = string.digits
    # pun = string.punctuation
    # symbol_del = re.escape(num + pun + "№")

    res_out = {}

    for pos in positions:
        # удаление спецсимволов, цифр и т.п.
        # ps = ' '.join(re.sub('[' + symbol_del + ']', " ", pos).split())
        ps = pos

        encoded_input = tokenizer([ps], padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        pos_emb = sentence_embeddings.numpy()[0].tolist()

        res = []

        for etalon in etalons_dict:
            cos_group = []
            for et in etalons_dict[etalon]:
                res_cos = ds.cosine(etalons_dict[etalon][et], pos_emb)
                cos_group.append(res_cos)

            distance = {}
            distance["Должность"] = etalon
            if "min" in metrics:
                distance["min"] = round(min(cos_group), 2)
            if "mean" in metrics:
                distance["mean"] = round(sum(cos_group) / len(cos_group), 2)
            if "max" in metrics:
                distance["max"] = round(max(cos_group), 2)

            res.append(distance)

        res_sort = sorted(res, key=lambda item: tuple(item[param] for param in metrics))[:N]
        res_step = {pos: []}
        for r in res_sort:
            res_step[pos].append(r["Должность"])

        res_out = {**res_out, **res_step}

    return res_out