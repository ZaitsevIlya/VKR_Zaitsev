from transformers import AutoTokenizer, AutoModel
import multiprocessing
import csv
import string
import re
import time
import torch
import json


"""
Получение файла эмбеддингов для обогащенного эталона с помощью sbert_large и использованием multiprocessing
"""
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def list2dict(file):
    num = string.digits
    pun = string.punctuation
    symbol_del = re.escape(num + pun + "№")

    dict_out = {}

    f = open(file, 'r', newline='', encoding="utf-8-sig")
    lst = csv.reader(f)

    for el in lst:
        new_el = el[0].lower().replace("﻿", "")
        new_el = ' '.join(re.sub('[' + symbol_del + ']', " ", new_el).split())
        dict_out[new_el] = el[0]

    f.close()

    return dict_out

def sequential(proc, data, model, tokenizer):
    print(f"Запускаем поток № {proc}")

    standarts_emb = {}
    for standart in data:
        name_group = standart
        group = data[standart]
        group.append(name_group)
        group_dict = {}
        for sd in group:
            encoded_input = tokenizer([sd], padding=True, truncation=True, max_length=24, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            group_dict[sd] = sentence_embeddings.numpy()[0].tolist()

        standarts_emb[name_group] = group_dict

        with open(f"emb_sprav_{proc}.json", 'w', encoding='utf-8') as f:
            json.dump(standarts_emb, f, ensure_ascii=False, indent=4)

    print(f"Вычисления закончены. Процессор № {proc}")

def processesed(procs, calc, sd, model, tokenizer):
    processes = []

    for proc in range(procs):
        calc_s = calc * proc
        calc_e = calc * (proc + 1)
        # print(calc_s, calc_e)
        data = {}
        for p, i in zip(sd, range(len(sd))):
            if i >= calc_s and i < calc_e:
                data[p] = sd[p]

        p = multiprocessing.Process(target=sequential, args=(proc, data, model, tokenizer))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")

    with open("etalon_sprav.json", encoding='utf-8') as f:
        standarts = json.load(f)

    n_proc = multiprocessing.cpu_count()
    calc = len(standarts) // n_proc + 1
    processesed(n_proc, calc, standarts, model, tokenizer)

    data_full = {}
    for i in range(n_proc):
        with open(f"emb_sprav_{i}.json", encoding='utf-8') as f:
            out_data = json.load(f)

        data_full = {**out_data, **data_full}

    with open(f"emb_sprav.json", 'w', encoding='utf-8') as f:
        json.dump(data_full, f, ensure_ascii=False, indent=4)

    print("\n\nВремя выполнения: ", time.time() - start_time)
