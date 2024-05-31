from transformers import AutoTokenizer, AutoModel
import multiprocessing
import scipy.spatial.distance as ds
import pandas as pd
import time
import torch
import json


"""
Ранжирование эталона с помощью sber_large с использованием multiprocessing
"""
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def sequential(proc, data, etalons, model, tokenizer):
    print(f"Запускаем поток № {proc}")

    metrics = ["min", "mean", "max"]
    N = 5
    res_out = {}
    counter = 0
    out_data = {}

    for pos in data:
        print(f"Процессор № {proc}, Должность: {counter}")
        encoded_input = tokenizer([pos], padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        pos_emb = sentence_embeddings.numpy()[0].tolist()

        res = []

        for etalon in etalons:
            cos_group = []
            for et in etalons[etalon]:
                res_cos = ds.cosine(etalons[etalon][et], pos_emb)
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

        # res_step = {pos: []}
        # for r in res_sort:
        #     res_step[pos].append(r["Должность"])
        #
        # res_out = {**res_out, **res_step}

        out_data[pos] = res_sort
        counter += 1

    out_file = f"sprav_{proc}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=4)

    print(f"Вычисления закончены. Процессор № {proc}")

def processesed(procs, calc, sd, etalon, model, tokenizer):
    processes = []

    for proc in range(procs):
        calc_s = calc * proc
        calc_e = calc * (proc + 1)
        print(calc_s, calc_e)
        data = []
        for p, i in zip(sd, range(len(sd))):
            if i >= calc_s and i < calc_e:
                data.append(sd[i])

        # print(data)
        print(len(data))

        p = multiprocessing.Process(target=sequential, args=(proc, data, etalon, model, tokenizer))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")

    with open('emb_sprav.json', 'r', encoding='utf-8') as f:
        etalons_dict = json.load(f)

    df = pd.read_excel("Должности со всех ОД 16.04.xlsx", header=None)
    positions = [df.to_dict()[0][d] for d in df.to_dict()[0]]
    del df

    n_proc = multiprocessing.cpu_count()
    calc = len(positions) // n_proc + 1
    processesed(n_proc, calc, positions, etalons_dict, model, tokenizer)

    data_full = {}
    for i in range(n_proc):
        with open(f"sprav_{i}.json", encoding='utf-8') as f:
            out_data = json.load(f)

        data_full = {**out_data, **data_full}

    with open(f"sprav_new_2.json", 'w', encoding='utf-8') as f:
        json.dump(data_full, f, ensure_ascii=False, indent=4)

    print("\n\nВремя выполнения: ", time.time() - start_time)
