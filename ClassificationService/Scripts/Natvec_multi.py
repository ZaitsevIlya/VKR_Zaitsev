import multiprocessing
from navec import Navec
import csv
import string
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time


"""
Ранжирование эталона с помощью Navec с использованием multiprocessing
"""
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

def sequential(proc, data,  sd, sds_emb, nvc):
    print(f"Запускаем поток № {proc}")

    max_similarity = 0.5
    data_pd = pd.DataFrame()

    for d in data:
        w2v = []
        for pos in data[d].split():
            if pos in nvc:
                w2v.append(nvc[pos])

        if w2v != []:
            if len(w2v) > 1:
                pos2vec = np.average(w2v, axis=0, keepdims=True)
            else:
                pos2vec = w2v

            res_sd = {}
            for sd_emb in sds_emb:
                res = cosine_similarity(pos2vec, sds_emb[sd_emb])

                if res[0][0] >= max_similarity:
                    res_sd[sd[sd_emb]] = res[0][0]

            res_sd = sorted(res_sd.items(), key=lambda item: item[1], reverse=True)

            data_out = {}
            data_out["название"] = d
            for rsd, i in zip(res_sd[:5], range(1, 6)):
                data_out[f"эталон {i}"] = rsd[0]
            data_pd = data_pd._append(data_out, ignore_index=True)

    writer = pd.ExcelWriter(f"out_{proc}.xlsx")
    data_pd.to_excel(writer, index=False)
    writer.close()
    print(f"Вычисления закончены. Процессор № {proc}")

def processesed(procs, calc, sd, sds_emb, pos, nvc):
    processes = []

    for proc in range(procs):
        calc_s = calc * proc
        calc_e = calc * (proc + 1)
        # print(calc_s, calc_e)
        data = {}
        for p, i in zip(pos, range(len(pos))):
            if i >= calc_s and i < calc_e:
                data[p] = pos[p]

        p = multiprocessing.Process(target=sequential, args=(proc, data, sd, sds_emb, nvc))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    start_time = time.time()

    path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
    navec = Navec.load(path)

    standarts = list2dict('Эталон.csv')
    positions = list2dict('Должности.csv')

    standarts_emb = {}
    for standart in standarts:
        w2v = []
        for sd in standart.split():
            if sd in navec:
                w2v.append(navec[sd])

        if w2v != []:
            if len(w2v) > 1:
                sd2vec = np.average(w2v, axis=0, keepdims=True)
                standarts_emb[standart] = sd2vec
            else:
                standarts_emb[standart] = w2v

    n_proc = multiprocessing.cpu_count()
    calc = len(positions) // n_proc + 1
    processesed(n_proc, calc, standarts, standarts_emb, positions, navec)

    print("\n\nВремя выполнения: ", time.time() - start_time)
