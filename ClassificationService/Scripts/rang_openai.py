"""
Ранжирование эталона после векторизации через GPT
"""
# import scipy.spatial.distance as ds
# import json
# import time
#
#
# start_time_f = time.time()
#
# with open('emb_sd_openai.json', 'r', encoding='utf-8') as f:
#     positions = json.load(f)
#
# with open('emb_sprav_openai_full.json', 'r', encoding='utf-8') as f:
#     etalons = json.load(f)
#
# res_out = {}
# for pos in positions:
#     print(pos)
#     res = []
#
#     for etalon in etalons:
#         cos_group = []
#         for et in etalons[etalon]:
#             res_cos = ds.cosine(etalons[etalon][et], positions[pos])
#             cos_group.append(res_cos)
#
#         distance = {}
#         distance["Должность"] = etalon
#         metrics = ["min", "mean", "max"]
#         if "min" in metrics:
#             distance["min"] = round(min(cos_group), 2)
#         if "mean" in metrics:
#             distance["mean"] = round(sum(cos_group) / len(cos_group), 2)
#         if "max" in metrics:
#             distance["max"] = round(max(cos_group), 2)
#
#         res.append(distance)
#
#     res_sort = sorted(res, key=lambda item: tuple(item[param] for param in metrics))[:5]
#
#     res_out[pos] = res_sort
#
# with open("sprav_openai.json", 'w', encoding='utf-8') as f:
#     json.dump(res_out, f, ensure_ascii=False, indent=4)
#
# print("Время выполнения: %f.2 сек." % (time.time() - start_time_f))


"""
Ранжирование эталона после векторизации через GPT с использованием multiprocessing
"""
import multiprocessing
import scipy.spatial.distance as ds
import time
import json


def sequential(proc, data, etalons):
    print(f"Запускаем поток № {proc}")

    res_out = {}
    counter = 0

    for pos in data:
        print(f"Процессор № {proc}, Должность: {counter}")
        res = []

        for etalon in etalons:
            cos_group = []
            for et in etalons[etalon]:
                res_cos = ds.cosine(etalons[etalon][et], data[pos])
                cos_group.append(res_cos)

            distance = {}
            distance["Должность"] = etalon
            metrics = ["min", "mean", "max"]
            if "min" in metrics:
                distance["min"] = round(min(cos_group), 2)
            if "mean" in metrics:
                distance["mean"] = round(sum(cos_group) / len(cos_group), 2)
            if "max" in metrics:
                distance["max"] = round(max(cos_group), 2)

            res.append(distance)

        res_sort = sorted(res, key=lambda item: tuple(item[param] for param in metrics))[:5]

        res_out[pos] = res_sort

        counter += 1

    with open(f"sprav_openai_{proc}.json", 'w', encoding='utf-8') as f:
        json.dump(res_out, f, ensure_ascii=False, indent=4)

    print(f"Вычисления закончены. Процессор № {proc}")

def processesed(procs, calc, sd, etalon):
    processes = []

    for proc in range(procs):
        calc_s = calc * proc
        calc_e = calc * (proc + 1)
        print(calc_s, calc_e)
        data = {}
        for p, i in zip(sd, range(len(sd))):
            if i >= calc_s and i < calc_e:
                data[p] = sd[p]

        # print(data)
        print(len(data))

        p = multiprocessing.Process(target=sequential, args=(proc, data, etalon))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    start_time = time.time()

    with open('emb_sd_openai.json', 'r', encoding='utf-8') as f:
        positions = json.load(f)

    with open('emb_sprav_openai_full.json', 'r', encoding='utf-8') as f:
        etalons_dic = json.load(f)

    n_proc = multiprocessing.cpu_count()
    calc = len(positions) // n_proc + 1
    processesed(n_proc, calc, positions, etalons_dic)

    data_full = {}
    for i in range(n_proc):
        with open(f"sprav_openai_{i}.json", encoding='utf-8') as f:
            out_data = json.load(f)

        data_full = {**out_data, **data_full}

    with open(f"sprav_openai.json", 'w', encoding='utf-8') as f:
        json.dump(data_full, f, ensure_ascii=False, indent=4)

    print("\n\nВремя выполнения: ", time.time() - start_time)
