import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os
import abbreviations as abbs
import torch.nn.functional as F
import json
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

device = "cuda:0" if torch.cuda.is_available() else "cpu"
vectorization_batch_size = 1000


print(device)
def read_file(file, sep='#'):
    with open(file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=sep)
        df = pd.DataFrame(reader)
    return df


def process_df_1(df):
    df.pop(1)
    df.pop(3)
    df.pop(0)
    df.pop(2)
    df = df.dropna()
    df[4] = df[4].str.lower()
    df = df.drop_duplicates(keep='first', ignore_index=True, subset=4)
    df = df.rename(columns={4:'str'})
    df.to_excel('./all-data.xlsx', index=False)
    return df

def process_df_abbrevs(df, column='str', new_column='no_abbs', need_save = False, filename = None):
    no_abbs = []
    for val in df[column].values:
        res = abbs.unmasking(val)
        no_abbs.append(res)
    df[new_column] = no_abbs
    df = process_df_1(df)
    if(need_save and not filename is None):
        df.to_excel(filename, index=False)
    return df

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def initModels(sentences):
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")
    print('Starting...')
    df = read_file('./Проекты/Алтай/Алтай ЗКГУ1.csv')
    df = process_df_1(df)
    sentences = df[4][:10000].tolist()


def comparison(set1, set2):
    m1 = 0
    for i in set1:
        if (i in set2):
            m1 += 1
    return m1 / (len(set1) + len(set2) - m1)


def UseAgglomerativeClustering(sentences, model_type = 1, distance = 0.125, embeddings = None):

    iter = 0
    corpus_embeddings = None
    if(embeddings is None):
        while(iter*vectorization_batch_size <= len(sentences)):
            tmp = vectorize(sentences[min(iter*vectorization_batch_size, len(sentences)-1):min(len(sentences), (iter+1)*vectorization_batch_size)], model_type)
            if(corpus_embeddings is None):
                corpus_embeddings = tmp
            else:
                corpus_embeddings = torch.concat([corpus_embeddings, tmp], 0)
            iter += 1
    else:
        print(len(embeddings))
        corpus_embeddings = embeddings
    print(corpus_embeddings.shape)
    #corpus_embeddings = vectorize(sentences, model_type)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='complete',
                                               distance_threshold=distance)
    corpus_embeddings = corpus_embeddings.cpu()
    print('Shape', corpus_embeddings.shape)
    clustering_model.fit(corpus_embeddings)

    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(sentences[sentence_id])

    res = pd.DataFrame(columns=['str', 'res'])
    for i, cluster in enumerate(clustered_sentences):
        for elem in clustered_sentences[cluster]:
            tmp_df = pd.DataFrame({'str': [elem], 'res': [i + 1]})
            res = pd.concat([res, tmp_df], ignore_index=True)
    res.to_excel('./AgglomerativeClustering-'+str(model_type)+'-'+str(distance)+'.xlsx', index=False)
    return clustering_model

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def UseAgglomerativeClustering_OpenAI(sentences, model_type = 1, distance = 0.125, embeddings = None):
    iter = 0
    corpus_embeddings = None
    if(embeddings is None):
        while(iter*vectorization_batch_size <= len(sentences)):
            tmp = vectorize(sentences[min(iter*vectorization_batch_size, len(sentences)-1):min(len(sentences), (iter+1)*vectorization_batch_size)], model_type)
            if(corpus_embeddings is None):
                corpus_embeddings = tmp
            else:
                corpus_embeddings = torch.concat([corpus_embeddings, tmp], 0)
            iter += 1
    else:
        corpus_embeddings = embeddings
    clustering_model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="ward", distance_threshold=distance)
    corpus_embeddings = corpus_embeddings.cpu()
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(sentences[sentence_id])
    res = pd.DataFrame(columns=['str', 'res'])
    for i, cluster in enumerate(clustered_sentences):
        for elem in clustered_sentences[cluster]:
            tmp_df = pd.DataFrame({'str': [elem], 'res': [i + 1]})
            res = pd.concat([res, tmp_df], ignore_index=True)
    res.to_excel('./AgglomerativeClustering-'+str(model_type)+'-'+str(distance)+'.xlsx', index=False)
    return clustering_model

def vectorize(sentences, model_type = 1):
    sentence_embeddings = None
    if(model_type == 1):
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = model.to(device)
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    elif(model_type == 2):
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        model = model.to(device)
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input, return_dict=True)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    elif (model_type == 3):
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        model = model.to(device)
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    elif (model_type == 4):
        embedder = AutoModel.from_pretrained("ai-forever/ruElectra-large")
        tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruElectra-large")
        embedder = embedder.to(device)
        encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = embedder(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

def get_dist(beg, end, step):
    arr = [beg]
    curr = beg + step
    while(curr <= end):
        arr.append(curr)
        curr += step
    return arr

def MergeFiles(dir):
    global data
    print('Process:', dir)
    for f in os.listdir(dir):
        if(os.path.isdir(dir+'/'+f)):
            MergeFiles(dir + '/' + f + '/')
        else:
            ext = f.split(".")[-1]
            if(ext != 'csv'):
                continue
            print('File:', f)
            curr_data = read_file(dir + '/' + f)
            data = pd.concat([data, curr_data], ignore_index=True)

def PreprocessingData():
    df = read_file('./all-data.csv')
    df = process_df_1(df)
    df.to_excel('./tmp.xlsx', index=False)
    #df = process_df_abbrevs(df, 'str', need_save=True, filename='./no_abbrevs.xlsx')
    #return df

def First():
    data = pd.DataFrame()
    MergeFiles('./Проекты')
    print(data)
    data.to_csv('./all-data.csv', sep='#', index=False)

def ClusteringData(df, models, dists, column='str'):
    sentences = df[column].tolist()
    for m in models:
        print('Starting of working with',m,'model')
        for dist in dists:
            print('Starting of working with', dist, 'dist')
            UseAgglomerativeClustering(sentences, m, dist)
            print('Ready')

def getIncidenceMatrix(data):
    result = {}
    M = data['res'].max()

    for i in range(1, M+1):
        linked = data.loc[data['res'] == i]['str'].to_list()
        for A1 in range(0, len(linked)):
            for A2 in range(A1, len(linked)):
                result[(linked[A1], linked[A2])] = 1
    return result

def getGroups(data):
    processed = []
    groups = {}
    df = pd.DataFrame(columns=[1, 2])
    for (k1, k2) in data.keys():
        tmp_df = pd.DataFrame({1:[k1], 2:[k2]})
        df = pd.concat([df, tmp_df], ignore_index=True)
    group_number = 0
    for key in df[1].values:
        if(not key in processed):
            group_number += 1
            processed.append(key)
            groups[group_number] = [key]
            list_2 = df.loc[df[1] == key][2].values
            for k2 in list_2:
                if(not k2 in processed):
                    groups[group_number].append(k2)
                    processed.append(k2)
    return groups

def getModelStability(m, dist, acc):
    data = pd.read_excel('AgglomerativeClustering-'+str(m)+'-'+str(dist)+'.xlsx', engine='openpyxl')
    data_plus = pd.read_excel('AgglomerativeClustering-'+str(m)+'-'+str(dist+acc)+'.xlsx', engine='openpyxl')
    data_minus = pd.read_excel('AgglomerativeClustering-'+str(m)+'-'+str(dist+acc)+'.xlsx', engine='openpyxl')

    matrix = getIncidenceMatrix(data)
    matrix_plus = getIncidenceMatrix(data_plus)
    matrix_minus = getIncidenceMatrix(data_minus)

    for (k1, k2) in matrix_plus.keys():
        try:
            matrix[(k1,k2)] += matrix_plus[(k1,k2)]
        except:
            matrix[(k1, k2)] = matrix_plus[(k1, k2)]
    for (k1, k2) in matrix_minus.keys():
        try:
            matrix[(k1, k2)] += matrix_minus[(k1, k2)]
        except:
            matrix[(k1, k2)] = matrix_minus[(k1, k2)]

    three_only = {}
    for (k1, k2) in matrix.keys():
        if(matrix[(k1, k2)] == 3 and k1 != k2):
            three_only[(k1,k2)] = 1
    groups = getGroups(three_only)

    result = pd.DataFrame(columns=['str', 'res'])
    for gr in groups.keys():
        for elem in groups[gr]:
            tmp = pd.DataFrame({'str':[elem], 'res':[gr]})
            result = pd.concat([result, tmp], ignore_index=True)

    result.to_excel('AgglomerativeClusteringStability-'+str(m)+'-'+str(dist)+'-'+str(acc)+'.xlsx', index=False)
    return result


def getData(m, dist):
    data = pd.read_excel('AgglomerativeClustering-' + str(m) + '-' + str(dist) + '.xlsx', engine='openpyxl')
    return data


def getAllData():
    df = pd.read_excel('./all-data.xlsx', engine='openpyxl')
    return df

#set1 \ set2
def getDifference(data):
    allData = getAllData()['str'].to_list()
    res = []
    for s in allData:
        if(not s in data):
            res.append(s)
    return res

def getModelsSummarization(models, dist, useStabilization = False, acc = 0):
    data = []
    for model in models:
        if(useStabilization and acc != 0):
            data.append(getModelStability(model, dist, acc))
        else:
            data.append(getData(model, dist))

    dict = {}

    for model in data:
        inc = getIncidenceMatrix(model)
        for (k1, k2) in inc.keys():
            try:
                dict[(k1,k2)] += 1
            except:
                dict[(k1,k2)] = 1

    three_only = {}
    for (k1, k2) in dict.keys():
        if (dict[(k1, k2)] == len(data) and k1 != k2):
            three_only[(k1, k2)] = 1

    groups = getGroups(three_only)

    result = pd.DataFrame(columns=['str', 'res'])
    for gr in groups.keys():
        for elem in groups[gr]:
            tmp = pd.DataFrame({'str': [elem], 'res': [gr]})
            result = pd.concat([result, tmp], ignore_index=True)

    result.to_excel('AgglomerativeClusteringSummarization-' + str(useStabilization) + '-' + str(dist) + '-' + str(acc) + '.xlsx',
                    index=False)

    diff = getDifference(result['str'].to_list())
    dfDiff = pd.DataFrame()
    dfDiff['str'] = diff
    dfDiff.to_excel('AgglomerativeClusteringDifference.xlsx', index=False)

    return result

def getModelsSummarizationSolo(models, dist, useStabilization = False, acc = 0):
    data = []
    for model in models:
        if (useStabilization and acc != 0):
            data.append(getModelStability(model, dist, acc))
        else:
            data.append(getData(model, dist))

    solos = []
    for model in data:
        curr = []
        M = model['res'].max()
        for i in range(1, M+1):
            loc = model.loc[model['res'] == i]
            if(len(loc) == 1):
                curr.append(loc['str'].values[0])
        solos.append(curr)

    dfAll = pd.DataFrame()
    for solo in solos:
        tmp = pd.DataFrame({'str':solo})
        dfAll = pd.concat([dfAll, tmp], ignore_index=True)

    dfAll.to_excel('SoloAll.xlsx', index=False)
    result = []

    for term in solos[0]:
        curr = True
        for i in range(1, len(solos)):
            if(not term in solos[i]):
                curr = False
                break
        if(curr):
            result.append(term)

    df = pd.DataFrame({'str':result})
    df.to_excel('SoloSummarization.xlsx', index=False)