import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
import time
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm

def by_prompt(data, prompt, result_path = None):

  groups = []

  for i in range(1, data['res'].max() + 1):
    find = data[data['res'] == i]
    if(len(find) > 2):
      groups.append(i)

  res = []
  res_data = []

  with tqdm(total = data['res'].max()+1) as pbar:
    for i in range(1, data['res'].max()+1):
      pbar.update(1)
      if(not i in groups):
        continue
      find = data[data['res'] == i]
      for val in find['str'].values:
        if(prompt in val):
          res.append(i)
          break

  with tqdm(total = len(res)*(len(res)-1)//2 - len(res)) as pbar:
    for i in range(len(res)):
      for j in range(i+1, len(res)):
        mi = 1e9
        ma = -1
        mi_ex1 = ''
        mi_ex2 = ''
        ma_ex1 = ''
        ma_ex2 = ''
        find_i = data[data['res'] == res[i]]
        find_j = data[data['res'] == res[j]]
        for val_i in find_i.values:
          for val_j in find_j.values:
            rho = 1-dot(val_i[2], val_j[2])
            if(mi > rho):
              mi = rho
              mi_ex1 = val_i[0]
              mi_ex2 = val_j[0]
            if(ma < rho):
              ma = rho
              ma_ex1 = val_i[0]
              ma_ex2 = val_j[0]
        res_data.append({"i":int(res[i]), "j":int(res[j]), "min":float(mi), "max":float(ma), "min_ex1":str(mi_ex1), "min_ex2":str(mi_ex2), "max_ex1":str(ma_ex1), "max_ex2":str(ma_ex2)})
        pbar.update(1)
  pd_data = pd.DataFrame(res_data)
  if(not result_path is None):
    pd_data.to_excel(result_path + prompt + '.xlsx')
  return pd_data