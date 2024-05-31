import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os
from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    NamesExtractor,

    Doc
)
from transformers import BertForMaskedLM,BertTokenizer, pipeline

#Initializing

model = BertForMaskedLM.from_pretrained('sberbank-ai/ruBert-base', cache_dir="../hugging_cache/")
tokenizer = BertTokenizer.from_pretrained('sberbank-ai/ruBert-base', do_lower_case=False, cache_dir="../hugging_cache/")
unmasker = pipeline('fill-mask', model=model,tokenizer=tokenizer)
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)

#Functions

def masking(text):
  doc = Doc(text)
  doc.segment(segmenter)
  doc.tag_morph(morph_tagger)
  for token in doc.tokens:
    token.lemmatize(morph_vocab)
  doc.parse_syntax(syntax_parser)
  doc.tag_ner(ner_tagger)
  for span in doc.spans:
    span.normalize(morph_vocab)
  mask = []
  obj = []
  mask_ind = []
  mask_words = []
  isAbbrev = False
  ind = 1

  for i in range(len(doc.tokens)-1, -1, -1):
    if(doc.tokens[i].pos == 'PUNCT'):
      if(doc.tokens[i].text == '.'):
        isAbbrev = True
      continue
    if(isAbbrev):
      mask.insert(0, '[MASK]')
      mask_words.append('')
      ind += 1
      isAbbrev = False
    else:
      if('-' in doc.tokens[i].text):
        mask.insert(0, '[MASK]')
      else:
        mask.insert(0, doc.tokens[i].text)
    obj.insert(0, doc.tokens[i].text)

  return (mask, obj)

def umnask_one(mask, obj, top_k = 1000):
  abbrevs = []
  abbrevs_inds = []
  for i in range(len(mask)):
    if(mask[i] == '[MASK]'):
      abbrevs.append(obj[i])
      abbrevs_inds.append(i)

  curr = 'Должность: '
  hasMask = False
  for i in range(len(mask)):
    if(mask[i] == '[MASK]' and hasMask):
        continue
    if(mask[i] == '[MASK]'):
      hasMask = True
    curr = curr + mask[i] + ' '

  curr = curr[:-1]
  res = unmasker(curr, top_k=top_k)
  found = False
  unmask = None
  abbrev_ind = 0
  for token in res:
    if(found):
      break
    current = token['token_str']
    for i in range(len(abbrevs)):
      if(found):
        break
      if('-' in abbrevs[i]):
        ind = abbrevs[i].find('-')
        before = abbrevs[i][:ind]
        after = abbrevs[i][ind+1:]
        if (current[:len(before)] == before and current[-len(after):] == after):
          unmask = current
          abbrev_ind = i
          found = True
      else:
        if (current[:len(abbrevs[i])] == abbrevs[i]):
          unmask = current
          abbrev_ind = i
          found = True
  if(unmask is None):
    unmask = abbrevs[0]
  mask[abbrevs_inds[abbrev_ind]] = unmask

def glue(obj):
  res = ''
  for i in obj:
    res = res + i + ' '
  res = res[:-1]
  return res

def unmasking(text, logger = False, abbrevs_old = None, abbrevs_new = None, top_k = 1000):
  mask, obj = masking(text)
  #print(mask)
  hasAbbrevs = False
  if(logger):
    if('[MASK]' in mask):
      hasAbbrevs = True
      print('=====================================')
      print('Abbreviation is found:', text)
      if(not abbrevs_old is None):
        abbrevs_old.append(glue(obj))
  while('[MASK]' in mask):
    umnask_one(mask, obj)
  result = ''
  for i in mask:
    result = result + i + ' '
  result = result[:-1]
  if(logger and hasAbbrevs):
      print('Abbreviation decoded:', result)
      if(not abbrevs_new is None):
        abbrevs_new.append(result)
  return result

def process_df_abbrevs(df, column='str', new_column='no_abbs', need_save=False, filename=None):
    no_abbs = []
    i = 1
    j = 1
    k = -1
    for val in df[column].values:
      k = k + 1
      print('Iter:', k)
      i = (i + 1)%250
      if(i == 0):
        j+=1
        print(j, 'iteration!%%%%%%%%%%%%%%%%%%%')
      try:
        res = unmasking(val, True)
        no_abbs.append(res)
      except:
        print('Exception:', val)
        no_abbs.append('')
    print('That\'s all')
    df[new_column] = no_abbs
    if (need_save and not filename is None):
      df.to_excel(filename, index=False)
    return df

def process_df_1(df):
  df.pop(1)
  df.pop(3)
  df.pop(0)
  df.pop(2)
  df = df.dropna()
  df[4] = df[4].str.lower()
  df = df.drop_duplicates(keep='first', ignore_index=True, subset=4)
  df = df.rename(columns={4: 'str'})
  df.to_excel('./all-data.xlsx', index=False)
  return df

def process_df_3(df):
  df.pop('str')
  df.pop('no_abbs')
  df = df.dropna()
  df['new_vals'] = df['new_vals'].str.lower()
  df = df.drop_duplicates(keep='first', ignore_index=True)
  df = df.rename(columns={'new_vals': 'str'})
  df.to_excel('./all-data-no_abbs.xlsx', index=False)
  return df

def process_df_2(df):
  new_vals = []
  for val in df['no_abbs'].values:
    try:
      tmp = val.replace('.', ' ')
      tmp = tmp.replace('-', ' ')
      tmp = tmp.replace('_', ' ')
      while(tmp.count('  ') != 0):
        tmp = tmp.replace('  ', ' ')
      while len(tmp) > 0 and tmp[0] == ' ':
        tmp = tmp[1:]
      while(len(tmp) > 0 and tmp[-1] == ' '):
        tmp = tmp[:-1]
      new_vals.append(tmp)
    except:
      new_vals.append('')
  df['new_vals'] = new_vals
  return df


def read_file(file, sep='#'):
  with open(file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter=sep)
    df = pd.DataFrame(reader)
  return df

def replace_trains(df):
  processed = []
  for val in df[4].values:
    tmp = ''
    words = val.split(' ')
    for w in words:
      if(len(w) > 10):
        res = w.replace('-', ' ')
      else:
        res = w
      tmp = tmp + res + ' '
    processed.append(tmp[:-1])
  df[4] = processed
  return df