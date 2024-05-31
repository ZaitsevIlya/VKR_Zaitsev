## abbreviations.py

Данный модуль предназначен для расшифровки сокращений при помощи модели **sberbank-ai/ruBert-base**
Расшифровываются сокращения вида «сокр.» и «сокр-ие»


```
unmasking(my_str, logs = False, top_k = 1000)
```
- my_str — строка для расшифровки сокращений
- logs — нужно ли логирование в процессе расшифровки (по умолчанию False, не рекомендуется)
- top_k — количество слов в Unmasking (чем больше, тем больше вероятность, что сокращение будет расшифровано)

## group-metrics.py — расчёт межгрупповых метрик по фильтру (prompt)

```
def by_prompt(data, prompt, save_path)
```
- data — данные в формате **pandas.DataFrame** с колонками str, res и vec (исходная строка, номер группы и вектор соотв.)
- prompt — строка для поиска подстроки
- save_path — путь для сохранения файлов в формате **save_path/prompt.xlsx**


## clustering.py (кластеризация, составление эталона)

```
def UseAgglomerativeClustering(values, prefix, distance_threshold, embeddings)
```

- values — массив строк для кластеризации
- prefix — префикс файла для выгрузки. В зависимости от prefix (если не указан embeddings) будет выбрана модель векторизации (см. ниже "Модели векторизации")
- distance_threshold — порог расстояния для кластеризации
- embeddings — массив векторов Torch (в том же порядке, что и values). Если None, то векторизация будет выполнена автоматически

Внимание! Если embeddings не указан (== None), то будет произведена автоматическая векторизация при помощи выбранной в prefix модели. Модель векторизации будет предварительно скачана.

### Модели векторизации
1. sentence-transformers/all-MiniLM-L6-v2
2. sentence-transformers/multi-qa-MiniLM-L6-cos-v1
3. sentence-transformers/all-mpnet-base-v2
4. ai-forever/ruElectra-large