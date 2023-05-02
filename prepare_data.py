#!/usr/bin/python
#-*-coding:utf-8-*-

import pandas as pd
import random

def concatenate(title, keyword, abstract):
    title = f"{title}"
    if type(keyword) == float:
        # print("Keyword: ", keyword)
        keyword = ""
    else:
        keyword = f"{keyword}."
    abstract = f"{abstract}"
    
    return title + "$" + keyword + "$" + abstract

dataset_name = 'covid'
df = pd.read_excel("dataset.xlsx")
df.dropna(subset=["Contextual", "Article title", "Article abstract"], inplace=True)
df["Concatenated"] = df.apply(
        lambda x: concatenate(x["Article title"], x["Article keywords"], x["Article abstract"]), axis=1,
    )
df = df.astype({"Contextual": "int32"})
df = df[["Concatenated", "Contextual"]]
df.reset_index(drop=True, inplace=True)

sentences = df["Concatenated"].to_list()
dataset_name = "covid"
labels = df["Contextual"].to_list()

train_or_test_list = ["train"]*len(sentences)
split_index = int(0.8*len(sentences))
for i in range(split_index, len(sentences)):
    train_or_test_list[i] = "test"
random.shuffle(train_or_test_list)



meta_data_list = []

for i in range(len(sentences)):
    meta = str(i) + '\t' + train_or_test_list[i] + '\t' + str(labels[i])
    meta_data_list.append(meta)

meta_data_str = '\n'.join(meta_data_list)

f = open('data/' + dataset_name + '.txt', 'w')
f.write(meta_data_str)
f.close()

corpus_str = '\n'.join(sentences)

f = open('data/corpus/' + dataset_name + '.txt', 'w')
f.write(corpus_str)
f.close()