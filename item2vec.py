import os
import pandas as pd
from datetime import datetime
import S3_info
import jieba
from opencc import OpenCC
from gensim.models import word2vec
import numpy as np


#get sales_data with wrangle
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d")
directory = "sales_data_"+dt_string
path = os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', directory)
data = pd.read_csv(path+'/sales_data.csv')

text_num = 0
with open(path+'/products.txt', 'w+', encoding='utf-8') as f:
    for text in data.stock_description.unique():
        f.write(text+'\n')
        text_num += 1
        if text_num % 100 == 0:
            print('{} products processed.'.format(text_num))
    f.close()
    print('{} products processed.'.format(text_num))

# Initial
cc = OpenCC('s2t')


# Tokenize
with open(path+'/products_seg.txt', 'w', encoding='utf-8') as new_f:
    with open(path+'/products.txt', 'r', encoding='utf-8') as f:
        for times, data in enumerate(f, 1):
            if times % 50 == 0 : print('data num:', times)
            data = cc.convert(data)
            data = jieba.cut(data)
            data = [word for word in data if word != ' ']
            data = ' '.join(data)

            new_f.write(data)

# Settings
seed = 666
sg = 1
window_size = 10
vector_size = 5
min_count = 0
workers = 8
epochs = 5
batch_words = 100

train_data = word2vec.LineSentence(path+'/products.txt')
model = word2vec.Word2Vec(
    sentences=train_data,
    vector_size=vector_size,
    min_count=min_count,
    workers=workers,
    epochs=epochs,
    window=window_size,
    sg=sg,
    seed=seed,
    batch_words=batch_words
)

model.save(os.path.join(os.path.dirname(os.path.abspath(__file__))+'/', 'model')+'/word2vec.model')

# stocklens_stock_embedding.csv
word_vectors = model.wv
stock_embedding = pd.DataFrame(zip(word_vectors.index_to_key,word_vectors.vectors),columns=['word','vector'])
stock_embedding['vector'] = stock_embedding['vector'].apply(lambda x : np.array2string(x, separator=', '))
stock_embedding.to_csv(path+'/stocklens_stock_embedding.csv')