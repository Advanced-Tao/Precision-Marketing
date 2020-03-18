#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from collections import defaultdict
import jieba
import random

'''
********************************************************************************************************
Precision-Marketing (Bi- LSTM).py
class:market

author: Andy
date:2020/3/18

market类的主要目的是分析新浪微博的博文数据（爬取的标签如“会不会得病”、“身体不舒服”等），分析博文中的担忧情绪。
因为有担忧情绪的人更有可能购买健康保险，所以这类人群是健康保险产品的潜在客户。

该类主要功能有以下三个：
1. 导入停用词库和词向量库(函数名：add_stopwords，read_vectors)
2. 构建embedding层（函数名：embedding_matrix） 
3. 将非结构化的文本数据处理成结构化的形式(函数名： word_cut,frequency，recoding，delete)

********************************************************************************************************
'''
class market():

'''
name: read_vectors
function（函数的功能）: 载入预训练的词向量库中的前topn个词向量
input:词向量库的存放路径(path), 导入的词语数目(topn)
output: 词向量的维度数（self.dim，如每个词语提取了300个特征，则dim=300）
提取的词语数量（self.max_words，如果提取10000个词向量，则self.max_words = 10000）
词语字典(self.word_index, 返回的结果是一个字典。每个词语对应一个整数编码
比如{"的"：1，"非常":2，"生病":3})
整数字典（self.index_word，每一个整数对应一个词语编码，该词典意义不大，已弃用。）
词向量(self.vectors，返回的结果是一个字典。每一个词语对应着dim个特征，如果dim=300，则每一个词都对应着一个1×300的向量，
比如{"的"：array[0.525421355,0.15235234,......],"非常":array[0.3535353111,0.3543636321,......]})
'''''

    def read_vectors(self,path, topn):  
        lines_num, dim = 0, 0
        vectors = {}
        iw = []
        wi = {}
        with open(path, encoding='utf-8', errors='ignore') as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    dim = int(line.rstrip().split()[1]) 
                    continue
                lines_num += 1
                tokens = line.rstrip().split(' ')
                vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
                iw.append(tokens[0]) # iw储存了所有的tokons[0]，意思是index_word
                if topn != 0 and lines_num >= topn:
                    break
        for i, w in enumerate(iw):
            wi[w] = i # wi是iw的反转，意思是word_index,用w来储存字符，用一个integer去给字符编码
        self.dim = dim
        self.max_words = topn
        self.word_index = wi
        self.index_word = iw
        self.vectors = vectors
        print("Load %s word vectors." % len(vectors))
'''
name: add_stopwords
function: 导入停用词库
input:停用词库的存放路径(path)
output: 停用词集合（self.stopwords，返回的结果是一个集合(set)，集合中储存了几百个停用词。
采用了改进后四川大学提供的停用词库，该词库中加入了保险中的特有名词，如“健康保险”）
'''
    def add_stopwords(self,path):
        stopwords = set()
        with open(path,'r',encoding = 'cp936') as file:
            for line in file:
                stopwords.add(line.strip())
        self.stopwords = stopwords
        print("Load %s stopwords" %len(stopwords))    
'''
name: add_stopwords
function: 构建Embedding矩阵，该矩阵维度数目为 词语数量 × 特征数量（比如10000×300），在神经网络中通过该层，
可以将每个词语编码成300个维度的密集向量
input: 无
output: Embedding矩阵（embedding_matrix）
'''
    def embedding_matrix(self):
        embedding_matrix = np.zeros((self.max_words,self.dim))
        for word,i in self.word_index.items():
            if i < self.max_words:
                embedding_vector = self.vectors.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        return embedding_matrix
'''
name: word_cut
function: 将博文分割成几个词语
input: 储存博文的文档（documents。该文档是一个不定长的list，长度为博文数量，但宽度未知（因为博文不定长），
比如[["今天胸口疼痛不舒服，自己会不会得病啊？"]["上班压力好大，长期下来积劳成疾怎么办"]......]
output: 按照一定规则，将博文切割成词语后的文档（texts，该文档仍然是以个不定长的list，不同词语之间以逗号分割，
比如[["今天,胸口,疼痛,不舒服，自己,会不会,得病,啊？"]["上班,压力,好大，长期下来,积劳成疾,怎么办"]......]）
'''
    def word_cut(self,documents):
        stopwords = self.stopwords
        texts = []
        for line in documents:
            words = ' '.join(jieba.cut(line)).split(' ') # 用空格去连接，连接后马上又拆分
            text = []
            for word in words:
                if (word not in stopwords) & (word != '')& (word != '\u3000')& (word != '\n')&(word != '\u200b'):
                    text.append(word)
            texts.append(text)
        self.docLength = len(documents)
        return(texts)
'''
name: frequency
function: 按照词语的出现频次过滤掉某些低频次
input: 切割成词语后的文档（texts）,允许的最低出现频率（freq，比如freq=5，意味着删掉在所有词语中出现次数 <= 5的词语）
output: 过滤后的文档(texts)
'''
    def frequency(self,texts,freq):
        frequency = defaultdict(int) # value为int
        for text in texts:
            for word in text:
                frequency[word] += 1
        texts = [[word for word in text if frequency[word] > freq] for text in texts]
        return(texts)
'''
name: recoding
function: 将词语编码成整数形式，如果词典（word_index）中没有该词语，则编码为-1
input: 过滤后的文档(texts)，词典（word_index）
output: 将词语按照整数编码后的文档(texts)
'''
    def recoding(self,texts,word_index):
        for i,sample in enumerate(texts):
            for j,word in enumerate(sample):
                if word not in word_index:
                    sample[j] = -1
                else:
                    sample[j] = word_index[word]
            texts[i] = sample
        return(texts)
'''
name: delete
function: 将文档中编码为-1的记录删去
input: 将词语按照整数编码后的文档(docs)
output: 删除了所有编码为-1的记录的文档（docs）
'''
    def delete(self,docs):
        for index in range(len(docs)):
            for i in range(len(docs[index])-1,-1,-1):
                if docs[index][i] == -1:
                    docs[index].pop(i)
        return docs
'''
name: random_pick
function: 对不担忧的样本做欠采样，比如不担忧的样本有936个，担忧的样本有300个，则欠采样的结果是从不担忧的936个样本里面随机选取300个
input: 需要欠采样的数据框（df），欠采样之后的样本数量（n）
output: 欠采样后的数据框(df)
'''
    def random_pick(self,df,n):
        rand = np.arange(0,(len(df)-1),1)
        random.shuffle(rand)
        rand = list(rand[:n])
        df = df.loc[rand,]
        return(df)
    
    

'''
因为预训练的词向量库中已预含词典，所以该函数被废弃
    def dictionary(self,docs):
        token_index ={}
        for sample in docs:
            for word in sample:
                if word not in token_index:
                    token_index[word] = len(token_index) + 1
        return(token_index)
'''

'''
因为预训练的词向量库中已预含词典，所以该函数被废弃
    def count(self,docs):
        token_length ={}
        for sample in docs:
            for word in sample:
                if word not in token_length:
                    token_length[word] = 1
                else:
                    token_length[word] += 1
        return(token_length)
'''   

'''
因为暂停了文本聚类项目，所以该函数被废弃
    def regroup(self,texts):
        new_texts = []
        for i,sentence in enumerate(texts):
            new_texts.append(" ".join(sentence))
        return(new_texts)
'''


# In[2]:


# 导入停用词库和词向量库
# 词向量库由北师大提供(github地址"https://github.com/Embedding/Chinese-Word-Vectors")，
# 停用词库由四川大学提供(github地址："https://github.com/fighting41love/funNLP/tree/master/data/%E5%81%9C%E7%94%A8%E8%AF%8D")

process = market()
process.add_stopwords("D:/Users/PYTHON/Precision-Marketing/stopwords.txt")
process.read_vectors("D:/NLP/sgns.target.word-word.dynwin5.thr10.neg5.dim300.txt",10000)


# In[3]:


# 构建词矩阵（由若干个词向量堆叠而成，如果导入了10000个词，每个词有300个特征，则矩阵维度为10000×300），
# 该步骤可以为下文嵌入keras的embedding层做铺垫
embedding_matrix = process.embedding_matrix()
embedding_matrix.shape


# In[16]:


# 导入数据，数据来源于微博，已经人工标注。数据可以从github上下载
# （github地址：https://github.com/Advanced-Tao/Precision-Marketing/master/关键词标签.xlsx）
os.chdir("D:/Users/PYTHON/Precision-Marketing")
df = pd.DataFrame()
num = 0
for i in range(10):
    df_temp = pd.read_excel("关键词标签.xlsx",sheet_name = i)
    df = df.append(df_temp)
    num += 1
print("一共读取了{}个sheet".format(num))
df[:2]


# In[17]:


# 对无担忧情绪（无买保险欲望）的标签(df_non_worry)做欠采样

df = df.loc[pd.notna(df["是否担忧（1=担忧，-1=完全不担忧，0=中性，有些担忧但不用买保险,2=疑似抑郁症）"]),]
df_worry = df[df["是否担忧（1=担忧，-1=完全不担忧，0=中性，有些担忧但不用买保险,2=疑似抑郁症）"] == 1]
df_worry.reset_index(drop = True,inplace = True)
df_non_worry = df[df["是否担忧（1=担忧，-1=完全不担忧，0=中性，有些担忧但不用买保险,2=疑似抑郁症）"] == -1]
df_non_worry.reset_index(drop = True,inplace = True)
df_non_worry = process.random_pick(df_non_worry,min(len(df_worry),len(df_non_worry)))
df_worry = df_worry[["博文","是否担忧（1=担忧，-1=完全不担忧，0=中性，有些担忧但不用买保险,2=疑似抑郁症）"]]
df_non_worry = df_non_worry[["博文","是否担忧（1=担忧，-1=完全不担忧，0=中性，有些担忧但不用买保险,2=疑似抑郁症）"]]
df_worry = df_worry.dropna()
df_non_worry = df_non_worry.dropna()


# In[18]:


# 合并欠采样后的无担忧标签和有担忧标签，得到类别数目平衡的数据框df_use

df_use = pd.concat([df_worry,df_non_worry])
df_use.reset_index(drop = True,inplace = True)
df_use = df_use.reindex(np.random.permutation(df_use.index))
df_use.head()


# In[7]:


# 自变量预处理

x_train = process.word_cut(df_use["博文"])
x_train = process.frequency(x_train,5)
x_train = process.recoding(x_train,process.word_index)
x_train = process.delete(x_train)

import keras
import tensorflow
from keras import preprocessing

max_len = 50
x_train = preprocessing.sequence.pad_sequences(x_train,maxlen = max_len) # 将博文50个字符以后的部分舍弃
x_train.shape


# In[9]:


# 因变量预处理

y_train = df_use[["是否担忧（1=担忧，-1=完全不担忧，0=中性，有些担忧但不用买保险,2=疑似抑郁症）"]]
y_train["是否担忧（1=担忧，-1=完全不担忧，0=中性，有些担忧但不用买保险,2=疑似抑郁症）"] = y_train["是否担忧（1=担忧，-1=完全不担忧，0=中性，有些担忧但不用买保险,2=疑似抑郁症）"].apply(lambda v: 0 if v == -1 else 1)
y_in = len(y_train)
y_train = np.array(y_train)
y_train = y_train.reshape(y_in)
y_train[:5]


# In[11]:


# 构建神经网络模型

from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding,LSTM,Bidirectional,Dropout

max_features = 10000
max_len = 50

model = Sequential()
model.add(Embedding(max_features,300,input_length = max_len,mask_zero = True)) # 遇到0，就不会反向传播更新权重
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(20),merge_mode = 'concat'))
model.add(Dense(1,activation = 'sigmoid'))
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics = ['acc'])
model.summary()


# In[ ]:


# 训练神经网络模型

history = model.fit(x_train,
                    y_train,
                    epochs = 10,
                    batch_size =128, # batch_size越大越好，但是太大会影响计算效率
                    validation_split= 0.3)


# In[14]:


# 保存神经网络

# model.save("Bi-LSTM(加入预训练的词向量库).h5")

