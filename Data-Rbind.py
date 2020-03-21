#!/usr/bin/env python
# coding: utf-8

# In[28]:


'''
name:read_data
function：读入路径下所有的csv或excel文件，并将其合并成一个总表
input:文件存放路径(path)，该路径下最好只有csv或excel文件。数据类型(type),如果是excel文件，则传入type = "excel"；如果是csv文件，则传入
type = "csv"
output:读取的文件名字(filename)，此项主要是为了检查读取的文件是否正确。读取并合并后的总表（result）。
example： filename,result = read_data("C:/Users/lenovo/Desktop/网络爬虫/csv测试",type = "csv")
'''

import os
import pandas as pd
def read_data(path,type = "csv"):
    filename = []

    frames = []
    if type == "excel":
        for root,dirs,files in os.walk(dirt):
            for file in files:
                filename.append(os.path.join(root,file))
                df = pd.read_excel(os.path.join(root,file))
                frames.append(df)
    elif type == "csv":
        for root,dirs,files in os.walk(dirt):
            for file in files:
                filename.append(os.path.join(root,file))
                df = pd.read_csv(os.path.join(root,file))
                frames.append(df)
    else:
        print("数据类型不符合要求，请输入参数type = 'csv'或type = 'excel'")
    result = pd.concat(frames)
    return filename,result


# In[31]:


dirt = "C:/Users/lenovo/Desktop/网络爬虫/csv测试"
filename,result = read_data(dirt,type = "csv")

