# -*- codeing = utf-8 -*-
# @Time : 2021/12/9 14:45
# @Author : 王伊念
# File : lijie.py
# @Software : PyCharm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# %matplotlib inline
import json
import warnings

# warnings.filterwarnings('ignore')
#
# import seaborn as sns
#
# sns.set(color_codes=True)
#
# # 设置表格用什么字体
# font = {
#     'family': 'SimHei'
# }
# matplotlib.rc('font', **font)
#
# 导入数据
movies = pd.read_csv('D:\\我的大学\\大三上\\大型数据库\\大数据竞赛练习题\\pppp\\QX\\tmdb_5000_movies.csv')
credits = pd.read_csv('D:\\我的大学\\大三上\\大型数据库\\大数据竞赛练习题\\pppp\\QX\\tmdb_5000_credits.csv')

# # 查看movies中数据
# movies.head()
#
# # 查看movies中所有列名，以字典形式存储
# movies.columns
#
# ##查看creditss中数据
# creditss.head()
#
# # 查看creditss中所有列名，以字典形式存储,一共4个列名
# creditss.columns
#
# # 两个数据框中的title列重复了，删除credits中的title列，还剩3个列名
# del creditss['title']
#
# # movies中的id列与credits中的movie_id列实际上等同，可当做主键合并数据框
# full = pd.merge(movies, creditss, left_on='id', right_on='movie_id', how='left')
#
# # 某些列不在本次研究范围，将其删除
# full.drop(['homepage', 'original_title', 'overview', 'spoken_languages',
#            'status', 'tagline', 'movie_id'], axis=1, inplace=True)
#
# # 查看数据信息，每个字段数据量。
# full.info()
#
# full.isnull().any()

# print('电影信息数据集', movies.shape, ',演员信息数据集', credits.shape)
