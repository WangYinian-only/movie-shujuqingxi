
import datetime
import os
import random
import time
import traceback

import jieba
import pandas as pd

import numpy as np
import seaborn as sns

from flask import Flask, render_template, request, jsonify, send_from_directory

from matplotlib import pyplot as plt
import collections  # 词频统计库

from matplotlib.ticker import MultipleLocator
from sqlalchemy import create_engine
import wordcloud  # 词云展示库
from PIL import Image  # 图像处理库
import matplotlib.pyplot as plt  # 图像展示库
from wordcloud import WordCloud


def to_str(key,df_tmdb_5000_movies_vote_0):
    str_list=[]
    num=df_tmdb_5000_movies_vote_0.shape[0]
    for i in range(num):
        dictionary=eval(df_tmdb_5000_movies_vote_0[key].iloc[i])#eval将字符串转换为字典
        str=""
        for j in dictionary:
            str=str+j["name"]+","
        str_list.append(str)
    df_tmdb_5000_movies_vote_0[key]=str_list
    print(df_tmdb_5000_movies_vote_0[key].head(10))

def data_first():
    tmdb_5000_movies="static/data/tmdb_5000_movies.csv"
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    #显示宽度
    pd.set_option('display.width', None)
    df_tmdb_5000_movies = pd.read_csv(tmdb_5000_movies)
    print(df_tmdb_5000_movies.head(3))
    #print(df_tmdb_5000_movies[df_tmdb_5000_movies['release_date']!=df_tmdb_5000_movies['release_date']])#输出指定列为控制的行
    #print(df_tmdb_5000_movies[df_tmdb_5000_movies['runtime']!=df_tmdb_5000_movies['runtime']])#输出指定列为控制的行#利用任何两个控制都不相等
    #print(df_tmdb_5000_movies['id'].drop_duplicates())#输出不同的id行
    #print(df_tmdb_5000_movies[df_tmdb_5000_movies.isnull().values == True].drop_duplicates())#将缺省行输出#drop_duplicates()将重复数据去掉，因为一行中或许有多个缺省值
    #df_tmdb_5000_movies['release_date'] = df_tmdb_5000_movies['release_date'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d')))
    #有比1970年小的年份，无法使用time.mktime转换
    print(df_tmdb_5000_movies['revenue'].describe())#票房
    print(df_tmdb_5000_movies['budget'].describe())#预算
    print(df_tmdb_5000_movies['popularity'].describe())#受欢迎程度
    print(df_tmdb_5000_movies['vote_average'].describe())#评分的描述
    print(df_tmdb_5000_movies['vote_count'].describe())#评论人数的描述
    df_tmdb_5000_movies_vote=df_tmdb_5000_movies[df_tmdb_5000_movies['vote_count'] >= 100]#评论人数小于100的去掉
    print(df_tmdb_5000_movies_vote.shape)
    df_tmdb_5000_movies_vote_0=df_tmdb_5000_movies_vote[df_tmdb_5000_movies_vote['revenue']>0]
    print(df_tmdb_5000_movies_vote_0.shape)
    df_tmdb_5000_movies_vote_0 = df_tmdb_5000_movies_vote_0[df_tmdb_5000_movies_vote_0['budget'] > 0]
    print(df_tmdb_5000_movies_vote_0.shape)
    df_tmdb_5000_movies_vote_0 = df_tmdb_5000_movies_vote_0[df_tmdb_5000_movies_vote_0['popularity'] > 0]
    print(df_tmdb_5000_movies_vote_0.shape)
    df_tmdb_5000_movies_vote_0 = df_tmdb_5000_movies_vote_0[df_tmdb_5000_movies_vote_0['vote_average'] > 0]
    num=df_tmdb_5000_movies_vote_0.shape[0]
    to_str('genres', df_tmdb_5000_movies_vote_0)
    to_str('keywords', df_tmdb_5000_movies_vote_0)
    to_str('production_companies', df_tmdb_5000_movies_vote_0)
    to_str('production_countries', df_tmdb_5000_movies_vote_0)
    to_str('spoken_languages', df_tmdb_5000_movies_vote_0)
    print(df_tmdb_5000_movies_vote_0.head(10))
    df_tmdb_5000_movies_vote_0.to_csv('clean_df_tmdb_5000_movies.csv',encoding='utf_8_sig',index=None)

def plot_bar_bin():
    clean_tmdb_5000_movies = "static/data/clean_df_tmdb_5000_movies.csv"
    # 电影分类
    # 统计分类列表
    clean_df_tmdb_5000_movies = pd.read_csv(clean_tmdb_5000_movies)
    temp_list = clean_df_tmdb_5000_movies["genres"].str.split(",").tolist()
    genre_list = list(set([i for j in temp_list for i in j]))
    # 构造全为0的数组
    zeros_df = pd.DataFrame(np.zeros((clean_df_tmdb_5000_movies.shape[0], len(genre_list))), columns=genre_list)
    #print(zeros_df)
    # 给每个顶电影出现的位置赋值为1
    for i in range(clean_df_tmdb_5000_movies.shape[0]):
        # zeros_df.loc[0,["Sci-fi","Mucical"]]=1
        zeros_df.loc[i, temp_list[i]] = 1

    #print(zeros_df.head(3))
    # 统计每个分类电影数量和
    genre_count = zeros_df.sum(axis=0)
    #print(genre_count)
    # 排序
    genre_count = genre_count.sort_values()
    genre_count_clean = genre_count[:-1]
    return (genre_count_clean)
    # _x = genre_count_clean.index
    # _y = genre_count_clean.values
    # # 设置图形的大小
    # # 直方图
    # plt.figure(figsize=(20, 8), dpi=80)
    # plt.bar(range(len(_x)), _y)
    # plt.xticks(range(len(_x)), _x)
    # plt.show()
    # # 饼状图
    # plt.figure(figsize=(20, 8), dpi=80)
    # labels = genre_count_clean.index
    # sizes = genre_count_clean.values
    # explode = (0, 0.1, 0, 0)  # 0.1表示将Hogs那一块凸显出来
    # plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)  # startangle表示饼图的起始角度
    # plt.axis('equal')  # 加入这行代码即可！
    # plt.show()
    # # 折线图
    # plt.figure(figsize=(20, 8), dpi=80)
    # plt.plot(range(len(_x)), _y)
    # plt.xticks(range(len(_x)), _x)
    # plt.show()

def create_0():
    list=[0,0,0,0,0
        ,0,0,0,0,0
        ,0,0,0,0,0
        ,0,0,0]
    return list

def combination():
    #budget预算revenue收入
    clean_tmdb_5000_movies = "static/data/clean_df_tmdb_5000_movies.csv"
    # 电影分类
    # 统计分类列表
    clean_df_tmdb_5000_movies = pd.read_csv(clean_tmdb_5000_movies)
    genre_count_clean=plot_bar_bin()
    _type=genre_count_clean.index
    budget_list=[]#预算
    budget_list=create_0()#初始值设为0
    revenue_list=[]#收入
    revenue_list=create_0()
    num=clean_df_tmdb_5000_movies.shape[0]#电影数目
    print(len(_type))
    for i in range(len(_type)):
        for j in range(num):
            if(_type[i] in clean_df_tmdb_5000_movies["genres"][j]):
                budget_list[i]=budget_list[i]+clean_df_tmdb_5000_movies["budget"][j]
                revenue_list[i]=revenue_list[i]+clean_df_tmdb_5000_movies["revenue"][j]
    print(budget_list)
    print(revenue_list)
    _x = genre_count_clean.index
    _y_budget = budget_list
    _y_revenue = revenue_list
    # 设置图形的大小
    # 直方图
    plt.figure(figsize=(20, 8), dpi=80)
    plt.bar(range(len(_x)), _y_budget)
    plt.plot(range(len(_x)), _y_revenue)
    plt.xticks(range(len(_x)), _x)
    plt.show()

def word_clound():#只是单个单词不太好
    # budget预算revenue收入
    clean_tmdb_5000_movies = "static/data/clean_df_tmdb_5000_movies.csv"
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 显示宽度
    pd.set_option('display.width', None)
    clean_df_tmdb_5000_movies = pd.read_csv(clean_tmdb_5000_movies)
    num = clean_df_tmdb_5000_movies.shape[0]
    text = ""
    for i in range(num):
        try:
            text = text + clean_df_tmdb_5000_movies["keywords"][i]
        except:
            print("NaN")
    # os.path.join()函数: 连接两个或者更多的路径名组件
    # 加入（encoding='gb18030', errors='ignore'）是为了防止出现解码错误，是可以省略的，但省略后如出现错误，可查阅“参考文献[1]”
    wc = WordCloud(scale=1, max_font_size=100)
    # 词云参数设置
    wc.generate(text)
    # genarate  v.生成;   Python中称为使用生成器
    plt.imshow(wc, interpolation='bilinear')
    # 显示图像
    # bilinear  adj.双直线的；双线性的；双一次性的;
    plt.axis('off')
    # 隐藏坐标轴
    plt.tight_layout()
    # tight_layout会自动调整子图参数，使之填充整个图像区域。
    # tight adj. 紧的；紧身的；挤满的；layout n.排版；布局；设计
    plt.savefig('tu1.png', dpi=300)
    # 保存词云图，分辨率为300，也可以用 wc.to_file('1900_basic.png')
    plt.show()
    # plt.imshow()函数负责对图像进行处理，并显示其格式
    # plt.show()则是将plt.imshow()处理后的函数显示出来。

def timu_5_3_1():
    clean_tmdb_5000_movies = "static/data/clean_df_tmdb_5000_movies.csv"
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 显示宽度
    pd.set_option('display.width', None)
    clean_df_tmdb_5000_movies = pd.read_csv(clean_tmdb_5000_movies)
    # 准备数据
    runtime_data = clean_df_tmdb_5000_movies["runtime"].tolist()
    runtime_data.sort()
    _y_count=[]
    for i in range(len(set(runtime_data))):
        _y_count.append(0)
    flag=0
    for i in set(runtime_data):
        _y_count[flag]=runtime_data.count(i)
        flag=flag+1
    _x = set(runtime_data)
    print(_x)
    print(_y_count)
    # 设置图形的大小
    # 直方图
    plt.figure(figsize=(20, 8), dpi=80)
    plt.bar(range(len(_x)), _y_count)
    plt.xticks(range(len(_x)), _x)
    x_major_locator = MultipleLocator(10)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(10)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.show()

def timu_5_3_2():
    clean_tmdb_5000_movies = "static/data/clean_df_tmdb_5000_movies.csv"
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 显示宽度
    pd.set_option('display.width', None)
    clean_df_tmdb_5000_movies = pd.read_csv(clean_tmdb_5000_movies)
    num = clean_df_tmdb_5000_movies.shape[0]
    #将日期转换为月份
    for i in range(num):
        clean_df_tmdb_5000_movies['release_date'][i] = clean_df_tmdb_5000_movies['release_date'][i].split("-")[1]
    fig = plt.figure(figsize=(8, 6))
    x = list(range(1, 13))
    y1 = clean_df_tmdb_5000_movies.groupby('release_date').revenue.size()
    y2 = clean_df_tmdb_5000_movies.groupby('release_date').revenue.mean()  # 每月单片平均票房

    # 左轴
    ax1 = fig.add_subplot(1, 1, 1)
    plt.bar(x, y1, color='b', label='Number of films')
    plt.grid(False)
    ax1.set_xlabel('month')  # 设置x轴label ,y轴label
    ax1.set_ylabel('Number of films', fontsize=16)
    ax1.legend(loc=2, fontsize=12)

    # 右轴
    ax2 = ax1.twinx()
    plt.plot(x, y2, 'ro--', label='Monthly average box office')
    ax2.set_ylabel('Monthly average box office', fontsize=16)
    ax2.legend(loc=1, fontsize=12)

    plt.show()

#获取制片公司的列表
def get_production_companies():
    # production_companies制片公司
    clean_tmdb_5000_movies = "static/data/clean_df_tmdb_5000_movies.csv"
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 显示宽度
    pd.set_option('display.width', None)
    clean_df_tmdb_5000_movies = pd.read_csv(clean_tmdb_5000_movies)
    temp_list = clean_df_tmdb_5000_movies["production_companies"].str.split(",").tolist()
    # Phantom Sound,Vision  The Sisterhood of the Traveling Pants 2
    # Artisan Entertainment  The Way of the Gun
    # Vincent Gallo  Buffalo '66
    genre_list = list(set([i for j in temp_list for i in j]))
    # 构造全为0的数组
    zeros_df = pd.DataFrame(np.zeros((clean_df_tmdb_5000_movies.shape[0], len(genre_list))), columns=genre_list)
    # print(zeros_df)
    # 给每个制片公司出现的位置赋值为1
    for i in range(clean_df_tmdb_5000_movies.shape[0]):
        # zeros_df.loc[0,["Sci-fi","Mucical"]]=1
        zeros_df.loc[i, temp_list[i]] = 1

    # print(zeros_df.head(3))
    # 统计每个制片公司数量和
    genre_count = zeros_df.sum(axis=0)
    # print(genre_count)
    # 排序
    genre_count = genre_count.sort_values()
    genre_count_clean = genre_count[:-1]
    genre_count_clean=genre_count_clean[genre_count_clean.values>=10]
    print(genre_count_clean)
    return (genre_count_clean)

#获取top票房分布
def timu_5_5_1():
    clean_tmdb_5000_movies = "static/data/clean_df_tmdb_5000_movies.csv"
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 显示宽度
    pd.set_option('display.width', None)
    clean_df_tmdb_5000_movies = pd.read_csv(clean_tmdb_5000_movies)
    production_companies_list = (get_production_companies().index)
    revenue_list = []  # 收入
    num = clean_df_tmdb_5000_movies.shape[0]  # 电影数目
    for i in range(len(production_companies_list)):
        revenue_list.append(0)
    for i in range(len(production_companies_list)):
        for j in range(num):
            if (production_companies_list[i] in clean_df_tmdb_5000_movies["production_companies"][j]):
                revenue_list[i] = revenue_list[i] + clean_df_tmdb_5000_movies["vote_average"][j]
    plt.figure(figsize=(20, 8), dpi=80)
    map= {}#利用字典进行排序
    for i in range(len(production_companies_list)):
        map[production_companies_list[i]]=revenue_list[i]
    sorted(map.items(), key=lambda item: item[1])
    _x=list(map.keys())[-10:]
    _y_count=list(map.values())[-10:]
    print(_y_count)
    # 直方图
    plt.figure(figsize=(20, 8), dpi=80)
    plt.bar(range(len(_x)), _y_count)
    plt.xticks(range(len(_x)), _x)
    plt.show()
    labels = production_companies_list[-20:]
    sizes = revenue_list[-20:]
    explode = (0, 0.1, 0, 0)  # 0.1表示将Hogs那一块凸显出来
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)  # startangle表示饼图的起始角度
    plt.axis('equal')  # 加入这行代码即可！
    plt.show()

def timu_5_6_1and2():
    clean_tmdb_5000_movies = "static/data/clean_df_tmdb_5000_movies.csv"
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 显示宽度
    pd.set_option('display.width', None)
    clean_df_tmdb_5000_movies = pd.read_csv(clean_tmdb_5000_movies)
    # 创建数据框
    original_df = pd.DataFrame()
    original_df['keywords'] =clean_df_tmdb_5000_movies['keywords'].str.contains('based on').map(lambda x: 1 if x else 0)
    #print(clean_df_tmdb_5000_movies['keywords'].str.contains('based on').map(lambda x: 1 if x else 0))
    #contains判断是否有子字符串，返回布尔类型，后面的map是将布尔类型转换为1，0表示
    original_df['profit'] = clean_df_tmdb_5000_movies['revenue']#收入
    original_df['budget'] = clean_df_tmdb_5000_movies['budget']#预算

    # 计算
    novel_cnt = original_df['keywords'].sum()  # 改编作品数量
    original_cnt = original_df['keywords'].count() - original_df['keywords'].sum()  # 原创作品数量
    # 按照 是否原创 分组
    original_df = original_df.groupby('keywords', as_index=False).mean()  # 注意此处计算的是利润和预算的平均值
    # 增加计数列
    original_df['count'] = [original_cnt, novel_cnt]
    #print(original_df)
    # 计算利润率
    original_df['profit_rate'] = (original_df['profit'] / original_df['budget']) * 100

    # 修改index
    original_df.index = ['original', 'based_on_novel']
    # 计算百分比
    original_pie = original_df['count'] / original_df['count'].sum()

    # 绘制饼图
    original_pie.plot(kind='pie', label='', startangle=90, shadow=False, autopct='%2.1f%%', figsize=(8, 8))
    plt.title('Original VS Adaptation', fontsize=20)
    plt.legend(loc=2, fontsize=10)
    plt.savefig('改编VS原创.png', dpi=300)
    plt.show()

    x = original_df.index
    y1 = original_df.budget
    y2 = original_df.profit_rate

    fig = plt.figure(figsize=(8, 6))

    # 左轴
    ax1 = fig.add_subplot(1, 1, 1)
    plt.bar(x, y1, color='b', label='Average budget', width=0.25)
    plt.xticks(rotation=0, fontsize=12)  # 更改横坐标轴名称
    ax1.set_xlabel('Original VS Adaptation')  # 设置x轴label ,y轴label
    ax1.set_ylabel('Average budget', fontsize=16)
    ax1.legend(loc=2, fontsize=10)

    # 右轴
    # 共享x轴，生成次坐标轴
    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'ro-.', linewidth=5, label='Average profit margin')
    ax2.set_ylabel('Average profit margin', fontsize=16)
    ax2.legend(loc=1, fontsize=10)  # loc=1,2,3,4分别表示四个角，和四象限顺序一致

    # 将利润率坐标轴以百分比格式显示
    import matplotlib.ticker as mtick
    fmt = '%.1f%%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax2.yaxis.set_major_formatter(yticks)

    plt.savefig('改编VS原创的预算以及利润率.png', dpi=300)
    plt.show()

def timu_5_7_1():
    clean_tmdb_5000_movies = "static/data/clean_df_tmdb_5000_movies.csv"
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 显示宽度
    pd.set_option('display.width', None)
    clean_df_tmdb_5000_movies = pd.read_csv(clean_tmdb_5000_movies)
    # 计算相关系数矩阵
    revenue_corr = clean_df_tmdb_5000_movies[
        ['runtime', 'popularity', 'vote_average', 'vote_count', 'budget', 'revenue']].corr()

    sns.heatmap(
        revenue_corr,
        annot=True,  # 在每个单元格内显示标注
        cmap="Blues",  # 设置填充颜色：黄色，绿色，蓝色
        #             cmap="YlGnBu", # 设置填充颜色：黄色，绿色，蓝色
        #             cmap="coolwarm", # 设置填充颜色：冷暖色
        cbar=True,  # 显示color bar
        linewidths=0.5,  # 在单元格之间加入小间隔，方便数据阅读
        # fmt='%.2f%%',  # 本来是确保显示结果是整数（格式化输出），此处有问题
    )
    plt.savefig('票房相关系数矩阵.png', dpi=300)
    plt.show()

def timu_5_7_2():
    clean_tmdb_5000_movies = "static/data/clean_df_tmdb_5000_movies.csv"
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 显示宽度
    pd.set_option('display.width', None)
    clean_df_tmdb_5000_movies = pd.read_csv(clean_tmdb_5000_movies)
    temp_list = clean_df_tmdb_5000_movies["genres"].str.split(",").tolist()
    genre_list = list(set([i for j in temp_list for i in j]))
    genre_list = genre_list[1:]
    # 创建数据框-电影类型
    genre_df = pd.DataFrame()

    # 对电影类型进行one-hot编码
    for i in genre_list:
        # 如果包含类型 i，则编码为1，否则编码为0
        genre_df[i] = clean_df_tmdb_5000_movies['genres'].str.contains(i).apply(lambda x: 1 if x else 0)
    # 将数据框的索引变为年份
    num = clean_df_tmdb_5000_movies.shape[0]
    for i in range(num):
        clean_df_tmdb_5000_movies['release_date'][i] = clean_df_tmdb_5000_movies['release_date'][i].split("-")[0]
    genre_df.index = clean_df_tmdb_5000_movies['release_date']
    print(genre_df.head(10))
    # 加上属性列budget，revenue，popularity，vote_count
    revenue_df = pd.concat([genre_df.reset_index(), clean_df_tmdb_5000_movies['revenue']
                               , clean_df_tmdb_5000_movies['budget']
                               , clean_df_tmdb_5000_movies['popularity']
                               , clean_df_tmdb_5000_movies['vote_count']], axis=1)
    print(revenue_df.head(10))
    # 绘制散点图
    fig = plt.figure(figsize=(17, 5))

    # # 学习seaborn参考：https://www.jianshu.com/p/c26bc5ccf604
    ax1 = plt.subplot(1, 3, 1)
    ax1 = sns.regplot(x='budget', y='revenue', data=revenue_df)

    # marker: 'x','o','v','^','<'
    # jitter:抖动项，表示抖动程度
    ax1.text(1.6e8, 2.2e9, 'r=0.7', fontsize=16)
    plt.title('budget-revenue-scatter', fontsize=20)
    plt.xlabel('budget', fontsize=16)
    plt.ylabel('revenue', fontsize=16)

    ax2 = plt.subplot(1, 3, 2)
    ax2 = sns.regplot(x='popularity', y='revenue', data=revenue_df, x_jitter=.1, color='g', marker='o')
    ax2.text(500, 3e9, 'r=0.59', fontsize=16)
    plt.title('popularity-revenue-scatter', fontsize=18)
    plt.xlabel('popularity', fontsize=16)
    plt.ylabel('revenue', fontsize=16)

    ax3 = plt.subplot(1, 3, 3)
    ax3 = sns.regplot(x='vote_count', y='revenue', data=revenue_df, x_jitter=.1, color='b', marker='v')
    ax3.text(7000, 2e9, 'r=0.75', fontsize=16)
    plt.title('voteCount-revenue-scatter', fontsize=20)
    plt.xlabel('vote_count', fontsize=16)
    plt.ylabel('revenue', fontsize=16)

    fig.savefig('revenue.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    timu_5_3_2()
    pass