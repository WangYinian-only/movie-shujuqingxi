# -*- codeing = utf-8 -*-
# @Time : 2021/12/8 16:54
# @Author : 王伊念
# File : import_wrong.py
# @Software : PyCharm
# 首先导入PyMySQL库
import logging

import pymysql

# 连接数据库，创建连接对象connection
# 连接对象作用是：连接数据库、发送数据库信息、处理回滚操作（查询中断时，数据库回到最初状态）、创建新的光标对象
conn = pymysql.connect(
    host='localhost',  # host属性
    port=3306,  # 端口号
    user='root',  # 用户名
    password='602511dtywyy',  # 登陆密码
    charset='utf8',  # UTF8防乱码
    # local_infile=1
    # database='import_movies'  # 数据库名
)

# 创建光标对象，一个连接可以有很多光标，一个光标跟踪一种数据状态。
# 光标对象作用是：、创建、删除、写入、查询等等
cur = conn.cursor()


# load_csv函数，参数分别为csv文件路径，表名称，数据库名称
def load_csv(csv_file_path, table_name, database='import_movies'):
    # 打开csv文件
    file = open(csv_file_path, 'r', encoding='utf-8')

    # 读取csv文件第一行字段名，创建表
    reader = file.readline()
    b = reader.split(',')
    colum = ''
    for a in b:
        colum = colum + a + 'varchar(255)'
    colum = colum[:-1]

    # 编写sql,create_sql创建表，data_sql负责导入数据
    create_sql = 'create table if not exists' + table_name + '' + '(' + colum + ')'
    data_sql = "LOAD DATA LOCAL INFILE '%s' INTO TABLE %s FIELDS TERMINATED BY ',' LINES TERMINATED BY '\\r\\n' IGNORE 1 LINES" % ('tmdb_5000_credits.csv',table_name)


    # 使用数据库
    cur.execute('use %s' % database)

    # 设置编码格式
    cur.execute('SET NAMES utf8;')
    cur.execute('SET character_set_connection=utf8;')

    # 执行create_sql,创建表
    cur.execute(create_sql)

    # 执行data_sql,导入数据
    cur.execute(data_sql)
    conn.commit()

    # 关闭连接
    conn.close()
    cur.close()


if __name__ == '__main__':
    load_csv('D:\\data\\tmdb_5000_credits.csv', 'credits')
