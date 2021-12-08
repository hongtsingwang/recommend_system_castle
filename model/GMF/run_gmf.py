# coding=utf-8

import os
import sys
import pickle

import tensorflow as tf

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from gmf_model import GmfModel

"""
GMF 模型： 即广义矩阵分解模型， 是一个非常初级，简单的模型
如果你想对矩阵分解模型有了解， 可以参考下面三篇文章
1-
推荐系统之矩阵分解模型-科普篇 - 腾讯技术工程的文章 - 知乎
https://zhuanlan.zhihu.com/p/69662980
2-
推荐系统之矩阵分解模型-原理篇 - 腾讯技术工程的文章 - 知乎
https://zhuanlan.zhihu.com/p/360689325
3-
推荐系统之矩阵分解模型-实践篇 - 腾讯技术工程的文章 - 知乎
https://zhuanlan.zhihu.com/p/363259363
"""

# 模型初始化，配置参数
current_dir = os.getcwd()
sample_data_dir = os.path.join(current_dir, "..", "..", "sample")
train_file_path = os.path.join(sample_data_dir, "train.csv")
test_file_path = os.path.join(sample_data_dir, "test.csv")
data_info_dict_path = os.path.join(sample_data_dir, "data_info.pkl")
data_info_dict = pickle.load(open(data_info_dict_path))
user_num = data_info_dict["user_num"]
item_num = data_info_dict["item_num"]


epochs = 10
batch_size = 8
lr = 0.001
output_path = "./gmf_model.h5"

gmf_model = GmfModel(user_num=user_num, item_num=item_num, epochs=epochs, batch_size=batch_size, lr=lr) 

train_data_set = tf.data.TextLineDataset(train_file_path)
test_data_set = tf.data.TextLineDataset(test_file_path)

train_data = train_data_set.repeat().shuffle(42).batch(32)
test_data = test_data_set.repeat().shuffle(42).batch(32)


gmf_model.compile()
gmf_model.get_model_info()
gmf_model.train(x_data=X_train, y_data=y_train, validation_data=(X_test, y_test))
gmf_model.save_model(output_path)
