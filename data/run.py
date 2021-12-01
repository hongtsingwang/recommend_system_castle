'''
Author: your name
Date: 2021-11-29 20:54:59
LastEditTime: 2021-12-01 22:49:47
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /recommend_system_castle/data/run.py
'''
# coding = utf-8

from genericpath import sameopenfile
import os
from random import sample

from tqdm import tqdm
from process_data import process_data


# 数据路径
current_dir = os.getcwd()
base_dir = os.path.join(current_dir, "..")
input_dir = os.path.join(base_dir, "input")
sample_dir = os.path.join(base_dir, "sample")
ratings_file_path = os.path.join(input_dir, "ml-20m", "ratings.csv")

train_file_path = os.path.join(sample_dir, "train.csv")
test_file_path = os.path.join(sample_dir, "test.csv")
user_item_set_file_path = os.path.join(sample_dir, "user_item_set.csv")
user_positive_item_set = os.path.join(sample_dir, "user_positive_item_set.csv")


# 参数配置, 当前所有参数都是随便拍的
negative_sample_ratio = 3.0
negative_sample_threshold = 0.5
negative_sample_method = "popular"
split_test_ratio = 0.2
shuffle_before_split = True
split_ensure_positive = True
topk_sample_user = 1000


user_num, item_num, train_data, test_data, test_user_item_set, test_user_positive_item_set = process_data(
        ratings_file_path,
        negative_sample_ratio,
        negative_sample_threshold,
        negative_sample_method,
        split_test_ratio,
        shuffle_before_split,
        split_ensure_positive,
        topk_sample_user
    )

print("user num is:", user_num)
print("item num is:", item_num)
with open(train_file_path, "w") as f:
    for data in tqdm(train_data):
        f.write(data)
        f.write("\n")

with open(test_file_path, "w") as f:
    for data in tqdm(test_data):
        f.write(data)
        f.write("\n")

with open(user_item_set_file_path, "w") as f:
    for data in test_user_item_set:
        f.write(data)
        f.write("\n")

with open(user_positive_item_set, "w") as f:
    for data in test_user_positive_item_set:
        f.write(data)
        f.write("\n")
