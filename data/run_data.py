# coding = utf-8

import os
import pickle

from tqdm import tqdm

from process_data import process_data

# 是否为DEBUG模式
debug_mode = False
# 路径定义
current_dir = os.getcwd()
# 代码根目录
base_dir = os.path.join(current_dir, "..")
# 输入数据目录
input_dir = os.path.join(base_dir, "input")
# 基于输入数据， 生成的训练样本的目录
sample_dir = os.path.join(base_dir, "sample")
# user-item_id打分文件路径
if not debug_mode:
    ratings_file_path = os.path.join(input_dir, "ml-20m", "ratings.csv")
else:
    ratings_file_path = os.path.join(input_dir, "ml-20m", "ratings_test.csv")

print("ratings_file_path is:", ratings_file_path)


train_file_path = os.path.join(sample_dir, "train.csv")
test_file_path = os.path.join(sample_dir, "test.csv")
# TODO 这个文件的作用需要搞清楚
user_item_set_file_path = os.path.join(sample_dir, "user_item_set.csv")
# TODO 这个文件的作用需要搞清楚
user_positive_item_set = os.path.join(sample_dir, "user_positive_item_set.csv")
# 将数据的基本信息写到一个dict文件之中
data_info_file_path = os.path.join(sample_dir, "data_info.pkl")


# 参数配置, 当前所有参数都是随便拍的
# 负样本的比例
negative_sample_ratio = 1.0
# 负样本采样阈值， 在这个分数之上的都是正样本， 在这个阈值之下的是非正样本
negative_sample_threshold = 0.5
# 采样方式， 计算item出现频率， 出现频率越高， 约容易被抽样为负样本
negative_sample_method = "popular"
# 测试集比例
split_test_ratio = 0.4
# 是否打乱重排
shuffle_before_split = True
split_ensure_positive = True
topk_sample_user = 100000


data_info_dict, train_data, test_data, test_user_item_set, test_user_positive_item_set = process_data(
        ratings_file_path,
        negative_sample_ratio,
        negative_sample_threshold,
        negative_sample_method,
        split_test_ratio,
        shuffle_before_split,
        split_ensure_positive,
        topk_sample_user
    )

with open(data_info_file_path, "wb") as f:
    pickle.dump(data_info_dict, f, pickle.HIGHEST_PROTOCOL)

with open(train_file_path, "w") as f:
    for data in tqdm(train_data):
        data = [str(i) for i in data]
        user_id, item_id, label = data
        f.write("\t".join([user_id, item_id, label]))
        f.write("\n")

with open(test_file_path, "w") as f:
    for data in tqdm(test_data):
        data = [str(i) for i in data]
        user_id, item_id, label = data
        f.write("\t".join([user_id, item_id, label]))
        f.write("\n")

with open(user_item_set_file_path, "w") as f:
    for key, value in test_user_item_set.items():
        f.write(str(key))
        f.write("\t")
        f.write(str(value))
        f.write("\n")

with open(user_positive_item_set, "w") as f:
    for key, value in test_user_positive_item_set.items():
        f.write(str(key))
        f.write("\t")
        f.write(str(value))
        f.write("\n")
