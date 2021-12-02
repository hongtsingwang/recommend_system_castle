# coding=utf-8

import os
import random
import numpy as np

from tqdm import tqdm
from collections import defaultdict


def process_data(
    input_file_path,
    negative_sample_ratio=1,
    negative_sample_threshold=0,
    negative_sample_method="random",
    split_test_ratio=0.4,
    shuffle_before_split=True,
    split_ensure_positive=False,
    topk_sample_user=None,
):
    """
    description: 对测试集输入数据进行处理， 输出为模型认识的数据类型
    :param input_file_path: 数据输入文件路径
    :param negative_sample_ratio: 负样本采样比例，如1代表正负样本比例1:1, 10代表正负样本1:10, 0代表不需要负样本
    :param negative_sample_threshold: 负采样的权重阈值，权重大于或者等于此值为正样例，小于此值既不是正样例也不是负样例
    :param negative_sample_method: 负采样方法，值为'random'或'popular'
    :param split_test_ratio: 切分时测试集占比，这个值在0和1之间
    :param shuffle_before_split: 切分前是否对数据集随机顺序
    :param topk_sample_user: 用来计算TopK指标时用户采样数量，为None则表示采样所有用户
    :return: 用户数量，物品数量，训练集，测试集，用于TopK评估数据
    @rtype: object
    """
    # 读取输入数据
    dataset = read_from_input_data(input_file_path)
    # 根据movie lens数据， 制作负样本
    data = generate_negative_sample(
        dataset,
        negative_sample_ratio,
        negative_sample_threshold,
        negative_sample_method,
    )
    # 数据处理, 稀疏值处理等
    data, user_num, item_num, _, _ = neaten_id(data)
    # 划分训练集和样本集
    train_data, test_data = train_test_split(
        data, split_test_ratio, shuffle_before_split, split_ensure_positive
    )
    test_user_item_set, test_user_positive_item_set = prepare_topk(
        train_data, test_data, user_num, item_num, topk_sample_user
    )
    data_info_dict = generate_user_info_dict(user_num, item_num)
    return (
        data_info_dict,
        train_data,
        test_data,
        test_user_item_set,
        test_user_positive_item_set,
    )


def generate_user_info_dict(user_num, item_num):
    data_info_dict = dict()
    data_info_dict["user_num"] = user_num
    data_info_dict["item_num"] = item_num
    return data_info_dict


def read_from_input_data(input_file_path):
    data_sets = list()
    with open(input_file_path, "r") as f:
        for line in tqdm(f):
            line_list = line.strip().split(",")
            # 将string类型转换为int类型
            # line_list 一共4列， 分别为user_id, item_id, rating, time_stamp
            user_id, item_id, rating, _ = line_list
            user_id = int(user_id)
            item_id = int(item_id)
            rating = float(rating)
            user_item_info = (user_id, item_id, rating)
            data_sets.append(user_item_info)
    return data_sets


def generate_negative_sample(
    dataset, negative_sample_ratio, negative_sample_threshold, negative_sample_method
):
    """
    description: 基于输入数据， 构造负样本
    param {*}
    return {*}
    """
    if not negative_sample_ratio:
        # 所有数据均为正样本， 以正样本形式输出
        dataset = [(d[0], d[1], 1) for d in dataset]  # 变成隐反馈数据
        return dataset

    # 输出rating文件中有记录的item列表
    item_set = set([i[1] for i in dataset])
    print(f"文件中有记录的item数量为{len(item_set)}个")
    # 定义每个item的权重， 对random模式， 每个item的权重都是1
    # 对popular模式， item出现频率越高， item权重越高
    if negative_sample_method == "random":
        negative_sample_weight_dict = defaultdict(lambda: 1)
        negative_sample_weight_dict.fromkeys(item_set)
    elif negative_sample_method == "popular":
        negative_sample_weight_dict = defaultdict(lambda: 0)
        for data in tqdm(dataset):
            item_id = data[1]
            negative_sample_weight_dict[item_id] += 1
    else:
        raise ValueError(
            f"negative_sample_method参数配置'{negative_sample_method}'错误，必须为popular或者random之一"
        )

    # 得到每个用户正样本与非正样本集合
    # 正样本之外的内容， 只能说是非正样本， 不能说是负样本， 只是打分偏低， 但这个阈值目前没有确定
    # key为用户ID， value为样本set
    user_positive_dict, user_unpositive_dict = defaultdict(set), defaultdict(set)

    # 整合数据
    for data in tqdm(dataset):
        user_id, item_id, rating = data
        if rating > negative_sample_threshold:
            user_positive_dict[user_id].add(item_id)
        else:
            user_unpositive_dict[user_id].add(item_id)

    # 取有正样本的用户的集合
    user_list = list(user_positive_dict.keys())

    # 获取每个用户对应的负样本
    negative_sample_list = _negative_sample(
        item_set,
        user_list,
        user_positive_dict,
        user_unpositive_dict,
        negative_sample_ratio,
        negative_sample_weight_dict,
    )

    new_data_set = []
    for user_id, negative_item_lists in tqdm(zip(user_list, negative_sample_list)):
        for item_id in negative_item_lists:
            new_data_set.append((user_id, item_id, 0))

    for user_id, positive_item_list in tqdm(user_positive_dict.items()):
        for item_id in positive_item_list:
            new_data_set.append((user_id, item_id, 1))
    return new_data_set


def _negative_sample(
    item_set,
    user_list,
    user_positive_dict,
    user_unpositive_dict,
    negative_sample_ratio,
    negative_sample_weight_dict,
):
    """
    description: 
    param {*}
    return {*}
    """
    # 可以取负样例的物品id列表, TODO 这里需要再深入理解一下
    user_negative_list = []
    for user in tqdm(user_list):
        user_positive_set = user_positive_dict[user]
        user_unpositive_set = user_unpositive_dict[user]
        candidate_negative_list = list(
            item_set - user_positive_set - user_unpositive_set
        )
        negative_sample_num = min(
            int(len(user_positive_set) * negative_sample_ratio),
            len(candidate_negative_list),
        )
        if negative_sample_num <= 0:
            return []
        negative_sample_weight_list = [
            negative_sample_weight_dict[item_id] for item_id in candidate_negative_list
        ]
        weights = np.array(negative_sample_weight_list, dtype=np.float)
        # 对权重归一化
        weights /= weights.sum()
        # 采集n_negative_sample个负样例（通过下标采样是为了防止物品id类型从int或str变成np.int或np.str）
        sample_indices = np.random.choice(
            range(len(negative_sample_weight_list)), negative_sample_num, False, weights
        )
        sample_result = [candidate_negative_list[i] for i in sample_indices]
        user_negative_list.append(sample_result)
    return user_negative_list


def neaten_id(data):
    # 虽然每个用户有自己的user_id, 但现在因为有部分user_id不参与模型训练， 需要对用户重新进行编号
    new_data = []
    user_num, item_num = 0, 0
    user_id_old2new, item_id_old2new = {}, {}
    for user_id_old, item_id_old, label in tqdm(data):
        if user_id_old not in user_id_old2new:
            user_id_old2new[user_id_old] = user_num
            user_num += 1
        if item_id_old not in item_id_old2new:
            item_id_old2new[item_id_old] = item_num
            item_num += 1
        new_data.append(
            (user_id_old2new[user_id_old], item_id_old2new[item_id_old], label)
        )
    return new_data, user_num, item_num, user_id_old2new, item_id_old2new


def train_test_split(data, test_ratio=0.4, shuffle=True, ensure_positive=False):
    """
    将数据切分为训练集数据和测试集数据
    :param data: 原数据，第一列为用户id，第二列为物品id，第三列为标签
    :param test_ratio: 测试集数据占比，这个值在0和1之间
    :param shuffle: 是否对原数据随机排序
    :param ensure_positive: 是否确保训练集每个用户都有正样例
    :return: 训练集数据和测试集数据
    """
    if shuffle:
        random.shuffle(data)
    n_test = int(len(data) * test_ratio)
    test_data, train_data = data[:n_test], data[n_test:]

    if ensure_positive:
        # 所有用户ID列表 - 有正样本的用户ID列表
        user_set = {d[0] for d in data} - {
            user_id for user_id, _, label in train_data if label == 1
        }
        if len(user_set) > 0:
            print(
                "警告：为了确保训练集数据每个用户都有正样例，%d(%f%%)条数据从测试集随机插入训练集"
                % (len(user_set), 100 * len(user_set) / len(data))
            )

        i = len(test_data) - 1
        while len(user_set) > 0:
            assert i >= 0, "无法确保训练集每个用户都有正样例，因为存在没有正样例的用户：" + str(user_set)
            if test_data[i][0] in user_set and test_data[i][2] == 1:
                user_set.remove(test_data[i][0])
                train_data.insert(random.randint(0, len(train_data)), test_data.pop(i))
            i -= 1
    return train_data, test_data


def prepare_topk(train_data, test_data, user_num, item_num, sample_user_num=None):
    """
    准备用于topk评估的数据
    :param train_data: 训练集数据，有三列，分别是user_id, item_id, label
    :param test_data: 测试集数据，有三列，分别是user_id, item_id, label
    :param user_num: 用户数量
    :param item_num: 物品数量
    :param sample_user_num: 用户取样数量，为None则表示采样所有用户
    :return: 用于topk评估的数据，类型为TopkData，其包括在测试集里每个用户的（可推荐物品集合）与（有行为物品集合）
    """
    # 对用户进行采样， 只取TOPK 用户， 如果没有配置这个值， 或者值过大， 就取当前用户的数目
    if sample_user_num is None or sample_user_num > user_num:
        sample_user_num = user_num
    # 从用户列表中随机抽取sample_user_num个用户
    user_set = np.random.choice(range(user_num), sample_user_num, False)

    def get_user_item_set(data, only_positive=False):
        # 如果only_positive为true
        # 将每个用户的item构建为一个set
        user_item_set = defaultdict(set)
        for user_id, item_id, label in data:
            if user_id in user_set and (not only_positive or label == 1):
                user_item_set[user_id].add(item_id)
        return user_item_set

    # 获取测试集的user_item_set
    # key: user_id value: item集合， 所有item中排除掉所选择的样本， 正样本一定是会被排除掉的。
    # 对非正样本， 如果only_positive为False, 也会被排除掉; 反之， 可以作为负样本出现
    test_user_item_set = {
        user_id: set(range(item_num)) - item_set
        for user_id, item_set in get_user_item_set(train_data).items()
    }
    # test集合， 只有正样本， 没有负样本
    test_user_positive_item_set = get_user_item_set(test_data, only_positive=True)
    return test_user_item_set, test_user_positive_item_set
