# coding=utf-8

def process_data(
        negative_sample_ratio=1,
        negative_sample_threshold=0,
        negative_sample_method='random',
        split_test_ratio=0.4,
        shuffle_before_split=True,
        split_ensure_positive=False,
        topk_sample_user=300
    ):
    '''
    description: 对测试集输入数据进行处理， 输出为模型认识的数据类型
    :param negative_sample_ratio: 负样本采样比例，如1代表正负样本比例1:1, 10代表正负样本1:10, 0代表不需要负样本
    :param negative_sample_threshold: 负采样的权重阈值，权重大于或者等于此值为正样例，小于此值既不是正样例也不是负样例
    :param negative_sample_method: 负采样方法，值为'random'或'popular'
    :param split_test_ratio: 切分时测试集占比，这个值在0和1之间
    :param shuffle_before_split: 切分前是否对数据集随机顺序
    :param topk_sample_user: 用来计算TopK指标时用户采样数量，为None则表示采样所有用户
    :return: 用户数量，物品数量，训练集，测试集，用于TopK评估数据
    '''
    # 读取输入数据
    dataset = read_from_input_data()
    # 根据movie lens数据， 制作负样本
    data = generate_negative_sample(dataset, negative_sample_ratio, negative_sample_threshold, negative_sample_method)
    # 数据处理, 稀疏值处理等
    data, n_user, n_item, _, _ = neaten_id(data)
    # 划分训练集和样本集
    train_data, test_data = train_test_split(data, split_test_ratio, shuffle_before_split, split_ensure_positive)
    
    


def read_from_input_data():
    pass