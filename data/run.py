# coding = utf-8

from process_data import generate_negative_sample

# 参数配置, 当前所有参数都是随便拍的
negative_sample_ratio = 3.0
negative_sample_threshold = 0.5
negative_sample_method = "popular"
split_test_ratio = 0.2
shuffle_before_split = True
split_ensure_positive = True
topk_sample_user = 1000


dataset = generate_negative_sample(
        negative_sample_ratio,
        negative_sample_threshold,
        negative_sample_method,
        split_test_ratio,
        shuffle_before_split,
        split_ensure_positive,
        topk_sample_user
    )

for data in dataset:
    print(data)

