# coding=utf-8

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from fm import FmModel

# 载入数据
field_dict = {i:i//5 for i in range(30)}
breast_cancer_data = load_breast_cancer()
# 模型初始化，配置参数
units_num = len(breast_cancer_data.feature_names)
epochs = 10
batch_size = 8
lr = 0.001
output_path = "./fm_model.h5"
ffm_model = FmModel(units_num=units_num, epochs=epochs, batch_size=batch_size, lr=lr)
X_train, X_test, y_train, y_test = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2,
                                                    random_state=42, stratify=breast_cancer_data.target)
lr_model.compile()
lr_model.get_model_info()
lr_model.train(x_data=X_train, y_data=y_train, validation_data=(X_test, y_test))
lr_model.save_model(output_path)