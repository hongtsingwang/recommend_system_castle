# coding=utf-8

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard


class BaseModel(object):
    def __init__(
        self,
        lr=0.01,
        batch_size=64,
        test_batch_size=None,
        epochs=1000,
        optimizer="SGD",
        loss="mean_squared_error",
        tb_log_path=None,
        metrics=None,
    ) -> None:
        """[所有模型的基类]

        Args:
            lr (float, optional): [学习率]. Defaults to 0.01.
            batch_size (int, optional): [模型训练batch大小]. Defaults to 64.
            test_batch_size ([type], optional): [验证集输入的batch大小]. Defaults to None.
            epochs (int, optional): [训练轮数]. Defaults to 1000.
            optimizer (str, optional): [优化器选择]. Defaults to "SGD".
            loss (str, optional): [损失函数选择]. Defaults to "mean_squared_error".
            metrics ([type], optional): [评测指标选择]. Defaults to None.
        """
        self.lr = lr
        self.loss = loss
        self.epochs = epochs
        self.metrics = metrics
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.tb_log_path = tb_log_path
        self.optimizer = self.get_optimizer(optimizer)
        # 构建模型
        self.model = self.build_model()

    def get_optimizer(self, optimizer_str):
        """[根据输入参数， 选择合适的优化器]

        Args:
            optimizer_str ([string]): [优化器名称]

        Raises:
            Exception: [超出参数选择范围]

        Returns:
            [optimizers]: [对应的优化器]
        """
        if optimizer_str == "SGD":
            optimizer = SGD(lr=self.lr)
        elif optimizer_str == "Adam":
            optimizer = Adam(lr=self.lr)
        else:
            raise Exception("No this optimizer availabel")
        return optimizer

    def build_model(self):
        """模型框架搭建, 不同模型应该有自己的构造模型框架的方法, 这里只是定义一下
        """
        pass

    def compile(self):
        """模型编译
        """
        self.model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)

    @staticmethod
    def generate_data_set(train_data, test_data, batch_size):
        def to_tensor(data):
            user_id_list = tf.constant([item[0] for item in data])
            item_id_list = tf.constant([item[1] for item in data])
            label_list = tf.constant([item[2] for item in data])
            return {"user_id": user_id_list, 'item_id': item_id_list}, label_list
        train_data = to_tensor(train_data)
        test_data = to_tensor(test_data)
        train_data_set = tf.data.Dataset.from_tensor_slices(train_data).shuffle(len(train_data)).batch(batch_size)
        test_data_set = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)
        return train_data_set, test_data_set

    def train(self, x_data, y_data, validation_data=None):
        """模型训练

        Args:
            x_data ([list]): 输入特征集合
            y_data ([list]): label集合
        """
        tb_callback = TensorBoard(log_dir=self.tb_log_path, write_graph=True, write_grads=True, histogram_freq=1,
                                  update_freq="epoch")
        if not validation_data:
            self.model.fit(
                x=x_data, y=y_data, epochs=self.epochs, batch_size=self.batch_size, callbacks=[tb_callback]
            )
        else:
            self.model.fit(
                x=x_data, y=y_data, epochs=self.epochs, batch_size=self.batch_size, callbacks=[tb_callback],
                validation_data=validation_data
            )

    def predict(self, input_data):
        """模型预测

        Args:
            input_data ([int]): 待预测的数据

        Returns:
            [int]: 预测结果
        """
        if self.test_batch_size:
            result = self.model.predict(input_data, self.test_batch_size)
        else:
            result = self.model.predict(input_data)
        return result

    def get_model_info(self):
        print(self.model.summary())
        plot_model(self.model, to_file=self.model_name + ".png", show_shapes=True)
