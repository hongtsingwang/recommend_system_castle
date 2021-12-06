# coding=utf-8

import sys
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Input, Dense

sys.path.append("..")
from base_model import BaseModel

# TODO 增加预测模块


class LrModel(BaseModel):
    def __init__(
        self,
        units_num=1,
        epochs=100,
        batch_size=16,
        lr=0.01,
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics="binary_accuracy",
        bias_regularizer=0.01,
        kernel_regularizer=0.02,
        model_name="lr",
        tb_log_path="./lr_model"
    ) -> None:
        self.units_num = units_num
        self.bias_regularizer = bias_regularizer
        self.kernel_regularizer = kernel_regularizer
        super(LrModel, self).__init__(
            lr=lr, epochs=epochs, optimizer=optimizer, loss=loss, metrics=metrics, batch_size=batch_size,
            tb_log_path=tb_log_path, model_name=model_name
        )

    def build_model(self):
        """模型框架搭建
        """
        inputs = Input(shape=(self.units_num,), name="feature_layer")
        outputs = Dense(
            1,
            name="output_layer",
            activation="sigmoid",
            use_bias=True,
            bias_regularizer=l2(self.bias_regularizer),
            kernel_regularizer=l1(self.kernel_regularizer)
        )(inputs)
        model = Model(inputs, outputs)
        return model

    def output_parameter(self):
        """输出训练好的模型的参数
        """
        weight, bias = self.model.get_layer("output_layer").get_weights()
        print("weight is:", weight)
        print("bias is:", bias)


if __name__ == "__main__":
    pass