# coding=utf-8

import sys
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input, Dense, Layer, Add

sys.path.append("..")
from base_model import BaseModel

class CrossLayer(Layer):
    def __init__(self, input_dim, output_dim=30, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(CrossLayer, self).build(input_shape)

    def call(self, x):
        a = K.pow(K.dot(x,self.kernel), 2)
        b = K.dot(K.pow(x, 2), K.pow(self.kernel, 2))
        return K.mean(a-b, 1, keepdims=True)*0.5

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
        })
        return config
class FmModel(BaseModel):
    def __init__(
        self,
        units_num=1,
        epochs=100,
        batch_size=16,
        lr=0.001,
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics="binary_accuracy",
        bias_regularizer=0.01,
        kernel_regularizer=0.02,
        model_name="fm",
        tb_log_path="./fm_model"
    ) -> None:
        self.units_num = units_num
        self.bias_regularizer = bias_regularizer
        self.kernel_regularizer = kernel_regularizer
        super(FmModel, self).__init__(
            lr=lr, epochs=epochs, optimizer=optimizer, loss=loss, metrics=metrics, batch_size=batch_size,
            tb_log_path=tb_log_path, model_name=model_name
        )

    def build_model(self):
        """模型框架搭建
        """
        inputs = Input(shape=(self.units_num,), name="feature_layer")
        linear_part = Dense(
            1,
            name="linear",
            bias_regularizer=l2(self.bias_regularizer),
            kernel_regularizer=l1(self.kernel_regularizer)
        )(inputs)
        cross_part = CrossLayer(self.units_num)(inputs)
        add_layer = Add()([linear_part, cross_part])
        outputs = Activation("sigmoid")(add_layer)
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
