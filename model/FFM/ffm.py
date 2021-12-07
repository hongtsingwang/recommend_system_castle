# coding=utf-8

import sys
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Input, Dense, Layer, Add, Reshape, Activation

sys.path.append("..")
from base_model import BaseModel

class CrossLayer(Layer):
    def __init__(self, field_dict, field_dim, input_dim, output_dim=30,  **kwargs):
        self.field_dict = field_dict
        self.field_dim = field_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.input_dim, self.field_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(CrossLayer, self).build(input_shape)

    def call(self, x):
        self.field_cross = K.variable(0, dtype='float32')
        for i in range(self.input_dim):
            for j in range(i+1, self.input_dim):
                weight = tf.math.reduce_sum(tf.math.multiply(self.kernel[i, self.field_dict[j]], self.kernel[j, self.field_dict[i]]))
                value = tf.math.multiply(weight, tf.math.multiply(x[:,i], x[:,j]))
                self.field_cross = tf.math.add(self.field_cross, value)
        return self.field_cross

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'field_dict': self.field_dict,
            'field_dim': self.field_dim,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
        })
        return config

class FfmModel(BaseModel):
    def __init__(
        self,
        units_num=1,
        epochs=100,
        batch_size=16,
        lr=0.001,
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics="binary_accuracy",
        # bias_regularizer=0.01,
        # kernel_regularizer=0.02,
        model_name="ffm",
        tb_log_path="./ffm_model",
        field_dim=None,
        output_dim=30,
        field_dict = None
    ) -> None:
        self.units_num = units_num
        self.field_dim = field_dim
        self.output_dim = output_dim
        self.field_dict = field_dict
        # self.bias_regularizer = bias_regularizer
        # self.kernel_regularizer = kernel_regularizer
        super(FfmModel, self).__init__(
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
            # bias_regularizer=l2(self.bias_regularizer),
            # kernel_regularizer=l1(self.kernel_regularizer)
        )(inputs)
        cross_part = CrossLayer(self.field_dict, self.field_dim, self.units_num, self.output_dim)(inputs)
        cross_part = Reshape((1,))(cross_part)
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