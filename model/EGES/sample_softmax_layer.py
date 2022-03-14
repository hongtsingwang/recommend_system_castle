'''
Author: your name
Date: 2022-01-23 13:51:11
LastEditTime: 2022-01-23 13:55:45
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /recommend_system_castle/model/EGES/sample_softmax_layer.py
'''
# coding=utf-8
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import GlorotUniform, Zeros


class SampledSoftmax(Layer):
    def __init__(self, item_nums, num_sampled, l2_reg, seed, **kwargs):
        super(SampledSoftmax, self).__init__(**kwargs)
        self.item_nums = item_nums
        self.num_sampled = num_sampled
        self.l2_reg = l2_reg
        self.seed = seed

    def build(self, input_shape):
        super(SampledSoftmax, self).build(input_shape)
        embed_size = input_shape[0][1]
        self.softmax_w = self.add_weight(
            name="softmax_w",
            shape=(self.item_nums, embed_size),
            initializer=GlorotUniform(self.seed),
            regularizer=L2(self.l2_reg)
        )
        self.softmax_b = self.add_weight(
            name="softmax_b",
            shape=(self.item_nums,),
            initializer=Zeros(),
            # trainable=False
        )

    def call(self, inputs, training=None, **kwargs):
        input_embed, labels = inputs
        if tf.keras.backend.learning_phase():
            softmax_loss = tf.nn.sampled_softmax_loss(
                weights=self.softmax_w,
                biases=self.softmax_b,
                labels=labels,
                inputs=input_embed,
                num_sampled=self.num_sampled,
                num_classes=self.item_nums,
                seed=self.seed,
                name="softmax_loss"
            )
        else:
            logits = tf.matmul(input_embed, tf.transpose(self.softmax_w))
            logits = tf.nn.bias_add(logits, self.softmax_b)
            labels_one_hot = tf.one_hot(labels, self.item_nums)
            softmax_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels_one_hot,
                logits=logits
            )
        return softmax_loss

    def compute_output_shape(self, input_shape):
        return (None,)

    def get_config(self, ):
        config = {
            'item_nums': self.item_nums,
            'num_sampled': self.num_sampled,
            "l2_reg": self.l2_reg,
            "seed": self.seed
        }
        base_config = super(SampledSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))