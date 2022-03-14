'''
Author: your name
Date: 2022-01-23 13:29:08
LastEditTime: 2022-01-23 13:50:29
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /recommend_system_castle/model/EGES/attention.py
'''

# coding=utf-8
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import L2

class Attention_Eges(Layer):
    def __init__(self, item_nums, l2_reg, seed, **kwargs):
        super(Attention_Eges, self).__init__(**kwargs)
        self.item_nums = item_nums
        self.seed = seed
        self.l2_reg = l2_reg

    def build(self, input_shape):
        super(Attention_Eges, self).build(input_shape)
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError("Attention_Eges must have two inputs")

        shape_set = input_shape

        self.feat_nums = shape_set[1][1]
        self.alpha_attention = self.add_weight(
                name='alpha_attention',
                shape=(self.item_nums, self.feat_nums),
                initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=self.seed),
                regularizer=L2(self.l2_reg)
        )

    def call(self, inputs, **kwargs):
        item_input = inputs[0]
        # (batch_size, feat_nums, embed_size)
        stack_embeds = inputs[1]
        # (batch_size, 1, feat_nums) 输出对每个item_id, 每个feature对应的attention权重
        alpha_embeds = tf.nn.embedding_lookup(self.alpha_attention, item_input)
        # 进行指数计算， 输出维度依然是(batch_size, 1, feat_nums) 
        alpha_embeds = tf.math.exp(alpha_embeds)
        # 输出维度: (batch_size, 1), 这个是作为分母来使用的
        alpha_sum = tf.reduce_sum(alpha_embeds, axis=-1)
        # 输出维度: (batch_size, 1, embed_size)
        merge_embeds = tf.matmul(alpha_embeds, stack_embeds)
        # 将维度缩小， 这样就输出每个batch的结果: (batch_size, embed_size), 归一化
        merge_embeds = tf.squeeze(merge_embeds, axis=1)  / alpha_sum
        return merge_embeds

    def compute_mask(self, inputs, mask):
        return None

    def compute_output_shape(self, input_shape):
        # 输出维度 batch_size, embed_size
        return (None, input_shape[1][2])

    def get_config(self):
        # 例行操作， 通知后面所需的参数情况 
        config = {'item_nums': self.item_nums, "l2_reg": self.l2_reg, 'seed': self.seed}
        base_config = super(Attention_Eges, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
