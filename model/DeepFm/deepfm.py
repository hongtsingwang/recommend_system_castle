# coding=utf-8

import sys
from tensorflow import int32
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Input, Dense, Embedding

sys.path.append("..")
from base_model import BaseModel

class DeepFmModel(BaseModel):
    def __init__(
        self,
        units_num=1,
        user_num=10,
        item_num=10,
        epochs=100,
        batch_size=16,
        lr=0.01,
        dim=8,
        l2_value=1e-6,
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics="binary_accuracy",
        bias_regularizer=0.01,
        kernel_regularizer=0.02,
        model_name="lr",
        tb_log_path="./deepfm_model"
    ):
        self.units_num = units_num
        self.user_num = user_num
        self.item_num = item_num
        self.dim = dim
        self.bias_regularizer = bias_regularizer
        self.kernel_regularizer = kernel_regularizer
        self.l2 = l2(l2_value)
        super(DeepFmModel, self).__init__(
            lr=lr, epochs=epochs, optimizer=optimizer, loss=loss, metrics=metrics, batch_size=batch_size,
            tb_log_path=tb_log_path, model_name=model_name
        )

    def build_model(self):
        user_id = Input(shape=(), name="user_id", dtype=int32)
        user_embedding = Embedding(self.user_num, self.dim, embeddings_regularizer=self.l2)

        item_id = Input(shape=(), name="item_id", dtype=int32)
        item_embedding = Embedding(self.item_num, self.dim, embeddings_regularizer=self.l2)

        user_bias = Embedding(self.user_num, 1, embeddings_initializer="zeros")(user_id)
        item_bias = Embedding(self.item_num, 1, embeddings_initializer="zeros")(item_id)

        fm = 


