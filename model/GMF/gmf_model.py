# coding=utf-8

import tensorflow as tf
from tensorflow import int32
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Embedding, Multiply, Dense

from model.base_model import BaseModel


class GmfModel(BaseModel):
    def __init__(
        self,
        user_num,
        item_num,
        dim=8,
        l2_parameter=1e-6,
        lr=0.001,
        epochs=10000,
        optimizer="adam",
        loss="binary_crossentropy",
    ) -> None:
        self.user_num = user_num
        self.item_num = item_num
        self.dim = dim
        self.l2 = l2(l2_parameter)
        self.model_name = "gmf_model"
        super(GmfModel, self).__init__(
            lr=lr, epochs=epochs, optimizer=optimizer, loss=loss
        )

    def build_model(self):
        user_id = Input(shape=(), name="user_id", dtype=int32)
        item_id = Input(shape=(), name="item_id", dtype=int32)(user_id)
        user_embedding = Embedding(self.user_num, self.dim, embeddings_regularizer=l2)
        item_embedding = Embedding(self.item_num, self.dim, embeddings_regularizer=l2)(item_id)
        multi_layer = Multiply(name="multiply_layer")([user_embedding, item_embedding])
        output_layer = Dense(1, activation="sigmoid", name="output_layer", kernel_regularizer=l2)(multi_layer)
        model = Model(output_layer)
        return model
