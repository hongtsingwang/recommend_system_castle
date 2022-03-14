# coding=utf-8

from tensorflow.keras.layers import Layer, InputSpec, Input, Embedding, Reshape, Dot, Dense
from tensorflow.keras.models import Model


class SkipGram(Layer):
    def __init__(self, word_count, embedding_size):
        self.word_count = word_count
        self.embedding_size = embedding_size
        pass

    def build(self):
        input_target = Input(shape=(1,), name="input_target")
        input_context = Input(shape=(1,), name="input_context")
        embedding_layer = Embedding(input_dim=self.word_count, output_dim=self.embedding_size, input_length=1, name="embedding_layer")
        target_embedding = embedding_layer(input_target)
        target_embedding = Reshape(target_shape=(self.embedding_size,), name="Reshape Target")
        context_embedding = embedding_layer(input_context)
        conteext_embedding = Reshape(target_shape=(self.embedding_size,), name="Reshape Context")

        product_result = Dot(axes=1, name="product 1")([target_embedding, conteext_embedding])
        product_result = Reshape(target_shape=(1, ), name="reshape dot")(product_result)
        output = Dense(units=(1,), activation="sigmoid", name="fc1")(product_result)
        model = Model(inputs=[input_target, input_context], output=output)
        return model
