import logging

import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers.core import Flatten
from tensorflow.keras.layers.embeddings import Embedding
from tensorflow.keras.layers.merge import dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import TFOptimizer
from tensorflow.keras.utils import plot_model

from src import skip_gram


class Word2Vec:
    """
        word2vec 模型搭建
    """

    def __init__(self):
        self.seed = None
        self.model = None

    def build(self, vector_dim, vocab_size, learn_rate):
        """ returns a word2vec model """
        logging.info("Building keras model")
        # 标准差计算公式
        stddev = 1.0 / vector_dim
        logging.info("Setting initializer standard deviation to: %.4f", stddev)
        initializer = RandomNormal(mean=0.0, stddev=stddev, seed=self.seed)
        # 输入层， 因为一次只输入一个词，所以shape为 batch_size * 1, 表达式写为 Input(shape=(1,), name="word_input")
        word_input = Input(shape=(1,), name="word_input")
        # 过Embedding层， 将一个词转化为Embedding向量
        word = Embedding(input_dim=vocab_size, output_dim=vector_dim, input_length=1,
                         name="word_embedding", embeddings_initializer=initializer)(word_input)
        # context input 我猜测是label
        context_input = Input(shape=(1,), name="context_input")
        # context input也会过一个embedding, 但它过的embedding和word_input是不一样的。
        context = Embedding(input_dim=vocab_size, output_dim=vector_dim, input_length=1,
                            name="context_embedding", embeddings_initializer=initializer)(context_input)
        # TODO 两个embedding进行点积操作。不明白为啥要做点积。
        merged = dot([word, context], axes=2, normalize=False, name="dot")
        # 去除无用维度
        merged = Flatten()(merged)
        # 过sigmoid函数。
        output = Dense(1, activation='sigmoid', name="output")(merged)
        #
        optimizer = TFOptimizer(tf.train.AdagradOptimizer(learn_rate))
        model = Model(inputs=[word_input, context_input], outputs=output)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
        self.model = model

    def train(self, sequence, window_size, negative_samples, batch_size, epochs, verbose):
        """ Trains the word2vec model """
        logging.info("Training model")

        # in order to balance out more negative samples than positive
        # 负样本权重
        negative_weight = 1.0 / negative_samples
        # 指定正负样本的权重
        class_weight = {1: 1.0, 0: negative_weight}
        logging.info("Class weights set to: %s", class_weight)

        sequence_length = len(sequence)
        approx_steps_per_epoch = (sequence_length * (window_size * 2.0) + sequence_length * negative_samples) / batch_size
        logging.info("Approx. steps per epoch: %d", approx_steps_per_epoch)
        seed = 1
        batch_iterator = skip_gram.batch_iterator(sequence, window_size, negative_samples, batch_size, seed)

        self.model.fit_generator(batch_iterator,
                                 steps_per_epoch=approx_steps_per_epoch,
                                 epochs=epochs,
                                 verbose=verbose,
                                 class_weight=class_weight,
                                 max_queue_size=100)

    def plot(self, path):
        """ Plots the model to a file"""
        logging.info("Saving model image to: %s", path)
        plot_model(self.model, show_shapes=True, to_file=path)