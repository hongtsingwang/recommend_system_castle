# coding=utf-8
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from attention import Attention_Eges
from function import basic_loss_function
from sample_softmax_layer import SampledSoftmax

class EgesModel(object):
    def __init__(self, item_feature, side_feature_columns, num_sampled=100, l2_reg_embedding=0.00001, init_std=0.0001, 
         seed=1024):
        self.item_feature = item_feature
        self.side_feature_columns = side_feature_columns,
        self.num_sampled = num_sampled,
        self.l2_reg_embedding = l2_reg_embedding
        self.init_std = init_std
        self.seed = seed
    
    def build_model(self):
        features = build_input_features(
            [self.item_feature] + self.side_feature_columns
        )
        labels = Input(shape=(1,))
        features["label"] = labels
        group_embedding_list, dense_value_list = input_from_feature_columns(
            features, [self.item_feature] + self.side_feature_columns, self.l2_reg_embedding,
            self.init_std, self.seed, seq_mask_zero=False, support_dense=False, support_group=False)
        # concat (batch_size, num_feat, embedding_size)
        concat_embeds = concat_func(group_embedding_list, axis=1)
        att_embeds = Attention_Eges(
            self.item_feature.vocabulary_size,
            self.l2_reg_embedding,
            self.seed
        )([features[self.item_feature.name], concat_embeds])
        loss = SampledSoftmax(
            self.item_feature.vocabulary_size,
            100,
            self.l2_reg_embedding,
            self.seed
        )([att_embeds, features["label"]])
        model = Model(inputs=features, outputs=loss)
        self.model = model
        return model
    
    def compile_model(self):
        self.model.compile(
            optimizer="adam",
            loss=basic_loss_function,
            metrics=None
        )
    
    def fit_model(self,
        model_inputs,
        data,
        batch_size=2048, 
        epochs=3,
        verbose=1,
        validation_split=0.2
    ):
        history = self.model.fit(
            model_inputs,
            data=data['label'].values,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split
        )





