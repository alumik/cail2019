import tensorflow as tf

from nlpgnn.models import bert


class BERTClassifier(tf.keras.Model):

    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.bert = bert.BERT(params)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, **kwargs):
        x = self.bert(inputs, training)
        x = x.get_pooled_output()
        x = self.dense(x)
        return x

    def predict(self, inputs, training=False, **kwargs):
        return self(inputs, training)

    def get_config(self):
        raise NotImplementedError
