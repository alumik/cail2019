import tensorflow as tf

from nlpgnn.models import bert


class BERTClassifier(tf.keras.Model):

    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.bert = bert.BERT(params)
        self.dense = tf.keras.layers.Dense(params.label_size, activation='relu')

    def call(self, inputs, training=True, **kwargs):
        h = self.bert(inputs, training)
        sequence_output = h.get_pooled_output()
        pre = self.dense(sequence_output)
        output = tf.math.softmax(pre, axis=-1)
        return output

    def predict(self, inputs, training=False, **kwargs):
        return self(inputs, training)

    def get_config(self):
        raise NotImplementedError
