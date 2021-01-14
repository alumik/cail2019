import tensorflow as tf

from nlpgnn.models import bert


class BERTClassifier(tf.keras.Model):
    """BERT model for binary classification."""

    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)

        # Create the BERT layer.
        self.bert = bert.BERT(params)

        # Create the output dense layer.
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, **kwargs):
        """Do forward propagation."""

        x = self.bert(inputs, training)

        # We choose the first token `[CLS]` as the identity of the input sequence.
        x = x.get_pooled_output()

        x = self.dense(x)
        return x

    def predict(self, inputs, training=False, **kwargs):
        return self(inputs, training)

    def get_config(self):
        raise NotImplementedError
