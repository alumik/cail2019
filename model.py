import tensorflow as tf

from nlpgnn.models import bert


class BERTModel(tf.keras.Model):
    """The BERT model."""

    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)

        # Create the BERT layer.
        self.bert = bert.BERT(params)

        # Create the subtract layer.
        self.subtract = tf.keras.layers.Subtract()

        # Create the output dense layer.
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, **kwargs):
        """Do forward propagation."""

        x1 = self.bert(inputs[0], training)
        x2 = self.bert(inputs[1], training)

        # We choose the first token `[CLS]` as the identity of the input sequence.
        x1 = x1.get_pooled_output()
        x2 = x2.get_pooled_output()

        x = self.subtract([x1, x2])
        x = self.dense(x)
        return x

    def predict(self, inputs, training=False, **kwargs):
        return self(inputs, training)

    def get_config(self):
        raise NotImplementedError
