import transformers
import tensorflow as tf


class Classifier(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create the BERT model.
        config = transformers.models.bert.BertConfig.from_pretrained('bert-base-chinese',
                                                                     return_dict=True,
                                                                     output_attentions=False,
                                                                     output_hidden_states=False,
                                                                     use_cache=True)
        self.bert = transformers.models.bert.TFBertModel.from_pretrained('bert-base-chinese', config=config)

        # Create the subtract layer.
        self.subtract = tf.keras.layers.Subtract()

        # Create the output dense layer.
        self.dense = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, inputs, training=True, **kwargs):
        """Do forward pass."""

        x1 = self.bert(input_ids=inputs[0],
                       token_type_ids=inputs[1],
                       attention_mask=inputs[2],
                       training=training)
        x2 = self.bert(input_ids=inputs[3],
                       token_type_ids=inputs[4],
                       attention_mask=inputs[5],
                       training=training)

        # We choose the first token `[CLS]` as the identity of the input sequence.
        x1 = x1.pooler_output
        x2 = x2.pooler_output

        x = self.subtract([x1, x2])
        x = self.dense(x)
        return x

    def predict(self, inputs, training=False, **kwargs):
        return self(inputs, training)

    def get_config(self):
        raise NotImplementedError
