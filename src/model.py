import transformers
import tensorflow as tf


class Classifier(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create the BERT model.
        config = transformers.AutoConfig.from_pretrained(
            'bert-base-chinese',
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=True,
        )
        self.bert = transformers.TFAutoModel.from_pretrained('bert-base-chinese', config=config)

        # Create the output layers.
        self.subtract = tf.keras.layers.Subtract()
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.dense = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs, **kwargs):
        x1 = self.bert(
            input_ids=inputs[0],
            token_type_ids=inputs[1],
            attention_mask=inputs[2],
        )
        x2 = self.bert(
            input_ids=inputs[3],
            token_type_ids=inputs[4],
            attention_mask=inputs[5],
        )

        # We choose the first token `[CLS]` as the identity of the input sequence.
        x1 = x1.pooler_output
        x2 = x2.pooler_output

        x = self.subtract([x1, x2])
        x = self.dropout(x)
        x = self.dense(x)
        return x

    def get_config(self):
        raise NotImplementedError
