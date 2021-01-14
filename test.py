import numpy as np
import tensorflow as tf

from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.datas.dataloader import TFWriter, TFLoader

from data import extract_text_tuples, make_input_file
from model import BERTClassifier

# Load pre-trained BERT model.
load_check = LoadCheckpoint()
loaded_params, vocab_file, model_path = load_check.load_bert_param()

# 定制参数
loaded_params.batch_size = 12
loaded_params.maxlen = 512
loaded_params.label_size = 2

# Make input files.
text_tuples = extract_text_tuples('data/valid.json')
make_input_file(text_tuples, path='Input/valid', max_len=loaded_params.maxlen)

model = BERTClassifier(loaded_params)
model.build(input_shape=(3, loaded_params.batch_size, loaded_params.maxlen))
model.summary()

writer = TFWriter(loaded_params.maxlen, vocab_file, modes=['valid'], task='cls', check_exist=True)
loader = TFLoader(loaded_params.maxlen, loaded_params.batch_size, task='cls')

# Load checkpoint.
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint('save'))

batch_idx = 0
accuracy_list = []


@tf.function
def test(inputs):
    return model.predict(inputs)  # [batch_size, max_length, label_size]


for X, token_type_id, input_mask, Y in loader.load_valid():
    predict = test([X, token_type_id, input_mask])
    accuracy_list.append((np.asarray(Y) == np.asarray(np.round(predict))).mean())

print(f'Accuracy: {np.mean(accuracy_list)}')
