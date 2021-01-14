import numpy as np
import tensorflow as tf

from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.datas.dataloader import TFWriter, TFLoader

from data import extract_text_tuples, make_input_file
from model import BERTClassifier

# Download the pre-trained BERT (Chinese) model (if not exists).
load_check = LoadCheckpoint()
loaded_params, vocab_file, model_path = load_check.load_bert_param()

# Set some hyper-parameters.
loaded_params.batch_size = 12
loaded_params.maxlen = 512  # The max sequence length that BERT can handle is 512.
loaded_params.label_size = 2  # Deteriorate the triplet problem into binary classification.

# Make input files for the BERT model.
text_tuples = extract_text_tuples('data/valid.json')
make_input_file(text_tuples, path='Input/valid', max_len=loaded_params.maxlen, mode='test')

# Build the BERT model.
model = BERTClassifier(loaded_params)
model.build(input_shape=(3, loaded_params.batch_size, loaded_params.maxlen))
model.summary()

# Transform the input data and build a data loader.
writer = TFWriter(loaded_params.maxlen, vocab_file, modes=['valid'], task='cls', check_exist=True)
loader = TFLoader(loaded_params.maxlen, loaded_params.batch_size, task='cls')

# Load the latest checkpoint.
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint('save'))

batch_idx = 0
accuracy_list = []


@tf.function
def test(inputs):
    return model.predict(inputs)  # [batch_size, max_length, label_size]


for X, token_type_id, input_mask, Y in loader.load_valid():
    predict = test([X, token_type_id, input_mask]).numpy()

    # Calculate accuracy for the original problem.
    score = predict[::2] - predict[1::2]
    accuracy_list.append((score > 0).mean())

print(f'Accuracy: {np.mean(accuracy_list):.4f}')
