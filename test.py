import numpy as np
import tensorflow as tf

from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.datas.dataloader import TFWriter, TFLoader

from data import extract_text_tuples, make_input_file
from model import BERTModel

# Download the pre-trained BERT (Chinese) model (if not exists).
load_check = LoadCheckpoint()
params, vocab_file, model_path = load_check.load_bert_param()

# Set some hyper-parameters.
params.batch_size = 6
params.maxlen = 512  # The max sequence length that BERT can handle is 512.
params.label_size = 2  # Deteriorate the triplet problem into binary classification.

# Make input files for the BERT model.
text_tuples = extract_text_tuples('data/test.json')
make_input_file(text_tuples, path='Input/valid', max_len=params.maxlen, mode='test')

# Build the BERT model.
model = BERTModel(params)
model.build(input_shape=(2, 3, params.batch_size, params.maxlen))
model.summary()

# Transform the input data and build a data loader.
writer = TFWriter(params.maxlen, vocab_file, modes=['valid'], task='cls', check_exist=True)
loader = TFLoader(params.maxlen, params.batch_size * 2, task='cls')

# Load the latest checkpoint.
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint('save'))

batch_idx = 0
accuracy_list = []


@tf.function
def test(inputs):
    return model.predict(inputs)


for X, token_type_id, input_mask, Y in loader.load_valid():
    X1, X2 = X[::2], X[1::2]
    token_type_id_1, token_type_id_2 = token_type_id[::2], token_type_id[1::2]
    input_mask_1, input_mask_2 = input_mask[::2], input_mask[1::2]
    Y = Y[::2]
    predict = test([[X1, token_type_id_1, input_mask_1], [X2, token_type_id_2, input_mask_2]]).numpy()

    # Calculate accuracy for the original problem.
    accuracy_list.append(np.asarray((np.round(predict) == np.asarray(Y))).mean())

print(f'Accuracy: {np.mean(accuracy_list):.4f}')
