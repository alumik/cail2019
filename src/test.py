import numpy as np
import tensorflow as tf

from model import Classifier
from preprocessing import get_dataset

# Set some hyper-parameters.
BATCH_SIZE = 6
MAX_LEN = 512  # The max sequence length that BERT can handle is 512.

# Get the dataset
print('Preparing dataset...')
dataset, size = get_dataset(mode='test', batch_size=BATCH_SIZE)

# Build the BERT model.
model = Classifier()
model(tf.zeros((6, BATCH_SIZE, MAX_LEN), dtype=tf.int32))
model.summary()

# Load the latest checkpoint.
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint('ckpt'))


@tf.function
def _test(_inputs):
    return model.predict(_inputs)


batch_idx = 0
acc_list = []
progbar = tf.keras.utils.Progbar(size, unit_name='example')
for inputs in dataset:
    y = inputs[-1]
    pred = _test(inputs)
    acc = (np.argmax(pred, axis=-1) == np.argmax(y, axis=-1)).mean()
    acc_list.append(acc)
    batch_idx += 1
    progbar.add(BATCH_SIZE, values=[('acc', acc)])

print(f'Accuracy: {np.mean(acc_list):.4f}')
