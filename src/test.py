import tensorflow as tf

from model import Classifier
from dataset import get_dataset

# Set some hyper-parameters.
BATCH_SIZE = 12
MAX_LEN = 512  # The max sequence length that BERT can handle is 512.

# Get the dataset
print('Preparing dataset...')
dataset, n = get_dataset(mode='test', batch_size=BATCH_SIZE)

# Build the BERT model.
model = Classifier()
model(tf.zeros((6, BATCH_SIZE, MAX_LEN), dtype=tf.int32))
model.summary()

# Load the latest checkpoint.
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint('ckpt'))


@tf.function
def _test_step(inputs):
    x, y = inputs[:-1], inputs[-1]
    pred = model.predict(x)
    accuracy.update_state(y, pred)


accuracy = tf.keras.metrics.CategoricalAccuracy()
progbar = tf.keras.utils.Progbar(n, unit_name='example')
for batch in dataset:
    _test_step(batch)
    progbar.add(BATCH_SIZE, values=[('acc', accuracy.result())])

print(f'Accuracy: {accuracy.result():.4f}')
