import numpy as np
import transformers
import tensorflow as tf

from model import Classifier
from preprocessing import get_dataset

# Set some hyper-parameters.
EPOCHS = 2
BATCH_SIZE = 6
MAX_LEN = 512  # The max sequence length that BERT can handle is 512.

# Get the dataset
print('Preparing dataset...')
dataset, size = get_dataset(mode='train', batch_size=BATCH_SIZE)

# Build the BERT model.
model = Classifier()
model(tf.zeros((6, BATCH_SIZE, MAX_LEN), dtype=tf.int32))
model.summary()

# Set up the optimizer and the loss function.
decay_lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=2e-5,
    decay_steps=10000,
    end_learning_rate=0.0
)
warmup_lr_scheduler = transformers.WarmUp(
    initial_learning_rate=2e-5,
    decay_schedule_fn=decay_lr_scheduler,
    warmup_steps=1000
)
optimizer = tf.keras.optimizers.Adam(learning_rate=warmup_lr_scheduler, clipnorm=1.0)
binary_cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()

# Make a checkpoint manager to save the trained model later.
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory='ckpt', checkpoint_name='model.ckpt', max_to_keep=3)


@tf.function
def _train(_inputs):
    _x = _inputs[:-1]
    _y = _inputs[-1]
    with tf.GradientTape() as tape:
        _predict = model(_x)
        _loss = binary_cross_entropy_loss(_y, _predict)
    grads = tape.gradient(_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return _loss, _predict


for i in range(EPOCHS):
    print(f'Epoch {i + 1}/{EPOCHS}')
    batch_idx = 0
    progbar = tf.keras.utils.Progbar(size, stateful_metrics=['acc'], unit_name='example')
    for inputs in dataset:
        y = inputs[-1]
        loss, predict = _train(inputs)
        acc = np.asarray(np.round(predict) == np.asarray(y)).mean()

        # Save a checkpoint every 10 batches.
        if batch_idx % 10 == 0:
            manager.save(checkpoint_number=batch_idx)

        batch_idx += 1
        progbar.add(BATCH_SIZE, values=[('acc', acc)])
