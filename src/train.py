import fire
import transformers
import tensorflow as tf

from model import Classifier
from dataset import get_dataset


def main(
        epochs: int = 5,
        batch_size: int = 12,
        max_len: int = 512,
):
    # Get the dataset
    print('Preparing dataset...')
    dataset = get_dataset(mode='train', batch_size=batch_size)

    # Build the BERT model.
    print('Building model...')
    model = Classifier()
    model(tf.zeros((6, batch_size, max_len), dtype=tf.int32))
    model.summary()

    # Set up the optimizer and the loss function.
    decay_lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=2e-5,
        decay_steps=10000,
        end_learning_rate=0.0,
    )
    warmup_lr_scheduler = transformers.WarmUp(
        initial_learning_rate=2e-5,
        decay_schedule_fn=decay_lr_scheduler,
        warmup_steps=1000,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=warmup_lr_scheduler, clipnorm=1.0)
    categorical_cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()

    model.compile(
        optimizer=optimizer,
        loss=categorical_cross_entropy_loss,
        metrics=[accuracy],
    )
    print('Training model...')
    model.fit(
        dataset,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath='ckpt/checkpoint_{epoch:02d}',
                save_weights_only=True,
            ),
        ],
    )


if __name__ == '__main__':
    fire.Fire(main)
