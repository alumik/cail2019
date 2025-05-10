import fire
import tensorflow as tf

from model import Classifier
from dataset import get_dataset


def main(
        load_weights: str = 'ckpt/checkpoint_01_740',
        batch_size: int = 12,
        max_len: int = 512,
):
    # Get the dataset
    print('Preparing dataset...')
    dataset = get_dataset(mode='test', batch_size=batch_size)

    # Build the BERT model.
    model = Classifier()
    model(tf.zeros((6, batch_size, max_len), dtype=tf.int32))
    model.summary()

    # Load the latest checkpoint.
    model.load_weights(load_weights)

    accuracy = tf.keras.metrics.CategoricalAccuracy()
    model.compile(metrics=[accuracy])
    model.evaluate(dataset)


if __name__ == '__main__':
    fire.Fire(main)
