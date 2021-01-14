import numpy as np
import tensorflow as tf

from nlpgnn.datas.checkpoint import LoadCheckpoint
from nlpgnn.datas.dataloader import TFWriter, TFLoader
from nlpgnn.optimizers import optim
from nlpgnn.tools import bert_init_weights_from_checkpoint

from data import extract_text_tuples, make_input_file
from model import BERTClassifier

# Load pre-trained BERT model.
load_check = LoadCheckpoint()
loaded_params, vocab_file, model_path = load_check.load_bert_param()

loaded_params.batch_size = 12
loaded_params.maxlen = 512
loaded_params.label_size = 2

# Make input files.
text_tuples = extract_text_tuples('data/train.json')
make_input_file(text_tuples, path='Input/train', max_len=loaded_params.maxlen, mode='train')

model = BERTClassifier(loaded_params)
model.build(input_shape=(3, loaded_params.batch_size, loaded_params.maxlen))
model.summary()

optimizer = optim.AdamWarmup(learning_rate=2e-5, decay_steps=10000, warmup_steps=1000)
binary_cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()
bert_init_weights_from_checkpoint(model, model_path, loaded_params.num_hidden_layers, pooler=True)

writer = TFWriter(loaded_params.maxlen, vocab_file, modes=['train'], task='cls', check_exist=True)
loader = TFLoader(loaded_params.maxlen, loaded_params.batch_size, task='cls', epoch=2)
summary_writer = tf.summary.create_file_writer('tensorboard')

# Save the trained model.
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory='save', checkpoint_name='model.ckpt', max_to_keep=3)


@tf.function
def train(inputs):
    with tf.GradientTape() as tape:
        predict = model(inputs)
        loss = binary_cross_entropy_loss(Y, predict)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))
    return loss, predict


batch_idx = 0
for X, token_type_id, input_mask, Y in loader.load_train():
    train_loss, train_predict = train([X, token_type_id, input_mask])
    accuracy = (np.asarray(Y) == np.asarray(np.round(train_predict))).mean()
    if batch_idx % 101 == 0:
        print(f'Batch {batch_idx}: loss: {train_loss.numpy():.4f} acc: {accuracy:.4f}')
    if batch_idx % 10 == 0:
        manager.save(checkpoint_number=batch_idx)
    with summary_writer.as_default():
        tf.summary.scalar('loss', train_loss, step=batch_idx)
        tf.summary.scalar('acc', accuracy, step=batch_idx)
    batch_idx += 1
