import os
import wget
import json
import zipfile
import transformers
import tensorflow as tf

from typing import Sequence, Tuple


def _download_data(path: str, overwrite: bool = False):
    if not os.path.exists(path) or overwrite:
        print('Dataset not found. Downloading CAIL2019 dataset...')
        downloaded = wget.download('https://cail.oss-cn-qingdao.aliyuncs.com/cail2019/CAIL2019-SCM.zip')
        with zipfile.ZipFile(downloaded) as zip_file:
            zip_file.extractall(path)
        os.remove(downloaded)


def _extract_examples(path: str, mode: str) -> Sequence:
    examples = []
    with open(path, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            items = json.loads(line)
            a = items.get('A').replace('\n', '')
            b = items.get('B').replace('\n', '')
            c = items.get('C').replace('\n', '')
            label = 0

            # `label` is the one more similar to A. We swap B and C if C is more like A when training.
            if items.get('label') == 'C':
                if mode == 'train':
                    b, c = c, b
                else:
                    label = 1

            examples.append((a, b, c, label))
    return examples


def _augment_examples(examples: Sequence) -> Sequence:
    augmented = []
    for a, b, c, label in examples:
        augmented.append((a, c, b, 1))
        augmented.append((b, a, c, 0))
        augmented.append((b, c, a, 1))
        augmented.append((c, c, a, 0))
        augmented.append((c, a, c, 1))
    return augmented


def _encode_examples(examples: Sequence) -> Tuple:
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-chinese')
    ab, ac, labels = [], [], []
    for a, b, c, label in examples:
        ab.append((a, b))
        ac.append((a, c))
        labels.append(label)
    ab = tokenizer(ab, truncation=True, padding='max_length', return_tensors='tf')
    ac = tokenizer(ac, truncation=True, padding='max_length', return_tensors='tf')
    return ab, ac, labels


def get_dataset(mode: str, batch_size: int) -> Tuple[tf.data.Dataset, int]:
    _download_data('data')
    examples = _extract_examples(os.path.join('data', f'{mode}.json'), mode=mode)
    if mode == 'train':
        examples = _augment_examples(examples)
    ab, ac, labels = _encode_examples(examples)
    size = len(labels)
    dataset = tf.data.Dataset.from_tensor_slices((ab.get('input_ids'),
                                                  ab.get('token_type_ids'),
                                                  ab.get('attention_mask'),
                                                  ac.get('input_ids'),
                                                  ac.get('token_type_ids'),
                                                  ac.get('attention_mask'),
                                                  labels))
    if mode == 'train':
        dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder=True)
        size -= size % batch_size
    else:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, size
