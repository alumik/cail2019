import json
import wget
import pathlib
import zipfile
import transformers
import tensorflow as tf


def _download_data(path: pathlib.Path):
    if path.exists():
        print('Using downloaded dataset.')
    else:
        print('Dataset not found. Downloading CAIL2019 dataset...')
        file = wget.download('https://cail.oss-cn-qingdao.aliyuncs.com/cail2019/CAIL2019-SCM.zip')
        with zipfile.ZipFile(file) as zip_file:
            zip_file.extractall(path)
        pathlib.Path(file).unlink()


def _extract_examples(path: pathlib.Path, mode: str) -> list[tuple[str, str, str, tuple[int]]]:
    examples = []
    with path.open(encoding='utf-8') as file:
        for line in file:
            item = json.loads(line.strip())
            a = item['A']
            b = item['B']
            c = item['C']
            label = (1, 0)

            # `label` is the one more similar to A.
            # We swap B and C if C is more like A when training and leave them untouched during evaluation.
            if item['label'] == 'C':
                if mode == 'train':
                    b, c = c, b
                else:
                    label = (0, 1)

            examples.append((a, b, c, label))
    return examples


def _augment_examples(examples: list[tuple[str, str, str, tuple[int]]]) -> list[tuple[str, str, str, tuple[int]]]:
    augmented = []
    for a, b, c, label in examples:
        augmented.append((a, b, c, (1, 0)))
        augmented.append((a, c, b, (0, 1)))
        augmented.append((b, a, c, (1, 0)))
        augmented.append((b, c, a, (0, 1)))
        augmented.append((c, c, a, (1, 0)))
        augmented.append((c, a, c, (0, 1)))
    return augmented


def _encode_examples(examples: list[tuple[str, str, str, tuple[int]]]) -> tuple[dict, dict, list[tuple[int, int]]]:
    ab, ac, labels = [], [], []
    for a, b, c, label in examples:
        ab.append((a, b))
        ac.append((a, c))
        labels.append(label)
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-chinese')
    ab = tokenizer(ab, truncation=True, padding='max_length', return_tensors='tf')
    ac = tokenizer(ac, truncation=True, padding='max_length', return_tensors='tf')
    return ab, ac, labels


def get_dataset(mode: str, batch_size: int) -> tf.data.Dataset:
    data_path = pathlib.Path('data')
    _download_data(data_path)
    examples = _extract_examples((data_path / f'{mode}.json'), mode=mode)
    if mode == 'train':
        examples = _augment_examples(examples)
    ab, ac, labels = _encode_examples(examples)
    encoded = (
        (
            ab['input_ids'],
            ab['token_type_ids'],
            ab['attention_mask'],
            ac['input_ids'],
            ac['token_type_ids'],
            ac['attention_mask'],
        ),
        labels,
    )
    dataset = tf.data.Dataset.from_tensor_slices(encoded)
    if mode == 'train':
        dataset = dataset.shuffle(1000).batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
