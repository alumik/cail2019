import os
import json

from typing import Sequence


def extract_text_tuples(path: str) -> Sequence:
    text_tuples = []
    with open(path, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            items = json.loads(line)
            a = list(items['A'].replace('\n', ''))
            b = list(items['B'].replace('\n', ''))
            c = list(items['C'].replace('\n', ''))
            if items['label'] == 'C':
                b, c = c, b
            text_tuples.append((a, b, c))
    return text_tuples


def make_input_file(text_tuples: Sequence, path: str, max_len: int, mode: str = 'test'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as outfile:
        for a, b, c in text_tuples:
            tokens_a = a[-(max_len // 2):]
            tokens_b = b[-(max_len - max_len // 2):]
            tokens_c = c[-(max_len - max_len // 2):]
            line_ab = ' '.join(tokens_a) + ' [SEP] ' + ' '.join(tokens_b) + '\t1'
            line_ac = ' '.join(tokens_a) + ' [SEP] ' + ' '.join(tokens_c) + '\t0'
            outfile.write(line_ab + '\n')
            outfile.write(line_ac + '\n')
            if mode == 'train':
                line_ba = ' '.join(tokens_b) + ' [SEP] ' + ' '.join(tokens_a) + '\t1'
                line_ca = ' '.join(tokens_c) + ' [SEP] ' + ' '.join(tokens_a) + '\t0'
                line_bc = ' '.join(tokens_b) + ' [SEP] ' + ' '.join(tokens_c) + '\t0'
                line_cb = ' '.join(tokens_b) + ' [SEP] ' + ' '.join(tokens_c) + '\t0'
                outfile.write(line_ba + '\n')
                outfile.write(line_ca + '\n')
                outfile.write(line_bc + '\n')
                outfile.write(line_cb + '\n')
