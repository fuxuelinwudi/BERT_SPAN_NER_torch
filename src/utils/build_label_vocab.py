# coding:utf-8

import os
from argparse import ArgumentParser


def build_label(args):

    label_list = set()
    with open(args.train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = line.split()
            if len(data) == 2:
                if data[1] == 'O':
                    continue
                else:
                    label_list.add(data[1].split('-')[1])

    label_list = list(label_list)
    label_list = sorted(label_list)

    out_label_path = os.path.join(args.lebert_file_path, 'label.txt')
    with open(out_label_path, 'w', encoding='utf-8') as f:
        for i in label_list:
            f.writelines(i + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--train_path', type=str, default='../../raw_data/train_500.txt')
    parser.add_argument('--lebert_file_path', type=str, default='../../raw_data')

    args = parser.parse_args()

    os.makedirs(args.lebert_file_path, exist_ok=True)

    build_label(args)
