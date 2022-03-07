# coding:utf-8

import json
from tqdm import trange
from argparse import ArgumentParser


def format_data(args):
    data_list = []
    with open(args.train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    words = ''
    labels = ''
    flag = 0
    for line in lines:
        if line == '\n':
            sample = words + '\t' + labels + '\n'
            data_list.append(sample)
            words = ''
            labels = ''
            flag = 0
            continue
        elif line == '  O\n':
            word = ' '
            label = 'O'
        else:
            word, label = line[0], line[2:]
            label = label.strip('\n')
        if flag == 1:
            words = words + '\002' + word
            labels = labels + '\002' + label
        else:
            words = words + word
            labels = labels + label
            flag = 1

    # 添加最后一行数据
    sample = words + '\t' + labels + '\n'
    data_list.append(sample)

    with open(args.out_train_path, 'w', encoding='utf-8') as f:
        for line_id, text in enumerate(data_list):
            if line_id < args.train_num:
                f.write(text)

    with open(args.out_dev_path, 'w', encoding='utf-8') as f:
        for line_id, text in enumerate(data_list):
            if line_id >= args.train_num:
                f.write(text)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--train_path', type=str, default='../../raw_data/train_500.txt')

    parser.add_argument('--out_train_path', type=str, default='../../raw_data/train.json')
    parser.add_argument('--out_dev_path', type=str, default='../../raw_data/dev.json')

    parser.add_argument('--train_num', type=int, default=400)
    parser.add_argument('--dev_num', type=int, default=100)

    args = parser.parse_args()

    format_data(args)