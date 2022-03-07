# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score


# 字符级验证方式
def seq_f1_with_mask(all_true_labels, all_pred_labels, all_label_mask, label_vocab):

    assert len(all_true_labels) == len(all_pred_labels), (len(all_true_labels), len(all_pred_labels))
    assert len(all_true_labels) == len(all_label_mask), (len(all_true_labels), len(all_label_mask))

    true_labels = []
    pred_labels = []

    sample_num = len(all_true_labels)
    for i in range(sample_num):
        tmp_true = []
        tmp_pred = []

        assert len(all_true_labels[i]) == len(all_pred_labels[i]), (len(all_true_labels[i]), len(all_pred_labels[i]))
        assert len(all_true_labels[i]) == len(all_label_mask[i]), (len(all_true_labels[i]), len(all_label_mask[i]))

        real_seq_length = np.sum(all_label_mask[i])
        for j in range(1, real_seq_length - 1):  # skip [CLS] and [SEP]
            if all_label_mask[i][j] == 1:
                tmp_true.append(label_vocab.convert_id_to_item(all_true_labels[i][j]).replace("M-", "I-"))
                tmp_pred.append(label_vocab.convert_id_to_item(all_pred_labels[i][j]).replace("M-", "I-"))

        true_labels.append(tmp_true)
        pred_labels.append(tmp_pred)

    acc = accuracy_score(true_labels, pred_labels)
    p = precision_score(true_labels, pred_labels)
    r = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    acc, p, r, f1 = round(acc, 4), round(p, 4), round(r, 4), round(f1, 4)

    return acc, p, r, f1, true_labels, pred_labels


# 实体级评测方式
class SpanEntityScore(object):
    def __init__(self, label_vocab):
        self.label_vocab = label_vocab
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return round(recall, 4), round(precision, 4), round(f1, 4)

    def result(self):
        class_info = {}
        origin_counter = Counter([self.label_vocab.convert_ids_to_items([x[0]])[0] for x in self.origins])
        found_counter = Counter([self.label_vocab.convert_ids_to_items([x[0]])[0] for x in self.founds])
        right_counter = Counter([self.label_vocab.convert_ids_to_items([x[0]])[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'precision': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([pre_entity for pre_entity in pred_subject if pre_entity in true_subject])


def bert_extract_item(start_ids, end_ids):
    S = []
    for i, start_id in enumerate(start_ids):
        if start_id == 0:
            continue
        for j, end_id in enumerate(end_ids[i:]):
            if start_id == end_id:
                S.append((start_id, i, i + j))
                break
    return S
