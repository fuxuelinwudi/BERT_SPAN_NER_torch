# -*- coding: utf-8 -*-

import os
import random
import numpy as np

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(2022)


class BERT_SPAN(BertPreTrainedModel):
    def __init__(self, config):
        super(BERT_SPAN, self).__init__(config)

        self.num_tags = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_classifier = nn.Linear(config.hidden_size, self.num_tags)
        self.end_classifier = nn.Linear(config.hidden_size, self.num_tags)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, start_ids=None, end_ids=None):

        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        sequence_output = self.dropout(sequence_output)

        start_logits = self.start_classifier(sequence_output)
        end_logits = self.end_classifier(sequence_output)

        output = (start_logits,) + (end_logits,) + (sequence_output,)

        if start_ids is not None and end_ids is not None:


            loss_fct = nn.CrossEntropyLoss()

            start_logits = start_logits.view(-1, self.num_tags)
            end_logits = end_logits.view(-1, self.num_tags)

            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_ids.view(-1)[active_loss]
            active_end_labels = end_ids.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)

            loss = start_loss + end_loss

            output = (loss, ) + output

        return output
