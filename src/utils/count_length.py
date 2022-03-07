# coding:utf-8

import json

length = []
with open('../../raw_data/train.json', 'r', encoding='utf-8') as f:
    for line_id, line in enumerate(f):
        data = json.loads(line)
        text = data['text']
        length.append(len(text))
with open('../../raw_data/dev.json', 'r', encoding='utf-8') as f:
    for line_id, line in enumerate(f):
        data = json.loads(line)
        text = data['text']
        length.append(len(text))

print(min(length))
print(max(length))
print(sum(length)/len(length))
