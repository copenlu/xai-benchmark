"""Preprocessing for IMDB dataset.
7th and 8th splits are used for validation and test correspondingly
as the 9th split doesn't have annotations of the salient words."""

import csv
import os
import random

import numpy as np

random.seed(73)
np.random.seed(73)

_CLS_ID = {'pos': 1, 'neg': 0}

dataset = []
for dir in ['data/movie/noRats_neg', 'data/movie/withRats_pos']:
    cls = _CLS_ID[dir.split('_')[-1]]
    for file in os.scandir(dir):
        with open(file) as out:
            dataset.append((out.read(), file.name, cls))

dataset = np.array(dataset)

output_splits_dir = 'data/imdb_rats/'
os.makedirs(output_splits_dir, exist_ok=True)

train = []
val = []
test = []

for instance in dataset:
    if instance[1].split('_')[1].startswith('7'):
        val.append(instance)
    elif instance[1].split('_')[1].startswith('8'):
        test.append(instance)
    else:
        train.append(instance)

print(len(train), len(test), len(val))

with open(output_splits_dir + f'train.tsv', 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for row in train:
        csv_writer.writerow(row)

with open(output_splits_dir + f'dev.tsv', 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for row in val:
        csv_writer.writerow(row)

with open(output_splits_dir + f'test.tsv', 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for row in test:
        csv_writer.writerow(row)
