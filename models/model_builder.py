"""Training utilities."""
import os
from argparse import Namespace

import numpy as np
import torch
from torch.nn import functional as F, init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from transformers import PreTrainedTokenizer

_glove_path = "glove.6B.{}d.txt".format


class EarlyStopping:
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)


def _get_glove_embeddings(embedding_dim: int, glove_dir: str):
    word_to_index = {}
    word_vectors = []

    with open(os.path.join(glove_dir, _glove_path(embedding_dim))) as fp:
        for line in tqdm(fp.readlines(),
                         desc=f'Loading Glove embeddings from {glove_dir}, '
                         f'dimension {embedding_dim}'):
            line = line.split(" ")

            word = line[0]
            word_to_index[word] = len(word_to_index)

            vec = np.array([float(x) for x in line[1:]])
            word_vectors.append(vec)

    return word_to_index, word_vectors


def get_embeddings(embedding_dim: int, embedding_dir: str,
                   tokenizer: PreTrainedTokenizer):
    """
    :return: a tensor with the embedding matrix - ids of words are from vocab
    """
    word_to_index, word_vectors = _get_glove_embeddings(embedding_dim,
                                                        embedding_dir)

    embedding_matrix = np.zeros((len(tokenizer), embedding_dim))

    for id in range(0, max(tokenizer.vocab.values()) + 1):
        word = tokenizer.ids_to_tokens[id]
        if word not in word_to_index:
            word_vector = np.random.rand(embedding_dim)
        else:
            word_vector = word_vectors[word_to_index[word]]

        embedding_matrix[id] = word_vector

    return torch.nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float),
                              requires_grad=True)


class LSTM_MODEL(torch.nn.Module):

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 args: Namespace,
                 n_labels: int,
                 device='cuda'):
        super().__init__()
        self.args = args
        self.n_labels = n_labels
        self.device = device

        self.embedding = torch.nn.Embedding(len(tokenizer), args.embedding_dim)
        self.embedding.weight = get_embeddings(args.embedding_dim,
                                               args.embedding_dir,
                                               tokenizer)

        self.enc_p = torch.nn.LSTM(input_size=args.embedding_dim,
                                   hidden_size=args.hidden_lstm,
                                   num_layers=args.num_layers,
                                   bidirectional=True,
                                   dropout=args.dropout,
                                   batch_first=True)

        self.dropout = torch.nn.Dropout(args.dropout)

        self.relu = torch.nn.Sigmoid()

        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_layers.append(
            torch.nn.Linear(args.hidden_lstm * 2, args.hidden_sizes[0]))
        for i in range(1, len(args.hidden_sizes)):
            self.hidden_layers.append(torch.nn.Linear(args.hidden_sizes[i - 1],
                                                      args.hidden_sizes[i]))

        self.hidden_layers.append(
            torch.nn.Linear(args.hidden_sizes[-1], n_labels))

        self.hidden_layers.apply(self.init_weights)
        for name, param in self.enc_p.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.xavier_uniform_(param)

    def init_weights(self, w):
        if isinstance(w, torch.nn.Linear):
            torch.nn.init.xavier_normal_(w.weight)
            w.bias.data.fill_(0.01)

    def forward(self, input, seq_lenghts=None):
        embedded = self.embedding(input)

        if seq_lenghts == None:
            seq_lenghts = []
            for instance in input:
                ilen = 1
                for _i in range(len(instance) - 1, -1, -1):
                    if instance[_i] != 0:
                        ilen = _i + 1
                        break
                seq_lenghts.append(ilen)

        packed_input = pack_padded_sequence(embedded, seq_lenghts,
                                            batch_first=True,
                                            enforce_sorted=False)
        lstm_out, self.hidden = self.enc_p(packed_input)
        output, input_sizes = pad_packed_sequence(lstm_out, batch_first=True)

        last_idxs = (input_sizes - 1).to(self.device)
        output = torch.gather(output, 1,
                              last_idxs.view(-1, 1).unsqueeze(2).repeat(1, 1,
                                                                        self.args.hidden_lstm * 2)).squeeze()

        for hidden_layer in self.hidden_layers:
            output = self.dropout(hidden_layer(output))

        return output


class CNN_MODEL(torch.nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizer, args: Namespace,
                 n_labels: int = 2):
        super().__init__()
        self.n_labels = n_labels
        self.args = args

        self.embedding = torch.nn.Embedding(len(tokenizer), args.embedding_dim)

        self.dropout = torch.nn.Dropout(args.dropout)

        self.embedding.weight = get_embeddings(args.embedding_dim,
                                               args.embedding_dir, tokenizer)

        self.conv_layers = torch.nn.ModuleList(
            [torch.nn.Conv2d(args.in_channels, args.out_channels,
                             (kernel_height, args.embedding_dim),
                             args.stride, args.padding)
             for kernel_height in args.kernel_heights])

        self.final = torch.nn.Linear(
            len(args.kernel_heights) * args.out_channels, n_labels)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)
        activation = F.relu(conv_out.squeeze(3))
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)

        return max_out

    def forward(self, input):
        input = self.embedding(input)
        input = input.unsqueeze(1)
        input = self.dropout(input)

        conv_out = [self.conv_block(input, self.conv_layers[i]) for i in
                    range(len(self.conv_layers))]
        all_out = torch.cat(conv_out, 1)
        fc_in = self.dropout(all_out)
        logits = self.final(fc_in)
        return logits
