"""Script to serialize the predictions with the confidence values for the
models."""
import argparse
import json
import os
import random
from argparse import Namespace
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, \
    BertTokenizer

from models.data_loader import get_collate_fn, get_dataset
from models.model_builder import CNN_MODEL, LSTM_MODEL


class BertModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model

    def forward(self, input, attention_mask, labels):
        return self.model(input, attention_mask=attention_mask)[0]


def get_model_embedding_emb(model):
    if args.model == 'trans':
        return model.bert.embeddings.embedding.word_embeddings
    else:
        return model.embedding.embedding


def generate_pred(model_path):
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    model_args = Namespace(**checkpoint['args'])
    if args.model == 'lstm':
        model = LSTM_MODEL(tokenizer, model_args,
                           n_labels=checkpoint['args']['labels']).to(device)
        model.load_state_dict(checkpoint['model'])
    elif args.model == 'trans':
        transformer_config = BertConfig.from_pretrained('bert-base-uncased',
                                                        num_labels=model_args.labels)
        model_cp = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', config=transformer_config).to(
            device)
        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
        model_cp.load_state_dict(checkpoint['model'])
        model = BertModelWrapper(model_cp)
    else:
        model = CNN_MODEL(tokenizer, model_args,
                          n_labels=checkpoint['args']['labels']).to(device)
        model.load_state_dict(checkpoint['model'])

    model.train()

    pad_to_max = False

    coll_call = get_collate_fn(dataset=args.dataset, model=args.model)

    return_attention_masks = args.model == 'trans'

    collate_fn = partial(coll_call, tokenizer=tokenizer, device=device,
                         return_attention_masks=return_attention_masks,
                         pad_to_max_length=pad_to_max)
    test = get_dataset(path=args.dataset_dir, mode=args.split,
                       dataset=args.dataset)
    batch_size = args.batch_size if args.batch_size is not None else \
        model_args.batch_size
    test_dl = DataLoader(batch_size=batch_size, dataset=test, shuffle=False,
                         collate_fn=collate_fn)

    # PREDICTIONS
    predictions_path = model_path + '.predictions'
    predictions = defaultdict(lambda: [])
    for batch in tqdm(test_dl, desc='Running test prediction... '):
        if args.model == 'trans':
            logits = model(batch[0], attention_mask=batch[1],
                           labels=batch[2].long())
        else:
            logits = model(batch[0])
        logits = logits.detach().cpu().numpy().tolist()
        predicted = np.argmax(np.array(logits), axis=-1)
        predictions['class'] += predicted.tolist()
        predictions['logits'] += logits

        with open(predictions_path, 'w') as out:
            json.dump(predictions, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        help="Path to the direcory with the datasets",
                        default='data/e-SNLI/dataset/',
                        type=str)
    parser.add_argument("--split", help="Which split of the dataset",
                        default='test', type=str,
                        choices=['train', 'test'])
    parser.add_argument("--dataset", help="Which dataset", default='snli',
                        type=str, choices=['snli', 'imdb', 'tweet'])
    parser.add_argument("--model", help="Which model", default='cnn',
                        choices=['cnn', 'lstm', 'trans'], type=str)
    parser.add_argument("--models_dir",
                        help="Path where the models can be found",
                        default='snli_bert', type=str)
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--output_dir",
                        help="Path where the saliencies will be serialized",
                        default='saliency_snli_lime',
                        type=str)
    parser.add_argument("--sw", help="Sliding window", type=int, default=1)
    parser.add_argument("--saliency", help="Saliency type", nargs='+')
    parser.add_argument("--batch_size",
                        help="Batch size for explanation generation", type=int,
                        default=None)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(args, flush=True)

    models_dir = args.models_dir
    base_model_name = models_dir.split('/')[-1]
    for model in range(1, 6):
        generate_pred(os.path.join(models_dir + f'_{model}'))
