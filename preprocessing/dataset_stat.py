"""Dataset statistics of the rationales."""
import argparse

import numpy as np
from transformers import BertTokenizer

from models.data_loader import IMDBDataset, NLIDataset, TwitterDataset, \
    _twitter_label
from models.saliency_utils import get_gold_saliency_esnli, \
    get_gold_saliency_imdb, get_gold_saliency_tweet


def get_dataset(path, mode='test'):
    if args.dataset == 'snli':
        ds = NLIDataset(path, type=mode, salient_features=True)
    elif args.dataset == 'imdb':
        ds = IMDBDataset(path, type=mode, salient_features=True)
    elif args.dataset == 'tweet':
        ds = TwitterDataset(path, type=mode, salient_features=True)
    return ds


saliency_func_map = {
    'snli': get_gold_saliency_esnli,
    'imdb': get_gold_saliency_imdb,
    'tweet': get_gold_saliency_tweet
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Which dataset", default='snli',
                        type=str, choices=['snli', 'imdb', 'tweet'])
    parser.add_argument("--dataset_dir",
                        help="Path to the direcory with the datasets",
                        default='data/e-SNLI/dataset/',
                        type=str)

    args = parser.parse_args()
    print(args, flush=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for mode in ['test']:
        ds = get_dataset(args.dataset_dir, mode=mode)
        saliency_gold_f = saliency_func_map[args.dataset]

        len_ = []
        rat_len = []

        for instance in ds:
            if args.dataset == 'snli':
                instance_gold = instance[2]
            elif args.dataset == 'imdb':
                instance_gold = instance[1]
            elif args.dataset == 'tweet':
                instance_gold = _twitter_label[instance[1]]

            if args.dataset == 'snli':
                token_ids = tokenizer.encode(instance[0], instance[1])
            else:
                token_ids = tokenizer.encode(instance[0])

            gold_saliency = saliency_gold_f(instance,
                                            tokenizer.convert_ids_to_tokens(
                                                token_ids),
                                            [tokenizer.cls_token,
                                             tokenizer.sep_token,
                                             tokenizer.pad_token], tokenizer)

            len_.append(len(gold_saliency))
            rat_len.append(sum(gold_saliency))

    print(np.mean(len_), np.std(len_))
    print(np.mean(rat_len), np.std(rat_len))
