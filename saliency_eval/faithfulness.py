"""Computing Faithfulness measure for the saliency scores."""
import argparse
import os
import random
from argparse import Namespace
from functools import partial

import numpy as np
import torch
from sklearn.metrics import auc
from transformers import BertConfig, BertForSequenceClassification, \
    BertTokenizer

from models import train_lstm_cnn, train_transformers
from models.data_loader import BucketBatchSampler, DatasetSaliency, \
    collate_threshold, get_collate_fn, get_dataset
from models.model_builder import CNN_MODEL, LSTM_MODEL


def get_model(model_path):
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    model_args = Namespace(**checkpoint['args'])
    if args.model == 'lstm':
        model_cp = LSTM_MODEL(tokenizer, model_args,
                              n_labels=checkpoint['args']['labels']).to(device)
    elif args.model == 'trans':
        transformer_config = BertConfig.from_pretrained('bert-base-uncased',
                                                        num_labels=model_args.labels)
        model_cp = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', config=transformer_config).to(
            device)
    else:
        model_cp = CNN_MODEL(tokenizer, model_args,
                             n_labels=checkpoint['args']['labels']).to(device)

    model_cp.load_state_dict(checkpoint['model'])

    return model_cp, model_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for running on gpu",
                        action='store_true', default=False)
    parser.add_argument("--saliency", help='Name of the saliency', type=str)
    parser.add_argument("--dataset", choices=['snli', 'imdb', 'tweet'])
    parser.add_argument("--dataset_dir",
                        help="Path to the direcory with the datasets",
                        default='data/e-SNLI/dataset/',
                        type=str)
    parser.add_argument("--test_saliency_dir",
                        help="Path to the saliency files", type=str)
    parser.add_argument("--model_path", help="Directory with all of the models",
                        type=str, nargs='+')
    parser.add_argument("--models_dir", help="Directory with all of the models",
                        type=str)
    parser.add_argument("--model", help="Type of model",
                        choices=['trans', 'lstm', 'cnn'])

    args = parser.parse_args()

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    thresholds = list(range(0, 110, 10))
    aucs = []

    coll_call = get_collate_fn(dataset=args.dataset, model=args.model)
    return_attention_masks = args.model == 'trans'

    if args.model == 'trans':
        eval_fn = train_transformers.eval_model
    else:
        eval_fn = train_lstm_cnn.eval_model

    for model_path in os.listdir(args.models_dir):
        if model_path.endswith('.predictions'):
            continue
        print('Model', model_path, flush=True)
        model_full_path = os.path.join(args.models_dir, model_path)
        model, model_args = get_model(model_full_path)

        random.seed(model_args.seed)
        torch.manual_seed(model_args.seed)
        torch.cuda.manual_seed_all(model_args.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(model_args.seed)

        model_scores = []
        for threshold in thresholds:
            collate_fn = partial(collate_threshold,
                                 tokenizer=tokenizer,
                                 device=device,
                                 return_attention_masks=return_attention_masks,
                                 pad_to_max_length=False,
                                 threshold=threshold,
                                 collate_orig=coll_call,
                                 n_classes=3 if args.dataset in ['snli',
                                                                 'tweet']
                                 else 2)

            saliency_path_test = os.path.join(args.test_saliency_dir,
                                              f'{model_path}_{args.saliency}')
            test = get_dataset(mode='test', dataset=args.dataset,
                               path=args.dataset_dir)
            test = DatasetSaliency(test, saliency_path_test)

            test_dl = BucketBatchSampler(batch_size=model_args.batch_size,
                                         dataset=test,
                                         collate_fn=collate_fn)

            results = eval_fn(model, test_dl, model_args.labels)
            model_scores.append(results[2])

        print(thresholds, model_scores)
        aucs.append(auc(thresholds, model_scores))

    print(f'{np.mean(aucs):.2f} ($\pm${np.std(aucs):.2f})')
