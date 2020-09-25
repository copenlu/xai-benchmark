"""Script to serialize the saliencies from the LIME method."""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from lime.lime_text import LimeTextExplainer
from pypapi import events, papi_high as high
from tqdm import tqdm
from transformers import BertConfig, BertForSequenceClassification, \
    BertTokenizer

from models.data_loader import get_dataset
from models.model_builder import CNN_MODEL, LSTM_MODEL


class BertModelWrapper(nn.Module):
    def __init__(self, model, device, tokenizer, args):
        super(BertModelWrapper, self).__init__()
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, token_ids):
        results = []
        token_ids = [[int(i) for i in instance_ids.split(' ') if i != ''] for
                     instance_ids in token_ids]
        for i in tqdm(range(0, len(token_ids), self.args.batch_size),
                      'Building a local approximation...'):
            batch_ids = token_ids[i:i + self.args.batch_size]
            max_batch_id = min(max([len(_l) for _l in batch_ids]), 512)
            batch_ids = [_l[:max_batch_id] for _l in batch_ids]
            padded_batch_ids = [
                _l + [self.tokenizer.pad_token_id] * (max_batch_id - len(_l))
                for _l in batch_ids]
            tokens_tensor = torch.tensor(padded_batch_ids).to(self.device)
            logits = self.model(tokens_tensor.long(),
                                attention_mask=tokens_tensor.long() > 0)
            results += logits[0].detach().cpu().numpy().tolist()
        return np.array(results)


class ModelWrapper(nn.Module):
    def __init__(self, model, device, tokenizer, args):
        super(ModelWrapper, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.args = args

    def forward(self, token_ids):
        results = []
        token_ids = [[int(i) for i in instance_ids.split(' ') if i != ''] for
                     instance_ids in token_ids]
        for i in tqdm(range(0, len(token_ids), self.args.batch_size),
                      'Building a local approximation...'):
            batch_ids = token_ids[i:i + self.args.batch_size]
            max_batch_id = max([len(_l) for _l in batch_ids])
            padded_batch_ids = [
                _l + [self.tokenizer.pad_token_id] * (max_batch_id - len(_l))
                for _l in batch_ids]
            tokens_tensor = torch.tensor(padded_batch_ids).to(self.device)
            logits = self.model(tokens_tensor)
            results += logits.detach().cpu().numpy().tolist()
        return np.array(results)


def generate_saliency(model_path, saliency_path):
    test = get_dataset(path=args.dataset_dir, mode=args.split,
                       dataset=args.dataset)
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**checkpoint['args'])

    if args.model == 'trans':
        model_args.batch_size = 7
        transformer_config = BertConfig.from_pretrained('bert-base-uncased',
                                                        num_labels=model_args.labels)
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', config=transformer_config).to(device)
        model.load_state_dict(checkpoint['model'])
        modelw = BertModelWrapper(model, device, tokenizer, model_args)
    else:
        if args.model == 'lstm':
            model_args.batch_size = 200
            model = LSTM_MODEL(tokenizer, model_args,
                               n_labels=checkpoint['args']['labels'],
                               device=device).to(device)
        else:
            model_args.batch_size = 300
            model = CNN_MODEL(tokenizer, model_args,
                              n_labels=checkpoint['args']['labels']).to(device)

        model.load_state_dict(checkpoint['model'])
        modelw = ModelWrapper(model, device, tokenizer, model_args)

    modelw.eval()

    explainer = LimeTextExplainer()
    saliency_flops = []

    with open(saliency_path, 'w') as out:
        for instance in tqdm(test):
            # SALIENCY
            if not args.no_time:
                high.start_counters([events.PAPI_FP_OPS, ])

            saliencies = []
            if args.dataset in ['imdb', 'tweet']:
                token_ids = tokenizer.encode(instance[0])
            else:
                token_ids = tokenizer.encode(instance[0], instance[1])

            if len(token_ids) < 6:
                token_ids = token_ids + [tokenizer.pad_token_id] * (
                            6 - len(token_ids))
            try:
                exp = explainer.explain_instance(
                    " ".join([str(i) for i in token_ids]), modelw,
                    num_features=len(token_ids),
                    top_labels=args.labels)
            except Exception as e:
                print(e)
                if not args.no_time:
                    x = high.stop_counters()[0]
                    saliency_flops.append(x)

                for token_id in token_ids:
                    token_id = int(token_id)
                    token_saliency = {
                        'token': tokenizer.ids_to_tokens[token_id]
                    }
                    for cls_ in range(args.labels):
                        token_saliency[int(cls_)] = 0
                    saliencies.append(token_saliency)

                out.write(json.dumps({'tokens': saliencies}) + '\n')
                out.flush()

                continue

            if not args.no_time:
                x = high.stop_counters()[0]
                saliency_flops.append(x)

            # SERIALIZE
            explanation = {}
            for cls_ in range(args.labels):
                cls_expl = {}
                for (w, s) in exp.as_list(label=cls_):
                    cls_expl[int(w)] = s
                explanation[cls_] = cls_expl

            for token_id in token_ids:
                token_id = int(token_id)
                token_saliency = {'token': tokenizer.ids_to_tokens[token_id]}
                for cls_ in range(args.labels):
                    token_saliency[int(cls_)] = explanation[cls_].get(token_id,
                                                                      None)
                saliencies.append(token_saliency)

            out.write(json.dumps({'tokens': saliencies}) + '\n')
            out.flush()

    return saliency_flops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_time",
                        help="Whether to output the time for generation in "
                             "flop",
                        action='store_true',
                        default=False)
    parser.add_argument("--dataset", help="Which dataset", default='snli',
                        type=str, choices=['snli', 'imdb', 'tweet'])
    parser.add_argument("--dataset_dir",
                        help="Path to the direcory with the datasets",
                        default='data/e-SNLI/dataset/',
                        type=str)
    parser.add_argument("--model", help="Which model", default='cnn',
                        choices=['cnn', 'lstm', 'trans'], type=str)
    parser.add_argument("--model_path",
                        help="Path to the model",
                        default='snli_bert', type=str)
    parser.add_argument("--output_dir",
                        help="Path where the saliency will be serialized",
                        default='', type=str)
    parser.add_argument("--gpu", help="Flag for running on gpu",
                        action='store_true', default=False)
    parser.add_argument("--gpu_id", help="", default=0, type=int)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="Number of target labels", type=int,
                        default=3)
    parser.add_argument("--split", help="", default='test', type=str,
                        choices=['train', 'test'])

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device(f"cuda") if args.gpu else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(args, flush=True)

    model_path = args.model_path
    print(model_path, flush=True)
    all_flops = generate_saliency(model_path, os.path.join(args.output_dir,
                                                           f'{model_path.split("/")[-1]}_lime'))
    print('FLOPS', np.average(all_flops), np.std(all_flops))
