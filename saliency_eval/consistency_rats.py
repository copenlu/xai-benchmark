"""Evaluating Consistency Rationale measure."""
import argparse
import json
import os
import random
import traceback
from argparse import Namespace
from functools import partial

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from transformers import BertConfig, BertForSequenceClassification, \
    BertTokenizer

from models.data_loader import get_collate_fn, get_dataset
from models.model_builder import CNN_MODEL, LSTM_MODEL


def get_model(model_path, device, model_type, tokenizer):
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    model_args = Namespace(**checkpoint['args'])
    if model_type == 'lstm':
        model_cp = LSTM_MODEL(tokenizer, model_args,
                              n_labels=checkpoint['args']['labels']).to(device)
    elif model_type == 'trans':
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


def get_saliencies(saliency_path):
    result = []
    n_labels = 2 if args.dataset == 'imdb' else 3
    with open(saliency_path) as out:
        for i, line in enumerate(out):
            instance_saliency = json.loads(line)
            saliency = instance_saliency['tokens']
            token_pred_saliency = []
            for _cls in range(0, n_labels):
                for record in saliency:
                    token_pred_saliency.append(record[str(_cls)])

            result.append(token_pred_saliency)
    return result


def save_activation(self, inp, out):
    global activations
    activations.append(out[0])


def get_layer_activation(layer, model, instance):
    handle = None
    for name, module in model.named_modules():
        # partial to assign the layer name to each hook
        if name == layer:
            handle = module.register_forward_hook(save_activation)

    global activations
    activations = []
    with torch.no_grad():
        batch = collate_fn([instance])
        if args.model == 'trans':
            model(batch[0], attention_mask=batch[1], labels=batch[2])
        else:
            model(batch[0])

    if handle:
        handle.remove()

    activ1 = None
    try:
        activations = activations[0]
        if isinstance(activations, tuple) and len(activations) == 1:
            activations = activations[0]

        if isinstance(activations[0], torch.nn.utils.rnn.PackedSequence):
            output, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(
                activations[0], batch_first=True)
            last_idxs = (input_sizes - 1).to(model.device)
            activations = torch.gather(output, 1,
                                       last_idxs.view(-1, 1).unsqueeze(
                                           2).repeat(1, 1,
                                                     model.args.hidden_lstm *
                                                     2)).squeeze()

        activ1 = activations.detach().cpu().numpy().ravel().tolist()
    except Exception as e:
        print(e, flush=True)
        print(print(traceback.format_exc()), flush=True)
    return activ1


def get_model_dist(model1, model2, x, layers):
    dist = []
    for layer in layers:
        act1 = get_layer_activation(layer, model1, x)
        act2 = get_layer_activation(layer, model2, x)
        if not act1 or not act2:
            continue
        dist.append(np.mean(np.array(act1).ravel() - np.array(act2).ravel()))
    return np.mean(np.abs(dist))


def get_layer_names(model, dataset):
    if model == 'trans':
        layers = ['bert.encoder.layer.0', 'bert.encoder.layer.1',
                  'bert.encoder.layer.2', 'bert.encoder.layer.3',
                  'bert.encoder.layer.4', 'bert.encoder.layer.5',
                  'bert.encoder.layer.6', 'bert.encoder.layer.7',
                  'bert.encoder.layer.8', 'bert.encoder.layer.9',
                  'bert.encoder.layer.10', 'bert.encoder.layer.11',
                  'classifier']
    elif model == 'lstm':
        if dataset == 'snli':
            layers = ["embedding", "enc_p", "hidden_layers.0",
                      "hidden_layers.1", "hidden_layers.2"]
        else:
            layers = ["embedding", "enc_p", "hidden_layers.0",
                      "hidden_layers.1", "hidden_layers.2"]
    else:
        if dataset == 'snli':
            layers = ["embedding", "conv_layers.0", "conv_layers.1",
                      "conv_layers.2", "conv_layers.3", "final"]
        else:
            layers = ["embedding", "conv_layers.0", "conv_layers.1",
                      "conv_layers.2", "final"]

    return layers


def get_sal_dist(sal1, sal2):
    return np.mean(
        np.abs(np.array(sal1).reshape(-1) - np.array(sal2).reshape(-1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir_trained",
                        help="Directory with trained models", type=str)
    parser.add_argument("--model_dir_random",
                        help="Directory with randomly initialized and "
                             "not trained models", type=str)
    parser.add_argument("--saliency_dir_trained",
                        help="Directory with saliencies serialized for the "
                             "trained models", type=str)
    parser.add_argument("--saliency_dir_random",
                        help="Directory with saliencies"
                             "serialized for the "
                             "non-trained models",
                        type=str)
    parser.add_argument('--saliencies',
                        help='Names of the saliencies to be considered',
                        type=str, nargs='+')
    parser.add_argument('--model', help='Name of the model that the saliencies '
                                        'are computed for',
                        type=str, default='trans')
    parser.add_argument("--gpu", help="Flag for running on gpu",
                        action='store_true', default=False)
    parser.add_argument("--dataset",
                        help='Which dataset are the saliencies for',
                        choices=['snli', 'imdb', 'tweet'])
    parser.add_argument("--dataset_dir",
                        help="Path to the direcory with the datasets",
                        default='data/e-SNLI/dataset/', type=str)
    parser.add_argument("--per_layer", help='Whether to compute the Rationales '
                                            'across separate layers or all of '
                                            'them',
                        default=False, action='store_true')

    args = parser.parse_args()
    print(args, flush=True)

    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    np.random.seed(1)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for saliency in args.saliencies:
        models_trained = [_m for _m in os.listdir(args.model_dir_trained) if
                          not _m.endswith('.predictions')]
        saliency_trained = [
            os.path.join(args.saliency_dir_trained, _m + f'_{saliency}') for _m
            in models_trained]

        models_rand = [_m for _m in os.listdir(args.model_dir_random) if
                       not _m.endswith('.predictions')]
        saliency_rand = [
            os.path.join(args.saliency_dir_random, _m + f'_{saliency}') for _m
            in models_rand]

        return_attention_masks = args.model == 'trans'

        device = torch.device("cuda") if args.gpu else torch.device("cpu")
        test = get_dataset(mode='test', dataset=args.dataset,
                           path=args.dataset_dir)
        coll_call = get_collate_fn(dataset=args.dataset, model=args.model)
        collate_fn = partial(coll_call,
                             tokenizer=tokenizer,
                             device=device,
                             return_attention_masks=return_attention_masks,
                             pad_to_max_length=False,
                             collate_orig=coll_call,
                             n_classes=3 if args.dataset in ['snli',
                                                             'tweet'] else 2)

        layers = get_layer_names(args.model, args.dataset)

        precomputed = []
        for _f in os.scandir('consist_rat'):
            _f = _f.name
            if args.model in _f and args.dataset in _f and _f.startswith(
                    'precomp_'):
                precomputed.append(_f)

        diff_activation, diff_saliency = [], []
        for f in precomputed:
            act_distances = json.load(open('consist_rat/' + f))
            ids = [int(_n) for _n in f.split('_') if _n.isdigit()]
            model_p = f.split('_')[3]
            if model_p == 'not':
                saliencies1, saliencies2 = get_saliencies(
                    saliency_trained[ids[0]]), get_saliencies(
                    saliency_trained[ids[1]])
            elif model_p == 'rand':
                saliencies1, saliencies2 = get_saliencies(
                    saliency_rand[ids[0]]), get_saliencies(
                    saliency_rand[ids[1]])
            else:
                saliencies1, saliencies2 = get_saliencies(
                    saliency_rand[ids[0]]), get_saliencies(
                    saliency_trained[ids[1]])

            for inst_id in range(len(test)):
                try:
                    sal_dist = get_sal_dist(saliencies1[inst_id],
                                            saliencies2[inst_id])
                    act_dist = act_distances[inst_id]
                    diff_activation.append(act_dist)
                    diff_saliency.append(sal_dist)
                except:
                    continue

        if args.per_layer:
            for i in range(len(diff_activation[0])):
                acts = [np.abs(_dist[i]) for _dist in diff_activation]
                diff_act = MinMaxScaler().fit_transform([[_d] for _d in acts])
                diff_act = [_d[0] for _d in diff_act]

                diff_sal = MinMaxScaler().fit_transform(
                    [[_d] for _d in diff_saliency])
                diff_sal = [_d[0] for _d in diff_sal]

                sp = spearmanr(diff_act, diff_sal)
                print(f'{sp[0]:.3f}', flush=True, end=' ')

        acts = [np.abs(np.mean(_dist)) for _dist in diff_activation]
        diff_act = MinMaxScaler().fit_transform([[_d] for _d in acts])
        diff_act = [_d[0] for _d in diff_act]

        diff_sal = MinMaxScaler().fit_transform([[_d] for _d in diff_saliency])
        diff_sal = [_d[0] for _d in diff_sal]

        sp = spearmanr(diff_act, diff_sal)
        print()
        print(f'{sp[0]:.3f} ({sp[1]:.1e})', flush=True)
