"""Script for training models for the TSE dataset."""
import argparse
import random
from functools import partial

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AdamW, BertConfig, BertForSequenceClassification, \
    BertTokenizer, get_constant_schedule_with_warmup

from models import train_lstm_cnn, train_transformers
from models.data_loader import BucketBatchSampler, TwitterDataset, \
    collate_twitter
from models.model_builder import CNN_MODEL, EarlyStopping, LSTM_MODEL


def get_model():
    if args.model == 'trans':
        transformer_config = BertConfig.from_pretrained('bert-base-uncased',
                                                        num_labels=args.labels)
        if args.init_only:
            model = BertForSequenceClassification(config=transformer_config).to(
                device)
        else:
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                config=transformer_config).to(device)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if
                           not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in param_optimizer if
                           any(nd in n for nd in no_decay)], 'weight_decay': 0.0
            }]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        es = EarlyStopping(patience=args.patience, percentage=False, mode='max',
                           min_delta=0.0)
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=0.05)
    else:
        if args.model == 'cnn':
            model = CNN_MODEL(tokenizer, args, n_labels=args.labels).to(device)
        elif args.model == 'lstm':
            model = LSTM_MODEL(tokenizer, args, n_labels=args.labels).to(device)

        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, verbose=True)
        es = EarlyStopping(patience=args.patience, percentage=False, mode='max',
                           min_delta=0.0)

    return model, optimizer, scheduler, es


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--random_seed", default=False, action='store_true')
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="num of lables", type=int, default=3)

    parser.add_argument("--dataset_dir", help="Path to the train datasets",
                        type=str)
    parser.add_argument("--model_path",
                        help="Path where the model will be serialized",
                        nargs='+',
                        type=str)

    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=5e-5)
    parser.add_argument("--epochs", help="Epochs number", type=int, default=100)
    parser.add_argument("--mode", help="Mode for the script", type=str,
                        default='train', choices=['train', 'test'])
    parser.add_argument("--model", help="Model for training", type=str,
                        default='trans',
                        choices=['lstm', 'cnn', 'trans'])
    parser.add_argument("--init_only", help="Whether to train the model",
                        action='store_true', default=False)

    # RNN + CNN  ARGUMENTS
    parser.add_argument("--patience", help="Early stopping patience", type=int,
                        default=5)
    parser.add_argument("--embedding_dir",
                        help="Path to directory with pretrained embeddings",
                        default='./', type=str)
    parser.add_argument("--dropout",
                        help="Path to directory with pretrained embeddings",
                        default=0.1, type=float)
    parser.add_argument("--embedding_dim", help="Dimension of embeddings",
                        choices=[50, 100, 200, 300], default=100,
                        type=int)

    # RNN ARGUMENTS
    parser.add_argument("--hidden_lstm",
                        help="Number of units in the hidden layer", default=300,
                        type=int)
    parser.add_argument("--num_layers", help="Number of rnn layers", default=4,
                        type=int)
    parser.add_argument("--hidden_sizes",
                        help="Number of units in the hidden layer",
                        default=[200, 50], type=int,
                        nargs='+')

    # CNN ARGUMENTS
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=100)
    parser.add_argument("--kernel_heights", help="filter windows", type=int,
                        nargs='+', default=[2, 3, 4, 5])
    parser.add_argument("--stride", help="stride", type=int, default=1)
    parser.add_argument("--padding", help="padding", type=int, default=0)

    args = parser.parse_args()

    if args.random_seed:
        seed = random.randint(0, 10000)
        args.seed = seed

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    collate_fn = partial(collate_twitter,
                         tokenizer=tokenizer,
                         device=device,
                         return_attention_masks=args.model == 'trans',
                         pad_to_max_length=False)
    print(args, flush=True)
    sort_key = lambda x: len(x[0])

    if args.model == 'trans':
        transformer_config = BertConfig.from_pretrained('bert-base-uncased',
                                                        num_labels=args.labels)
        train_fn = train_transformers.train_model
        eval_fn = train_transformers.eval_model
    else:
        train_fn = train_lstm_cnn.train_model
        eval_fn = train_lstm_cnn.eval_model

    if args.mode == 'test':
        scores = []
        for model_path in args.model_path:
            model, _, _, _, = get_model()
            test = TwitterDataset(args.dataset_dir, type='test')
            test_dl = BucketBatchSampler(batch_size=args.batch_size,
                                         sort_key=sort_key, dataset=test,
                                         collate_fn=collate_fn)

            checkpoint = torch.load(model_path)

            model.load_state_dict(checkpoint['model'])
            _, _, _, _, labels, pred = eval_fn(model, test_dl, args.labels)
            p, r, f1, _ = precision_recall_fscore_support(labels, pred,
                                                          average='macro')
            scores.append((p, r, f1))

        for i, name in zip(range(len(scores[0])), ['p', 'r', 'f1']):
            l = [model_scores[i] for model_scores in scores]
            print(name, np.average(l), np.std(l))

    else:
        print("Loading datasets...")
        model, optimizer, scheduler, es = get_model()
        train = TwitterDataset(args.dataset_dir, type='train')
        dev = TwitterDataset(args.dataset_dir, type='dev')
        train_dl = BucketBatchSampler(batch_size=args.batch_size,
                                      sort_key=sort_key, dataset=train,
                                      collate_fn=collate_fn)
        dev_dl = BucketBatchSampler(batch_size=args.batch_size,
                                    sort_key=sort_key, dataset=dev,
                                    collate_fn=collate_fn)
        num_train_optimization_steps = int(
            args.epochs * len(train) / args.batch_size)

        if args.init_only:
            best_model_w, best_perf = model.state_dict(), {'val_f1': 0}
        else:
            if args.model == 'trans':
                best_model_w, best_perf = train_fn(model, train_dl, dev_dl,
                                                   optimizer, scheduler,
                                                   args.epochs, args.labels, es)
            else:
                best_model_w, best_perf = train_fn(model, train_dl, dev_dl,
                                                   optimizer, scheduler,
                                                   args.epochs, es)

        checkpoint = {
            'performance': best_perf,
            'args': vars(args),
            'model': best_model_w
        }

        print(best_perf)
        print(args)

        torch.save(checkpoint, args.model_path[0])

        print('F1', best_perf['val_f1'])
