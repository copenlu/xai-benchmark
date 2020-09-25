"""Script for training LSTM and CNN models for the e-SNLI dataset."""
import argparse
import random
from functools import partial
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
from transformers import BertTokenizer

from models.data_loader import BucketBatchSampler, NLIDataset, collate_nli
from models.model_builder import CNN_MODEL, EarlyStopping, LSTM_MODEL


def train_model(model: torch.nn.Module,
                train_dl: BatchSampler, dev_dl: BatchSampler,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LambdaLR,
                n_epochs: int,
                early_stopping: EarlyStopping) -> (Dict, Dict):
    loss_f = torch.nn.CrossEntropyLoss()

    best_val, best_model_weights = {'val_f1': 0}, None

    for ep in range(n_epochs):
        model.train()
        for batch in tqdm(train_dl, desc='Training'):
            optimizer.zero_grad()
            logits = model(batch[0])
            loss = loss_f(logits, batch[1])
            loss.backward()
            optimizer.step()

        val_p, val_r, val_f1, val_loss, _, _ = eval_model(model, dev_dl)
        current_val = {
            'val_p': val_p, 'val_r': val_r, 'val_f1': val_f1,
            'val_loss': val_loss, 'ep': ep
        }

        print(current_val, flush=True)

        if current_val['val_f1'] > best_val['val_f1']:
            best_val = current_val
            best_model_weights = model.state_dict()

        scheduler.step(val_loss)
        if early_stopping.step(val_f1):
            print('Early stopping...')
            break

    return best_model_weights, best_val


def eval_model(model: torch.nn.Module, test_dl: BucketBatchSampler,
               measure=None):
    model.eval()

    loss_f = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        labels_all = []
        logits_all = []
        losses = []
        for batch in tqdm(test_dl, desc="Evaluation"):
            logits_val = model(batch[0])
            loss_val = loss_f(logits_val, batch[1])
            losses.append(loss_val.item())

            labels_all += batch[1].detach().cpu().numpy().tolist()
            logits_all += logits_val.detach().cpu().numpy().tolist()

        prediction = np.argmax(np.array(logits_all), axis=-1)

        if measure == 'acc':
            p, r = None, None
            f1 = accuracy_score(labels_all, prediction)
        else:
            p, r, f1, _ = precision_recall_fscore_support(labels_all,
                                                          prediction,
                                                          average='macro')

        print(confusion_matrix(labels_all, prediction))

    return p, r, f1, np.mean(losses), labels_all, prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--init_only", help="Whether to train the model",
                        action='store_true', default=False)

    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="number of labels", type=int,
                        default=3)

    parser.add_argument("--dataset_dir", help="Path to the train datasets",
                        default='data/e-SNLI/dataset/', type=str)
    parser.add_argument("--model_path",
                        help="Path where the model will be serialized",
                        type=str, nargs='+')

    parser.add_argument("--batch_size", help="Batch size", type=int,
                        default=128)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=0.001)
    parser.add_argument("--epochs", help="Epochs number", type=int, default=100)
    parser.add_argument("--mode", help="Mode for the script", type=str,
                        default='train', choices=['train', 'test'])
    parser.add_argument("--patience", help="Early stopping patience", type=int,
                        default=5)

    parser.add_argument("--model", help="Model for training", type=str,
                        default='lstm', choices=['lstm', 'cnn'])

    parser.add_argument("--embedding_dir",
                        help="Path to directory with pretrained embeddings",
                        default='./', type=str)
    parser.add_argument("--dropout",
                        help="Path to directory with pretrained embeddings",
                        default=0.1, type=float)
    parser.add_argument("--embedding_dim", help="Dimension of embeddings",
                        choices=[50, 100, 200, 300], default=100, type=int)

    # RNN ARGUMENTS
    parser.add_argument("--hidden_lstm",
                        help="Number of units in the hidden layer", default=300,
                        type=int)
    parser.add_argument("--num_layers", help="Number of rnn layers", default=4,
                        type=int)
    parser.add_argument("--hidden_sizes",
                        help="Number of units in the hidden layer",
                        default=[200, 50], type=int, nargs='+')

    # CNN ARGUMENTS
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=100)
    parser.add_argument("--kernel_heights", help="filter windows", type=int,
                        nargs='+', default=[2, 3, 4, 5])
    parser.add_argument("--stride", help="stride", type=int, default=1)
    parser.add_argument("--padding", help="padding", type=int, default=0)

    args = parser.parse_args()

    seed = random.randint(0, 10000)
    args.seed = seed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda") if args.gpu else torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    collate_fn = partial(collate_nli, tokenizer=tokenizer, device=device,
                         return_attention_masks=False, pad_to_max_length=False)
    sort_key = lambda x: len(x[0]) + len(x[1])

    if args.model == 'lstm':
        model = LSTM_MODEL(tokenizer, args, n_labels=args.labels).to(device)
    else:
        model = CNN_MODEL(tokenizer, args, n_labels=args.labels).to(device)

    if args.mode == 'test':
        test = NLIDataset(args.dataset_dir, type='test')
        test_dl = BucketBatchSampler(batch_size=args.batch_size,
                                     sort_key=sort_key, dataset=test,
                                     collate_fn=collate_fn)
        optimizer = AdamW(model.parameters(), lr=args.lr)

        scores = []
        for model_path in args.model_path:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model'])

            p, r, f1, loss, _, _ = eval_model(model, test_dl)
            scores.append((p, r, f1, loss))

        for i, name in zip(range(len(scores[0])), ['p', 'r', 'f1', 'loss']):
            l = [model_scores[i] for model_scores in scores]
            print(name, np.average(l), np.std(l))
    else:
        print("Loading datasets...")
        train = NLIDataset(args.dataset_dir, type='train')
        dev = NLIDataset(args.dataset_dir, type='dev')

        train_dl = BucketBatchSampler(batch_size=args.batch_size,
                                      sort_key=sort_key, dataset=train,
                                      collate_fn=collate_fn)
        dev_dl = BucketBatchSampler(batch_size=args.batch_size,
                                    sort_key=sort_key, dataset=dev,
                                    collate_fn=collate_fn)

        print(model)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, verbose=True)
        es = EarlyStopping(patience=args.patience, percentage=False, mode='max',
                           min_delta=0.0)

        if not args.init_only:
            best_model_w, best_perf = train_model(model, train_dl, dev_dl,
                                                  optimizer, scheduler,
                                                  args.epochs, es)
        else:
            best_model_w, best_perf = model.state_dict(), {'val_f1': 0}

        checkpoint = {
            'performance': best_perf,
            'args': vars(args),
            'model': best_model_w,
        }
        print(best_perf)
        print(args)

        torch.save(checkpoint, args.model_path[0])
