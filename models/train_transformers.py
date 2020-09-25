"""Script for training a Transformer model for the e-SNLI dataset."""
import argparse
import random
from functools import partial
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, \
    precision_recall_fscore_support
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
from transformers import AdamW, BertConfig, BertForSequenceClassification, \
    BertTokenizer, get_constant_schedule_with_warmup

from models.data_loader import BucketBatchSampler, NLIDataset, collate_nli
from models.model_builder import EarlyStopping


def train_model(model: torch.nn.Module,
                train_dl: BatchSampler, dev_dl: BatchSampler,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LambdaLR,
                n_epochs: int,
                labels: int = 3,
                early_stopping: EarlyStopping = None) -> (Dict, Dict):
    best_val, best_model_weights = {'val_f1': 0}, None

    for ep in range(n_epochs):
        for batch in tqdm(train_dl, desc='Training'):
            model.train()
            optimizer.zero_grad()
            loss, _ = model(batch[0], attention_mask=batch[1],
                                 labels=batch[2].long())[:2]

            loss.backward()
            optimizer.step()
            scheduler.step()

        val_p, val_r, val_f1, val_loss, _, _ = eval_model(model, dev_dl, labels)
        current_val = {
            'val_f1': val_f1, 'val_p': val_p, 'val_r': val_r,
            'val_loss': val_loss, 'ep': ep
        }
        print(current_val, flush=True)

        if current_val['val_f1'] > best_val['val_f1']:
            best_val = current_val
            best_model_weights = model.state_dict()

        if early_stopping and early_stopping.step(val_f1):
            print('Early stopping...')
            break

    return best_model_weights, best_val


def eval_model(model: torch.nn.Module, test_dl: BatchSampler, labels,
               measure=None):
    model.eval()

    with torch.no_grad():
        labels_all = []
        logits_all = []
        losses = []
        for batch in tqdm(test_dl, desc="Evaluation"):
            loss, logits_val = model(batch[0], attention_mask=batch[1],
                                     labels=batch[2].long())[:2]
            losses.append(loss.item())

            labels_all += batch[2].detach().cpu().numpy().tolist()
            logits_all += logits_val.detach().cpu().numpy().tolist()

        prediction = np.argmax(np.asarray(logits_all).reshape(-1, labels),
                               axis=-1)

        if measure == 'acc':
            p, r = None, None
            f1 = accuracy_score(labels_all, prediction)
        else:
            p, r, f1, _ = precision_recall_fscore_support(labels_all,
                                                          prediction,
                                                          average='macro')
            print(confusion_matrix(labels_all, prediction), flush=True)

        return p, r, f1, np.mean(losses), labels_all, prediction.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="num of lables", type=int, default=3)

    parser.add_argument("--dataset_dir", help="Path to the train datasets",
                        default='data/e-SNLI/dataset/', type=str)
    parser.add_argument("--model_path",
                        help="Path where the model will be serialized",
                        default='nli_bert', nargs='+', type=str)

    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=5e-5)
    parser.add_argument("--epochs", help="Epochs number", type=int, default=4)
    parser.add_argument("--mode", help="Mode for the script", type=str,
                        default='train', choices=['train', 'test'])
    parser.add_argument("--init_only", help="Whether to train the model",
                        action='store_true', default=False)

    args = parser.parse_args()

    seed = random.randint(0, 10000)
    args.seed = seed

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    collate_fn = partial(collate_nli, tokenizer=tokenizer, device=device,
                         return_attention_masks=True, pad_to_max_length=False)

    transformer_config = BertConfig.from_pretrained('bert-base-uncased',
                                                    num_labels=args.labels)

    print(args, flush=True)
    sort_key = lambda x: len(x[0]) + len(x[1])
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          config=transformer_config).to(
        device)
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

    if args.mode == 'test':
        test = NLIDataset(args.dataset_dir, type='test')
        # print(Counter([_x['label'] for _x in test]).most_common(3))

        test_dl = BucketBatchSampler(batch_size=args.batch_size,
                                     sort_key=sort_key, dataset=test,
                                     collate_fn=collate_fn)

        scores = []
        for model_path in args.model_path:
            checkpoint = torch.load(model_path)

            model.load_state_dict(checkpoint['model'])
            p, r, f1, loss, _, _ = eval_model(model, test_dl, args.labels)
            scores.append((p, r, f1, loss))

        for i, name in zip(range(len(scores[0])), ['p', 'r', 'f1', 'loss']):
            l = [model_scores[i] for model_scores in scores]
            print(name, np.average(l), np.std(l))

    else:
        print("Loading datasets...")
        train = NLIDataset(args.dataset_dir, type='train')
        dev = NLIDataset(args.dataset_dir, type='dev')

        # print(Counter([_x['label'] for _x in train]).most_common(3))
        # print(Counter([_x['label'] for _x in dev]).most_common(3))

        train_dl = BucketBatchSampler(batch_size=args.batch_size,
                                      sort_key=sort_key, dataset=train,
                                      collate_fn=collate_fn)
        dev_dl = BucketBatchSampler(batch_size=args.batch_size,
                                    sort_key=sort_key, dataset=dev,
                                    collate_fn=collate_fn)

        num_train_optimization_steps = int(
            args.epochs * len(train) / args.batch_size)

        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=0.05)
        if args.init_only:
            best_model_w, best_perf = model.state_dict(), {'val_f1': 0}
        else:
            best_model_w, best_perf = train_model(model, train_dl, dev_dl,
                                                  optimizer, scheduler,
                                                  args.epochs)

        checkpoint = {
            'performance': best_perf,
            'args': vars(args),
            'model': best_model_w,
        }
        print(best_perf)
        print(args)

        torch.save(checkpoint, args.model_path)
