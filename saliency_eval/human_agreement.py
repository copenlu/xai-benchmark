"""Computing Human Agreement measure."""
import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score
from transformers import BertTokenizer

from models.data_loader import _twitter_label, get_dataset
from models.saliency_utils import get_gold_saliency_esnli, \
    get_gold_saliency_imdb, get_gold_saliency_tweet

saliency_func_map = {
    'snli': get_gold_saliency_esnli, 'imdb': get_gold_saliency_imdb,
    'tweet': get_gold_saliency_tweet
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", help="What mode to run the evaluation on",
                        default='all', type=str, choices=['all', 'c', 'w'])
    parser.add_argument("--dataset", help="Which dataset", default='snli',
                        type=str, choices=['snli', 'imdb', 'tweet'])
    parser.add_argument("--dataset_dir",
                        help="Path to the direcory with the datasets",
                        default='data/e-SNLI/dataset/',
                        type=str)
    parser.add_argument("--saliency_path",
                        help="Path to the file with saliencies", type=str,
                        nargs='+')
    parser.add_argument("--saliencies",
                        help="The saliencies to compute the MAP for", type=str,
                        nargs='+')

    args = parser.parse_args()
    print(args, flush=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    test = get_dataset(path=args.dataset_dir, mode='test', dataset=args.dataset)
    saliency_gold_f = saliency_func_map[args.dataset]

    for saliency_name in args.saliencies:
        avg_seeds = []
        for saliency_path in args.saliency_path:
            avgp = []

            prediction_path = saliency_path.replace('saliency',
                                                    'models') + '.predictions'
            predictions = json.load(open(prediction_path))['class']
            saliency_path = saliency_path + '_' + saliency_name
            with open(saliency_path) as out:
                for i, line in enumerate(out):
                    try:
                        instance_saliency = json.loads(line)
                    except:
                        line = next(out)
                        instance_saliency = json.loads(line)
                    saliency = instance_saliency['tokens']

                    instance = test[i]
                    if args.dataset == 'snli':
                        instance_gold = instance[2]
                    elif args.dataset == 'imdb':
                        instance_gold = instance[1]
                    elif args.dataset == 'tweet':
                        instance_gold = _twitter_label[instance[1]]
                    predicted = predictions[i]

                    if args.subset == 'c' and predicted != instance_gold:
                        continue
                    elif args.subset == 'w' and predicted == instance_gold:
                        continue

                    if args.dataset == 'snli':
                        token_ids = tokenizer.encode(instance[0], instance[1])
                    else:
                        token_ids = tokenizer.encode(instance[0])

                    token_pred_saliency = []
                    for record in saliency:
                        token_pred_saliency.append(record[str(instance_gold)])

                    gold_saliency = saliency_gold_f(instance,
                                                    tokenizer.convert_ids_to_tokens(
                                                        token_ids),
                                                    [tokenizer.cls_token,
                                                     tokenizer.sep_token,
                                                     tokenizer.pad_token],
                                                    tokenizer)
                    # remove pad token saliencies
                    if args.dataset == 'imdb':
                        if len(gold_saliency) > len(token_pred_saliency):
                            token_pred_saliency += [0.0] * (
                                        len(gold_saliency) - len(
                                    token_pred_saliency))
                    else:
                        gold_saliency = gold_saliency[:509]
                    token_pred_saliency = token_pred_saliency[
                                          :len(gold_saliency)]

                    avgp.append(average_precision_score(gold_saliency,
                                                        token_pred_saliency))

                print(len(avgp), np.mean(avgp), flush=True)
                avg_seeds.append(np.mean(avgp))
        print(saliency_name, flush=True)
        print(f'{np.mean(avg_seeds):.3f} ($\pm${np.std(avg_seeds):.3f})',
              flush=True)
