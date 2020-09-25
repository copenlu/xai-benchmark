"""Generates random saliency scores for a baseline."""
import argparse
import json
import random

import numpy as np
import torch
from pypapi import events, papi_high as high

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saliency_paths", help="Path to original saliency",
                        type=str, nargs='+')
    parser.add_argument("--output_paths", help="Path to save random saliency",
                        type=str, nargs='+')
    parser.add_argument("--seeds", help="Random seed", type=int, nargs='+',
                        default=[1, 2, 3, 4, 5])
    parser.add_argument("--labels", help="Number of labels", type=int,
                        default=3)
    args = parser.parse_args()
    classes = list(range(args.labels))

    flops = []
    for saliency_path, output_path, seed in zip(args.saliency_paths,
                                                args.output_paths, args.seeds):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)

        with open(saliency_path) as out:
            with open(output_path, 'w') as output_sal:
                saliency_flops = []
                for j, line in enumerate(out):
                    high.start_counters([events.PAPI_FP_OPS, ])

                    try:
                        instance_saliency = json.loads(line)
                    except:
                        line = next(out)
                        instance_saliency = json.loads(line)

                    for i, token in enumerate(instance_saliency['tokens']):
                        if token['token'] == '[PAD]':
                            continue
                        for _c in classes:
                            instance_saliency['tokens'][i][
                                str(_c)] = np.random.rand()

                    output_sal.write(json.dumps(instance_saliency) + '\n')

                    x = sum(high.stop_counters())
                    saliency_flops.append(x)

        print(np.mean(saliency_flops), np.std(saliency_flops))
        flops.append(np.mean(saliency_flops))

    print('FLOPs', f'{np.mean(flops):.2f} ($\pm${np.std(flops):.2f})')
