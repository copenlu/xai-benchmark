"""Utilities extracting the annotated salient words from the datasets"""
from string import punctuation


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_gold_saliency_imdb(instance, tokens, special_tokens, tokenizer):
    gold_tokens = tokenizer.tokenize(instance[3])
    in_gold_token = 0
    saliency_gold = []
    gold_tokens = [t.replace('##', '') for t in gold_tokens]
    tokens = [t.replace('##', '') for t in tokens]
    for token in tokens:
        # '<', 'ne', '##g', '>'  '<', 'po', '##s', '>'
        if ''.join(gold_tokens[0:4]) in ['<neg>',
                                         '<pos>']:
            in_gold_token = 1
            for _ in range(4):
                gold_tokens.pop(0)
        # '<', '/', 'ne', '##g', '>'  '<', '/', 'po', '##s', '>'
        elif ''.join(gold_tokens[0:5]) in ['</neg>',
                                           '</pos>']:
            in_gold_token = 0
            for _ in range(5):
                gold_tokens.pop(0)

        if token in special_tokens:
            saliency_gold.append(0)
        elif token == gold_tokens[0]:
            saliency_gold.append(in_gold_token)
            gold_tokens.pop(0)
        else:
            print('OOOPs', token, flush=True)
            saliency_gold.append(in_gold_token)
    return saliency_gold


def get_gold_saliency_esnli(instance, tokens, special_tokens, tokenizer=None):
    gold_tokens = instance[3].lower().split(' ') + instance[4].lower().split(
        ' ')
    gold_tokens = [t for t in gold_tokens if len(t) > 0]
    in_gold_token = 0
    saliency_gold = []

    for token in tokens:
        token = token.replace('#', '')
        if token in special_tokens:
            saliency_gold.append(0)
            continue
        if token == gold_tokens[0]:
            saliency_gold.append(in_gold_token)
            gold_tokens.pop(0)
            continue

        if all(_t in punctuation for _t in gold_tokens):
            gold_tokens.pop(0)

        if gold_tokens[0].startswith('*') and len(gold_tokens[0]) == 1:
            in_gold_token = 0
            gold_tokens.pop(0)

        if gold_tokens[0].startswith('*') and len(gold_tokens[0]) > 1:
            in_gold_token = 1
            gold_tokens[0] = gold_tokens[0][1:]

        if gold_tokens[0].startswith(token):
            saliency_gold.append(in_gold_token)
            gold_tokens[0] = gold_tokens[0][len(token):]
            if gold_tokens[0] == '*':
                gold_tokens.pop(0)
                in_gold_token = 0

        else:
            print('OOOPs', token)
            saliency_gold.append(0)

    return saliency_gold


def get_gold_saliency_tweet(instance, tokens, special_tokens, tokenizer=None):
    gold_tokens = [_t.replace('##', '') for _t in
                   tokenizer.tokenize(instance[-1])]
    gold_text = ''.join(gold_tokens)
    saliency_gold = [0 for _ in tokens]
    tokens = [_t.replace('##', '') for _t in tokens]

    for i, token in enumerate(tokens):
        if gold_text in ''.join(tokens[i:i + len(gold_tokens)]):
            for j in range(i, i + len(gold_tokens)):
                saliency_gold[j] = 1
            break
    if sum(saliency_gold) == 0:
        print(gold_tokens, gold_text, tokens)
    return saliency_gold
