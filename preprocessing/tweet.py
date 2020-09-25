"""Preprocessing for the TSE dataset.
It contains only a train dataset and we split it a test, dev and train splits."""
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/tweet_sent/train.csv', sep=',')

train, test = train_test_split(df, test_size=0.2, random_state=0)
val, test = train_test_split(test, test_size=0.5, random_state=0)

train.to_csv('data/tweet_sent/train.tsv', sep='\t', header=None)
val.to_csv('data/tweet_sent/dev.tsv', sep='\t', header=None)
test.to_csv('data/tweet_sent/test.tsv', sep='\t', header=None)
