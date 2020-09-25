# A Diagnostic Study of Explainability Techniques for Text Classification

This is repository for the paper 
[A Diagnostic Study of Explainability Techniques for Text Classification]() 
accepted at EMNLP 2020.

<p align="center">
  <img src="sal_example.png" width="450" alt="Adversarial Architecture">
</p>


In this paper, we develop a comprehensive list of diagnostic properties
for evaluating existing explainability techniques. We then employ the proposed 
list to compare a set of diverse explainability techniques on downstream text 
classification tasks and neural network architectures. We also compare the 
saliency scores assigned by the explainability techniques with human 
annotations of salient input regions to find relations between a model's 
performance and the agreement of its rationales with human ones. Overall, we 
find that the gradient-based explanations perform best across tasks and model 
architectures, and we present further insights into the properties of the 
reviewed explainability techniques.

## Code Base

### preprocessing
The SNLI dataset is used as is. For the IMDB and the TSE datasets, 
we have to make additional splits with the scripts in the package. 

### models
Contains code to training Transformer, LSTM, and CNN models for the three datasets.
We train five versions of each models with different seeds and we show an example with one.

```
# e-SNLI
# Transformer, all seeds: 684, 3615, 4275, 9194, 5301; 1340, 6636, 7006,5017, 6612 (init only)
python models/train_transformers.py --lr 2e-5 --epochs 4 --model_path data/models/snli/transformer/transformer_snli_2e5_1 --gpu
python models/train_transformers.py --lr 2e-5 --epochs 4 --model_path data/models/snli/random_transformer/transformer_snli_2e5_1 --gpu --init_only

# CNN, all seeds: 9874, 5832, 4429, 4773, 5874; 2550, 6168, 1601, 1197, 6385 (init only)
python models/train_lstm_cnn.py --embedding_dim 300 --model cnn --model_path data/models/snli/cnn/cnn_1 --gpu --batch_size 256 --out_channels 300 --dropout 0.05 --kernel_heights 4 5 6 7 --lr 0.0001
python models/train_lstm_cnn.py --embedding_dim 300 --model cnn --model_path data/models/snli/random_cnn/cnn_1 --gpu --batch_size 256 --out_channels 300 --dropout 0.05 --kernel_heights 4 5 6 7 --lr 0.0001 --init_only

# LSTM, all seeds: 8493, 156, 9357, 1979, 7877; 3902, 3266, 9957, 6958, 4352 (init only)
python models/train_lstm_cnn.py --gpu --model_path data/models/snli/lstm/lstm_1 --epoch 100 --model lstm --embedding_dim 100 --batch_size 256 --lr 0.01 --dropout 0.1 --hidden_lstm 100 --num_layers 4 --hidden_sizes 100 50
python models/train_lstm_cnn.py --gpu --model_path data/models/snli/random_lstm/lstm_1 --epoch 100 --model lstm --embedding_dim 100 --batch_size 256 --lr 0.01 --dropout 0.1 --hidden_lstm 100 --num_layers 4 --hidden_sizes 100 50 --init_only


# IMDB
# Transformer, all seeds: 6227, 9141, 655, 9218, 4214; 5362, 400, 5799, 3113, 1858 (init only)
python models/train_imdb.py --gpu --labels 2 --dataset_dir data/imdb_rats --model_path data/models/imdb/transformer/trans_1 --batch_size 8 --lr 2e-5 --epochs 3 --patience 3 --random_seed
python models/train_imdb.py --gpu --labels 2 --dataset_dir data/imdb_rats --model_path data/models/imdb/random_transformer/trans_1 --batch_size 8 --lr 2e-5 --epochs 20 --patience 3 --random_seed --init_only

# CNN, all seeds: 1552, 4676, 5789, 2116, 6865; 4923, 2426, 8602, 4273, 9933 (init only)
python models/train_imdb.py --dataset_dir data/imdb_rats --embedding_dim 300 --model cnn --model_path data/models/imdb/cnn/cnn_1 --gpu --batch_size 64 --out_channels 50 --dropout 0.05 --kernel_heights 2 3 4 --lr 0.001 --labels 2 --random_seed
python models/train_imdb.py --dataset_dir data/imdb_rats --embedding_dim 300 --model cnn --model_path data/models/imdb/random_cnn/cnn_1 --gpu --batch_size 64 --out_channels 50 --dropout 0.05 --kernel_heights 2 3 4 --lr 0.001 --labels 2 --random_seed --init_only

# LSTM, all seeds: 4237, 630, 7208, 8013, 8505; 4907, 119, 1859, 9937, 2029 (init only)
python models/train_imdb.py --dataset_dir data/imdb_rats --gpu --model_path data/models/imdb/rnn/rnn_1 --epoch 100 --model lstm --embedding_dim 100 --batch_size 16 --lr 0.001 --dropout 0.1 --hidden_lstm 100 --num_layers 1 --hidden_sizes 50 25 --labels 2 --random_seed
python models/train_imdb.py --dataset_dir data/imdb_rats --gpu --model_path data/models/imdb/random_rnn/rnn_1 --epoch 100 --model lstm --embedding_dim 100 --batch_size 16 --lr 0.001 --dropout 0.1 --hidden_lstm 100 --num_layers 1 --hidden_sizes 50 25 --labels 2 --random_seed --init_only


# TSE
# Transformer, all seeds: 9218, 655, 2406, 2337, 8598; 7895, 9312, 863, 6469, 8084 (init only)
python models/train_twitter.py --gpu --labels 3 --dataset_dir data/tweet_sent --model_path data/models/tweet/transformer/trans_1 --batch_size 8 --lr 3e-5 --epochs 5 --patience 3 --random_seed
python models/train_twitter.py --gpu --labels 3 --dataset_dir data/tweet_sent --model_path data/models/tweet/random_transformer/trans_1 --batch_size 8 --lr 3e-5 --epochs 20 --patience 3 --random_seed --init_only

# CNN, all seeds: 6240, 5457, 4192, 3354, 279; 4340, 967, 1602, 1050, 6502 (init only)
python models/train_twitter.py --dataset_dir data/tweet_sent --embedding_dim 300 --model cnn --model_path data/models/tweet/cnn/cnn_1 --gpu --batch_size 64 --out_channels 50 --dropout 0.05 --kernel_heights 3 4 5 --lr 0.001 --labels 3 --random_seed
python models/train_twitter.py --dataset_dir data/tweet_sent --embedding_dim 300 --model cnn --model_path data/models/tweet/random_cnn/cnn_1 --gpu --batch_size 64 --out_channels 50 --dropout 0.05 --kernel_heights 3 4 5 --lr 0.001 --labels 3 --random_seed --init_only

# LSTM, all seeds: 2679, 1315, 5117, 5685, 1269; 9536, 5258, 9693, 6227, 9141 (init only)
python models/train_twitter.py --dataset_dir data/tweet_sent --gpu --model_path data/models/tweet/lstm/lstm_1 --epoch 100 --model lstm --embedding_dim 200 --batch_size 16 --lr 0.001 --dropout 0.05 --hidden_lstm 100 --num_layers 1 --hidden_sizes 200 100 --labels 3 --random_seed
python models/train_twitter.py --dataset_dir data/tweet_sent --gpu --model_path data/models/tweet/random_lstm/lstm_1 --epoch 100 --model lstm --embedding_dim 200 --batch_size 16 --lr 0.001 --dropout 0.05 --hidden_lstm 100 --num_layers 1 --hidden_sizes 200 100 --labels 3 --random_seed --init_only
```

### Generating Explanations

### Evaluating explanations
