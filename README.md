# Quesiton Answering on SQuAD v1.1

This folder contains:
- `compute_answer.py`: given the question file, this script will download the best model and save the predictions
- `QA.ipynb`: the notebook used for train and evaluate the best model
## Experiments:
- rnn: https://wandb.ai/veri/SQUAD
- transformers: https://wandb.ai/buio/SQUAD
## Branches
- `main`: merged from the `huggingface` branch
- `rnn`: baseline model based on RNN
- `rnn-regression`: experiment with RNN + regression heads
- `huggingface`: transformer-base models, comparation between BERT, ELECTRA, RoBERTa, Longformer
- `huggingface-regression`: experiment with RoBERTa + regression head
