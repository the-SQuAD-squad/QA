# Quesiton Answering on SQuAD v1.1

This folder contains:
- `compute_answer.py`: given the question file, this script will download the best model and save the predictions
- `QA.ipynb`: the notebook used for train and evaluate the best model
## Experiment plot:
- rnn: https://wandb.ai/veri/SQUAD/reports/RNN--Vmlldzo1Mzk2NTU
- transformers: https://wandb.ai/buio/SQUAD/reports/Transformers-Report--Vmlldzo1Mzk3MjE
## Branches
- `main`: merged from the `huggingface` branch
- `rnn`: baseline model based on RNN
- `rnn-regression`: experiment with RNN + regression heads
- `huggingface`: transformer-base models, comparation between BERT, ELECTRA, RoBERTa, Longformer
- `huggingface-regression`: experiment with RoBERTa + regression head
