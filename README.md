# Quesiton Answering on SQuAD v1.1
This repository contains code for the Question Answering part of our project work for NLP2020 class by Paolo Torroni @unibo.  
It implements 2 main architectures: a BiLSTM fully trained model and one based on BERT fine-tuning.

<img width="817" alt="Schermata 2022-04-21 alle 16 53 33" src="https://user-images.githubusercontent.com/38630200/164485737-aaba5ab0-1b9b-4e1b-a62e-ec96c7228308.png">
<img width="817" alt="Schermata 2022-04-21 alle 16 53 14" src="https://user-images.githubusercontent.com/38630200/164485759-bfa1ccae-6317-41b9-8583-7f26995c3853.png">



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
