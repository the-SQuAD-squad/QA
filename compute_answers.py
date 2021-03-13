#compute_answers.py


import sys
import os
import random
import math
import numpy as np
import tensorflow as tf
import json
import pandas as pd
import re
import string
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

os.system("pip install -q transformers")
import transformers
from transformers import TFBertModel, TFRobertaModel, TFElectraModel, TFLongformerModel
from transformers import AutoTokenizer
pd.set_option('display.max_colwidth', -1)

def build_model(bert_hf_layer):
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    #pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])

    #HUGGINGFACE 🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗🤗
    sequence_output = bert_hf_layer(input_ids=input_word_ids, attention_mask=input_mask, 
                                    token_type_ids=input_type_ids).last_hidden_state

    #do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(sequence_output)
    start_logits = layers.Flatten(name="flatten_start")(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(sequence_output)
    end_logits = layers.Flatten(name="flatten_end")(end_logits)

    start_probs = layers.Activation(keras.activations.softmax, name="softmax_start")(start_logits)
    end_probs = layers.Activation(keras.activations.softmax, name="softmax_end")(end_logits)

    model = keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], 
                        outputs=[start_probs, end_probs],
                        name="BERT_QA")

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    optimizer = keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.summary(line_length=150)

    return model


try:
    path_to_json_file = sys.argv[1]
except:
    print("the path to the json file is needed as argument of this script")

try:
    huggingface_pretrained_model = sys.argv[2]
except:
    huggingface_pretrained_model = "roberta-base" 



#questi non servono?
##################################
# fix random seeds
seed_value = 42 
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

tf.compat.v1.set_random_seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

###################################

# BERT params
# hf model and input sequence max length 
hf_Models = {"bert-base-uncased": (TFBertModel, 512, "https://api.wandb.ai/files/buio/SQUAD/2a1u1bxu/model-best.h5"),
             "roberta-base" : (TFRobertaModel, 512, "https://api.wandb.ai/files/buio/SQUAD/184b7gum/model-best.h5"),
             "google/electra-base-discriminator" : (TFElectraModel, 512, "https://api.wandb.ai/files/buio/SQUAD/2rab6oli/model-best.h5"),
             "allenai/longformer-base-4096" : (TFLongformerModel, 1024, "not_yet_trained")}

TFHFModel, max_seq_length, weights_path = hf_Models[huggingface_pretrained_model]

# actual bert model
bert_hf_layer = TFHFModel.from_pretrained(huggingface_pretrained_model)

# actual tokenizer
tokenizer = AutoTokenizer.from_pretrained(huggingface_pretrained_model)

# load bert weights form the weights and biases platform

os.system(f"wget {weights_path}")
model = build_model(bert_hf_layer)
model.load_weights("model-best.h5")


# preprocess dev set
from tqdm.notebook import tqdm

with open(path_to_json_file, "r") as f:
    json_file = json.load(f)
data = json_file["data"]

rows = []
for document in data:
  for par in document['paragraphs']:
    for qas in par['qas']:         
      rows.append({
        'id' : qas['id'],
        'title': document["title"],
        'passage': par['context'],
        'question' : qas['question']
      })


df_dev = pd.DataFrame(rows)

def preprocess_bert(text):
    tokenized_text = tokenizer(list(text), return_offsets_mapping=True)

    rows_out  = [{'input_ids': tokenized_text.input_ids[i],
                  'offsets': tokenized_text.offset_mapping[i]} for i in range(len(text))]

    return rows_out

def labeling(df):
    skip = []
    input_word_ids = []
    input_type_ids = []
    input_mask = []
    context_token_to_char = []

    for id in tqdm(df.index):

        tokenized_context = df.loc[id]['passage']
        tokenized_question = df.loc[id]['question']

        # create inputs as usual
        input_ids = tokenized_context['input_ids'] + tokenized_question['input_ids'][1:] #removing CLS from the beginning of the question 
        token_type_ids = [0] * len(tokenized_context['input_ids']) + [1] * len(tokenized_question['input_ids'][1:])
        attention_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        
        # add padding if necessary
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:
            skip.append(id)
            continue
        input_word_ids.append(np.array(input_ids))
        input_type_ids.append(np.array(token_type_ids))
        input_mask.append(np.array(attention_mask))
        context_token_to_char.append(np.array(tokenized_context["offsets"]))

    df = df.drop(skip)
    df['input_word_ids'] = input_word_ids
    df['input_type_ids'] = input_type_ids
    df['input_mask'] = input_mask
    df['context_token_to_char'] = context_token_to_char
 
    return df, skip

# pre-process passage and question text
df_dev = df_dev.set_index('id')
df_bert_dev = df_dev.copy()

df_bert_dev['passage'] = preprocess_bert(df_dev['passage'])
df_bert_dev['question'] = preprocess_bert(df_dev['question'])

df_bert_dev, skipped = labeling(df_bert_dev)
df_bert_dev.head(1)

x_test = [np.stack(df_bert_dev["input_word_ids"]),
          np.stack(df_bert_dev["input_mask"]),
          np.stack(df_bert_dev["input_type_ids"])]

predictions = model.predict(x_test, verbose=1)

# save predictions to file for script evaluation
num_samples = len(predictions[0])

start, end = list(np.argmax(predictions, axis=-1).squeeze())
lines_c = 0
with open("predictions.txt","w") as out:
    out.write("{")
    for id in skipped:
        out.write(f'''"{id}": "error: sequence too long",\n''')

    for ans_idx in range(num_samples):
        # no answer
        if end[ans_idx] == 0:
            if ans_idx == num_samples-1:
                out.write(f'''"{df_bert_dev.index[ans_idx]}": ""''')
            else:
                out.write(f'''"{df_bert_dev.index[ans_idx]}": "",\n''')

        # extract answer text
        else:
            predicted_ans = tokenizer.decode(df_bert_dev.iloc[ans_idx]['passage']["input_ids"][start[ans_idx] : end[ans_idx]+1]).replace("\n"," ")
            if ans_idx == num_samples-1:
                out.write(f'''"{df_bert_dev.index[ans_idx]}": "{predicted_ans.replace('"',"")}"''')
            else:
                out.write(f'''"{df_bert_dev.index[ans_idx]}": "{predicted_ans.replace('"',"")}",\n''')

    out.write("}")