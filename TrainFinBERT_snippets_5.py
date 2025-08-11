#Imports
from datasets import Dataset, DatasetDict
from pprint import PrettyPrinter
import logging
import transformers
import torch
import os
import evaluate
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
from scripts import corpusMLfunctions as cmf
import json

#Some constants and other housekeeping before diving into coding
MODEL_NAME = "TurkuNLP/bert-base-finnish-cased-v1"
KEYLISTS = "Keylists.jsonl"
BASE_BEG = "SnippetDatasets/"
BASE_MID = "sniplen_"
BASE_END = ".jsonl"
SNIPPET_LENS = ['5','10','25','50','75','100']
logging.disable(logging.INFO)
pprint = PrettyPrinter(compact=True).pprint
os.environ['WANDB_MODE'] = 'disabled'

#Do actual masking of PROPN for FinBERT

def maskPropnWithMask(example):
    df = cmf.snippetConllu2DF(example['conllu'])
    df.loc[df['upos'] == 'PROPN', 'lemma'] = "[mask]"
    df.loc[df['upos'] == 'PROPN', 'text'] = "[mask]"
    example['masked_text'] = ' '.join(df['text'].to_numpy('str'))
    return example

keylists = []
with open(KEYLISTS, 'r') as f:
    for line in f:
        keylists.append(json.loads(line))

#Load a keylist dataset with conllu-data, then mask PROPN, and finally map to raw text
cache_dir = "cache_dir/"
cache_file_train = cache_dir+str(0)+"_"+str(SNIPPET_LENS[0])+"_train.jsonl"
cache_file_eval = cache_dir+str(0)+"_"+str(SNIPPET_LENS[0])+"_eval.jsonl"
cache_file_test = cache_dir+str(0)+"_"+str(SNIPPET_LENS[0])+"_test.jsonl"
hf_cache_dir = cache_dir+str(0)+"_"+str(SNIPPET_LENS[0])+"_ds"
train_keys = keylists[0]['train_keys']
#Temporary edit to test with combining eval+test as we are not param optimizing
eval_keys = keylists[0]['eval_keys']
test_keys = keylists[0]['test_keys']
train_dss = cmf.combineSnippedBooksToDS(train_keys, SNIPPET_LENS[0], hf_cache_dir, cache_file_train, inc_conllu=True, folder=BASE_BEG)
eval_dss = cmf.combineSnippedBooksToDS(eval_keys, SNIPPET_LENS[0], hf_cache_dir, cache_file_eval, inc_conllu=True, folder=BASE_BEG)
test_dss = cmf.combineSnippedBooksToDS(test_keys, SNIPPET_LENS[0], hf_cache_dir, cache_file_test, inc_conllu=True, folder=BASE_BEG)

train_dss = train_dss.map(maskPropnWithMask)
eval_dss = eval_dss.map(maskPropnWithMask)
test_dss = test_dss.map(maskPropnWithMask)

dataset = DatasetDict({'train':train_dss, 'test':test_dss, 'eval':eval_dss})
dataset = dataset.class_encode_column('label')

#Load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(ex):
    return tokenizer(
        ex['masked_text'],
        return_overflowing_tokens=False,
        max_length=512,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        stride=20
    )

tokenized_ds = dataset.map(tokenize, num_proc=4, batched=True)

model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Set training arguments
trainer_args = transformers.TrainingArguments(
    "checkpoints",
    evaluation_strategy="steps",
    logging_strategy="steps",
    load_best_model_at_end=True,
    eval_steps=100,
    logging_steps=100,
    learning_rate=0.00001,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    max_steps=500,
)

def evaluate_score(pred):
  y_pred = pred.predictions.argmax(axis=-1)
  y_true = pred.label_ids
  #return { 'accuracy': sum(y_pred == y_true) / len(y_true) }
  #The output is the same as in the name of the function and the second index holds the values for positive predicitons, which we're interested in
  return {'accuracy' : precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2])[2][1]}

data_collator = transformers.DataCollatorWithPadding(tokenizer)


# Argument gives the number of steps of patience before early stopping
early_stopping = transformers.EarlyStoppingCallback(
    early_stopping_patience=5
)

from collections import defaultdict

class LogSavingCallback(transformers.TrainerCallback):
    def on_train_begin(self, *args, **kwargs):
        self.logs = defaultdict(list)
        self.training = True

    def on_train_end(self, *args, **kwargs):
        self.training = False

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if self.training:
            for k, v in logs.items():
                if k != "epoch" or v not in self.logs[k]:
                    self.logs[k].append(v)

training_logs = LogSavingCallback()

trainer = transformers.Trainer(
    model=model,
    args=trainer_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["eval"],
    compute_metrics=evaluate_score,
    callbacks=[early_stopping, training_logs]
)

trainer.train()

trainer.save_model('TCBC_v1-0_sniplen5_finetuned_FinBERT')

eval_results = trainer.evaluate(dataset["test"])

pprint(eval_results)

print('Accuracy:', eval_results['eval_accuracy'])

import matplotlib.pyplot as plt

def plot(logs, keys, labels):
    values = sum([logs[k] for k in keys], [])
    plt.ylim(max(min(values)-0.1, 0.0), min(max(values)+0.1, 1.0))
    for key, label in zip(keys, labels):
        plt.plot(logs["epoch"], logs[key], label=label)
    plt.legend()
    plt.show()

plot(training_logs.logs, ["loss", "eval_loss"], ["Training loss", "Evaluation loss"])

plot(training_logs.logs, ["eval_accuracy"], ["Evaluation accuracy"])
