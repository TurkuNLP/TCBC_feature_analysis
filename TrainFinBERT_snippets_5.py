#Imports
from datasets import Dataset, DatasetDict
from pprint import PrettyPrinter
import logging
import transformers
import torch
import os
import evaluate
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from scripts import corpusMLfunctions as cmf
import json
import optuna
import shutil

#Some constants and other housekeeping before diving into coding
MODEL_NAME = "TurkuNLP/bert-base-finnish-cased-v1"
KEYLISTS = "Keylists.jsonl"
BASE_BEG = "SnippetDatasets/"
BASE_MID = "sniplen_"
BASE_END = ".jsonl"
SNIPPET_LENS = ['5','10','25','50','75','100']
pprint = PrettyPrinter(compact=True).pprint
os.environ['WANDB_MODE'] = 'disabled'

#Do actual masking of PROPN for FinBERT

def maskPropnWithMask(example):
    df = cmf.snippetConllu2DF(example['conllu'])
    df.loc[df['upos'] == 'PROPN', 'lemma'] = "[MASK]"
    df.loc[df['upos'] == 'PROPN', 'text'] = "[MASK]"
    example['masked_text'] = ' '.join(df['text'].to_numpy('str'))
    #if example['label'] == '7-8':
    #    example['label'] = 0
    #elif example['label'] == '9-12':
    #    example['label'] = 1
    #else:
    #    example['label'] = 2
    return example

def reMapLabels(ex):
    age = ex['age']
    if age < 13:
        ex['label'] = 0
    else:
        ex['label'] = 1
    return ex

keylists = []
with open(KEYLISTS, 'r') as f:
    for line in f:
        keylists.append(json.loads(line))

#Load a keylist dataset with conllu-data, then mask PROPN, and finally map to raw text
#Utilize a cache directory and custom cache files to manage HF datasets in CSC environments
#Also makes it easier to clean cache files and use space more efficiently
cache_dir = "cache_dir/"
cache_file_train = cache_dir+str(0)+"_"+str(SNIPPET_LENS[0])+"_train.jsonl"
cache_file_eval = cache_dir+str(0)+"_"+str(SNIPPET_LENS[0])+"_eval.jsonl"
cache_file_test = cache_dir+str(0)+"_"+str(SNIPPET_LENS[0])+"_test.jsonl"
hf_cache_dir = cache_dir+str(0)+"_"+str(SNIPPET_LENS[0])+"_ds"
#Load the keyset (aka author and genre splits into train/test/eval)
train_keys = keylists[0]['train_keys']
eval_keys = keylists[0]['eval_keys']
test_keys = keylists[0]['test_keys']
#Generate Dataset objects from the snippet files
train_dss = cmf.combineSnippedBooksToDS(train_keys, SNIPPET_LENS[0], hf_cache_dir, cache_file_train, inc_conllu=True, folder=BASE_BEG)
eval_dss = cmf.combineSnippedBooksToDS(eval_keys, SNIPPET_LENS[0], hf_cache_dir, cache_file_eval, inc_conllu=True, folder=BASE_BEG)
test_dss = cmf.combineSnippedBooksToDS(test_keys, SNIPPET_LENS[0], hf_cache_dir, cache_file_test, inc_conllu=True, folder=BASE_BEG)

#Mask PROPN with [mask] so we don't overfit on character names because e.g. there are three Artemis Fowl books in the training data
train_dss = train_dss.map(maskPropnWithMask)
eval_dss = eval_dss.map(maskPropnWithMask)
test_dss = test_dss.map(maskPropnWithMask)
train_dss = train_dss.map(reMapLabels)
eval_dss = eval_dss.map(reMapLabels)
test_dss = test_dss.map(reMapLabels)

#Combine into a DatasetDict for cleaner code
dataset = DatasetDict({'train':train_dss, 'test':test_dss, 'eval':eval_dss})

#Load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

#Encode label
train_dss = train_dss.class_encode_column('label')
eval_dss = eval_dss.class_encode_column('label')
test_dss = test_dss.class_encode_column('label')

#Tokenizer. We truncate from start till 510 tokens (so that there is room for the special tokens)
#We pad with [pad] so that every example has 512 input ids
def tokenize(ex):
    return tokenizer(
        ex['masked_text'],
        max_length=512,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
    )
#Tokenize datasets
dataset = dataset.map(tokenize, num_proc=len(os.sched_getaffinity(0)), batched=True).select_columns(['input_ids', 'attention_mask', 'label'])

#Evaluate with f1-score using macro averaging
def evaluate_score(pred):
  y_pred = pred.predictions.argmax(axis=-1)
  y_true = pred.label_ids
  return {'accuracy' : f1_score(y_true, y_pred, average='macro')}

#Something that the HF-optuna combo requires
def compute_objective(metrics):
    return metrics["eval_accuracy"]

#Data collator so that everything works as intended :)
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

#Function that the trainer calls to initialize a fresh model (so that we don't train the same one :)
model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Set training arguments
trainer_args = transformers.TrainingArguments(
    "checkpoints",
    eval_strategy="steps",
    logging_strategy="steps",
    save_total_limit=2,
    save_steps=5000,
    eval_steps=5000,
    logging_steps=5000,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    remove_unused_columns=True,
    num_train_epochs=2
)

#Trainer arguments
trainer = transformers.Trainer(
    model=model,
    args=trainer_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['eval'],
    processing_class = tokenizer,
    data_collator = data_collator,
    compute_metrics=evaluate_score,
    callbacks=[early_stopping, training_logs]
)

trainer.train()

trainer.save_model("FinBERT_for_book_snippets_5_two_groups")

eval_results = trainer.evaluate(dataset["test"])

pprint(eval_results)

print('Accuracy:', eval_results['eval_accuracy'])

#Clean up cache files as we don't need them anymore/we want to save space
os.remove(cache_file_train)
os.remove(cache_file_test)
os.remove(cache_file_eval)
shutil.rmtree(hf_cache_dir)

#Some plotting

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