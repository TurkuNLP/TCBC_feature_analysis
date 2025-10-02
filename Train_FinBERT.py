#Imports
from datasets import Dataset, DatasetDict
from pprint import PrettyPrinter, pprint
import transformers
import os
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from scripts import corpusMLfunctions as cmf
from TCBC_tools import Structure, MachineLearning as ml, FeatureExtraction as fe
import json
import sys

#Some constants and other housekeeping before diving into coding
MODEL_NAME = "TurkuNLP/bert-base-finnish-cased-v1"
pprint = PrettyPrinter(compact=True).pprint
os.environ['WANDB_MODE'] = 'disabled'

def trainFinBERT(SPLIT_ID):
    #Do actual masking of PROPN for FinBERT

    def conllu2RawText(conllu_text):
        conllu_lines = conllu_text.split("\n")
        return " ".join([x.split('\t')[1] for x in conllu_lines if len(x) > 0])

    def mapConlluData2RawText(ex):
        return {'data':[conllu2RawText(x) for x in ex['data']]}

    def assignLabel(ex):
        age = int(Structure.findAgeFromID(ex))
        if age < 9:
            return 1
        elif age < 13:
            return 2
        else:
            return 0
        

    def mapLabels(ex):
        return {'label':[assignLabel(x) for x in ex['book_id']]}

    keylists = ml.getKeylist("Keylists.jsonl")

    #Nab keys
    train_keys = keylists[SPLIT_ID]['train_keys']
    eval_keys = keylists[SPLIT_ID]['eval_keys']
    test_keys = keylists[SPLIT_ID]['test_keys']
    #Load base dataset. We use sniplen 5 so that we have as many example as possible, as well as to minimize the textual material taht gets clipped for being too long
    base_dataset = Dataset.load_from_disk("TCBC_datasets/sniplen5")
    #Training dataset
    train_ds = base_dataset.filter(lambda x: x['book_id'] in train_keys).shuffle()
    train_ds = train_ds.map(mapLabels, batched=True)
    train_ds = train_ds.map(mapConlluData2RawText, batched=True)
    #Evaluation dataset
    eval_ds = base_dataset.filter(lambda x: x['book_id'] in eval_keys).shuffle()
    eval_ds = eval_ds.map(mapLabels, batched=True)
    eval_ds = eval_ds.map(mapConlluData2RawText, batched=True)
    #Test dataset
    test_ds = base_dataset.filter(lambda x: x['book_id'] in test_keys).shuffle()
    test_ds = test_ds.map(mapLabels, batched=True)
    test_ds = test_ds.map(mapConlluData2RawText, batched=True)

    #Encode label
    train_ds = train_ds.class_encode_column('label')
    eval_ds = eval_ds.class_encode_column('label')
    test_ds = test_ds.class_encode_column('label')

    #Combine into a DatasetDict for cleaner code
    dataset = DatasetDict({'train':train_ds, 'test':test_ds, 'eval':eval_ds})

    #Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    #Tokenizer. We truncate from start till 510 tokens (so that there is room for the special tokens)
    #We pad with [pad] so that every example has 512 input ids
    def tokenize(ex):
        return tokenizer(
            ex['data'],
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
            print("Learning rate is: ",args[0].to_dict()['learning_rate'])
            print("Train batch size is: ",args[0].to_dict()['per_device_train_batch_size'])
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
    def model_init():
        return transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    #Define hyperparam spaces to search
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64]),
        }

    # Set training arguments
    trainer_args = transformers.TrainingArguments(
        "checkpoints_"+str(SPLIT_ID),
        eval_strategy="steps",
        logging_strategy="steps",
        save_total_limit=2,
        eval_steps=100,
        logging_steps=100,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        load_best_model_at_end=True,
        remove_unused_columns=True,
        max_steps = 500,
        #num_train_epochs=5
    )

    #Something that the HF-optuna combo requires
    def compute_objective(metrics):
        return metrics["eval_accuracy"]
    
    #Trainer arguments
    trainer = transformers.Trainer(
        model_init=model_init,
        args=trainer_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        processing_class = tokenizer,
        data_collator = data_collator,
        compute_metrics=evaluate_score,
        callbacks=[early_stopping, training_logs]
    )

    #Hyperparam optimization
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=50,
        compute_objective=compute_objective,
    )

    pprint(best_run)

    
    """
    trainer.train()

    trainer.save_model("FinBERT_for_book_snippets_5_new_split_"+str(SPLIT_ID))

    eval_results = trainer.evaluate(dataset["test"])

    pprint(eval_results)

    print('Accuracy:', eval_results['eval_accuracy'])

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
    """

#Main function
def main(cmd_args):
    #args 0 is the split_id we use for training
    trainFinBERT(int(cmd_args[0]))
    
#Pass cmd args to main function
if __name__ == "__main__":
    main(sys.argv[1:])

