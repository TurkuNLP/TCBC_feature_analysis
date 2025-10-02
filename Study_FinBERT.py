# %%
#Imports
from datasets import Dataset, DatasetDict
from scripts import corpusMLfunctions as cmf
from TCBC_tools import Structure
import transformers
from pprint import PrettyPrinter
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import os
import json
import sys

# %%
#Some constants and other housekeeping before diving into coding
pprint = PrettyPrinter(compact=True).pprint
os.environ['WANDB_MODE'] = 'disabled'
LABELS = ['7-8', '9-12', '13+']

# %%
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

def swapBackLabels(y):
    y_swap = []
    for p in y:
        if p == 0:
            y_swap.append('13+')
        elif p == 1:
            y_swap.append('7-8')
        else:
            y_swap.append('9-12')
    return y_swap


def mapLabels(ex):
    return {'label':[assignLabel(x) for x in ex['book_id']]}

keylists = []
with open("Keylists.jsonl", 'r') as f:
    for line in f:
        keylists.append(json.loads(line))
def testFinBERT(SPLIT_ID):
    MODEL_NAME = "FinBERT_for_book_snippets_5_new_split_"+str(SPLIT_ID)+"/"
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

    dataset = DatasetDict({'train':train_ds, 'test':test_ds, 'eval':eval_ds})

    # %%
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    #Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    #Data collator so that everything works as intended :)
    data_collator = transformers.DataCollatorWithPadding(tokenizer)

    # %%
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
        y_pred = swapBackLabels(y_pred)
        y_true = pred.label_ids
        y_true = swapBackLabels(y_true)
        return {'accuracy' : f1_score(y_true, y_pred, average='macro', labels=LABELS), 'report':classification_report(y_true, y_pred, labels=LABELS), 'conf_matrix':confusion_matrix(y_true, y_pred, labels=LABELS).tolist()}

    # %%
    #Trainer arguments
    trainer = transformers.Trainer(
        model=model,
        processing_class = tokenizer,
        data_collator = data_collator,
        compute_metrics=evaluate_score,
    )

    # %%
    test_results = trainer.evaluate(dataset['test'])

    with open("TestResults/FinBERT_split_"+str(SPLIT_ID)+".json", 'w') as writer:
        writer.write(json.dumps(test_results))

    pprint(test_results)

    #print('Accuracy:', eval_results['eval_accuracy'])

#Main function
def main(cmd_args):
    #args 0 is the split_id we use for training
    testFinBERT(int(cmd_args[0]))
    
#Pass cmd args to main function
if __name__ == "__main__":
    main(sys.argv[1:])


