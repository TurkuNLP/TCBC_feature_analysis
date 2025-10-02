#Imports
from TCBC_tools import Structure, MachineLearning as ml
import pandas as pd
from datasets import Dataset, disable_progress_bars
import os
from sklearn.metrics import f1_score
import json
from pprint import pprint
import sys

#Constants
AGES = ['5','6','7','8','9','10','11','12','13','14','15']
KEYLISTS = "Keylists.jsonl"
keylists = []
with open(KEYLISTS, 'r') as f:
    for line in f:
        keylists.append(json.loads(line))

#Helper functions
def do_nothing(ex):
    return ex.lower()

def conllu_tokenizer(ex):
    return ex.replace("\n", "\t").replace("|", "\t").split("\t")

def whitespace_tokenizer(ex):
    return ex.split(" ")

def assignLabel(ex):
    age = int(Structure.findAgeFromID(ex))
    if age < 9:
        return '7-8'
    elif age < 13:
        return '9-12'
    else:
        return '13+'

#Function for generating a snippet dataset
def generateSnippetDatasetFromRawConllu(ex, sniplen):
        conllu = ex['data'][0]
        new_data = []
        label = assignLabel(ex['book_id'][0])
        df = Structure.snippetConllu2DF(conllu)
        tree = Structure.buildIdTreeFromConllu(df)
        df.loc[df['upos'] == 'PROPN', 'lemma'] = "[MASK]"
        df.loc[df['upos'] == 'PROPN', 'text'] = "[MASK]"
        conllu_lines = ["\t".join(x) for x in df.to_numpy("str").tolist()]
        tree_heads = list(tree.keys())
        start = 0
        for i in range(sniplen-1, len(tree), sniplen):
            new_data.append("\n".join(conllu_lines[start:list(tree[tree_heads[i]].keys())[-1]+1]))
            start = list(tree[tree_heads[i]].keys())[-1]+1
        return {"book_id":[ex['book_id'][0]]*len(new_data), "data": new_data, 'label':[label]*len(new_data)}

#Main function
def main(cmd_args):
    #Read sniplen wanted
    sniplen = int(cmd_args[0])
    #Generate and save to disk sniplen ??
    Dataset.load_from_disk("cache_dir/RawDataset").map(ml.buildDatasetFromRawConllus, fn_kwargs={"sniplen":sniplen}, batched=True, batch_size=1, remove_columns=['book_id', 'data'], num_proc=len(os.sched_getaffinity(0))).save_to_disk("TCBC_datasets/sniplen"+str(sniplen), num_proc=len(os.sched_getaffinity(0)))
#Pass cmd args to main function
if __name__ == "__main__":
    main(sys.argv[1:])
