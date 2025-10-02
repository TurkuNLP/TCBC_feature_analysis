#Imports
from TCBC_tools import Structure, MachineLearning as ml
import pandas as pd
import numpy as np
from datasets import Dataset
import os
import json
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
    
def minMaxNormalization(min_vector: list, max_vector:list, feature_vector:list):
    """
    Helper function for performing min-max normalization for feature vectors
    """
    to_return = []
    for i in range(len(feature_vector)):
        min_max_neg = (max_vector[i]-min_vector[i])
        if min_max_neg == 0:
            to_return.append(0)
        else:
            to_return.append((feature_vector[i]-min_vector[i])/(max_vector[i]-min_vector[i]))
    return to_return

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

def generateHPFVForDataset(ex):
    conllu = ex['data']
    df = Structure.snippetConllu2DF(conllu)
    hpfv = ml.customConlluVectorizer(df)
    ex['data'] = hpfv
    return ex


#Main function
def main(cmd_args):
    #Read sniplen wanted
    sniplen = int(cmd_args[0])
    #Generate and save to disk sniplen ??
    raw_ds = Dataset.load_from_disk("TCBC_datasets/sniplen"+str(sniplen)).map(generateHPFVForDataset, num_proc=len(os.sched_getaffinity(0)))
    #Filter out non novels
    raw_ds = raw_ds.filter(lambda x: x['book_id'][-1] == '1')
    #Normalizing the data
    all_values = np.array(raw_ds['data'])
    maxs = all_values.max(axis=0)
    mins = all_values.min(axis=0)
    def map2MinMax(ex):
        return {'data':[minMaxNormalization(mins, maxs, x) for x in ex['data']]}
    #Write to disk the normalized version
    raw_ds.map(map2MinMax, batched=True).save_to_disk("TCBC_datasets/sniplen"+str(sniplen)+"_hpfv_novels", num_proc=len(os.sched_getaffinity(0)))
#Pass cmd args to main function
if __name__ == "__main__":
    main(sys.argv[1:])
