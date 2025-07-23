#Imports
from scripts import corpusMLfunctions as cmf
from datasets import logging, disable_caching, disable_progress_bars
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import json
import sys
import multiprocessing as mp
import os
import numpy as np
import shutil
#Constants
BASE_BEG = "SnippetDatasets/"
BASE_MID = "sniplen_"
BASE_END = ".jsonl"
KEYLISTS = "Keylists.jsonl"
#SNIPPET_LENS = ['5','10','25','50','75','100']
CHOSEN_PARAMS = [{'c':0.15, 'tol':1e-6}, {'c':5, 'tol':1e-4}, {'c':60, 'tol':1e-4}, {'c':120, 'tol':1e-4}]
#Set logging to not be as annoying
logging.set_verbosity(40)

#Defining functions for the program
def do_nothing(ex):
    return ex

def whitespace_tokenizer(ex):
    return ex.split(" ")

#Version for only using TfIdfVectorizer with raw text as input
def manualStudy(params, SNIPPET_LENS, keylists, i, k, cache_dir, overwrite: bool=True):
    disable_progress_bars()
    filename = "TestResults/ParamOptim_List_"+str(i)+"_SnipLen_"+str(SNIPPET_LENS[k])+"_Results.jsonl"
    cache_file = cache_dir+str(i)+"_"+str(SNIPPET_LENS[k])+".jsonl"
    if overwrite or not os.path.exists(filename):
        hf_cache_dir = cache_dir+str(i)+"_"+str(SNIPPET_LENS[k])+"_ds"
        train_keys = keylists[i]['train_keys']
        #Temporary edit to test with combining eval+test as we are not param optimizing
        eval_keys = keylists[i]['eval_keys']+keylists[i]['train_keys']
        train_dss = cmf.combineSnippedBooksToDS(train_keys, SNIPPET_LENS[k], hf_cache_dir, cache_file, inc_hpfv=True, folder=BASE_BEG)
        eval_dss = cmf.combineSnippedBooksToDS(eval_keys, SNIPPET_LENS[k], hf_cache_dir, cache_file, inc_hpfv=True, folder=BASE_BEG)
        #Empty cache after we don't need it
        os.remove(cache_file)
        #with open(cache_file, 'w') as writer:
        #    writer.write("")
        #Continue on
        #vecd_train_data = csr_matrix(np.array(train_dss['hp_fv'], (np.vstack([[m]*len(train_dss['hp_fv'][0]) for m in range(len(train_dss['hp_fv']))]), np.hstack([[m]*len(train_dss['hp_fv'][0]) for m in range(len(train_dss['hp_fv']))]))))
        #print("Worker for length ",SNIPPET_LENS[k]," and keylist ",i," activated!")
        returnable = []
        for pair in params:
            #Train a new classifier for each set of params
                clf = LinearSVC(
                    random_state=42,
                    C=pair['c'],
                    tol=pair['tol']
                )
                clf.fit(np.array(train_dss['hp_fv']), train_dss['label'])
                predicted = clf.predict(np.array(eval_dss['hp_fv']))
                f1 = f1_score(eval_dss['label'], predicted, average="macro")
                #Now we can query index2feature to get the feature names as we need
                high_prio = {}
                # make a list of (weight, index), sort it
                for j in clf.classes_:
                    lst=[]
                    for idx,weight in enumerate(clf.coef_[list(clf.classes_).index(j)]):
                        lst.append((weight,idx))
                    lst.sort() #sort
                    highest_prio = []
                    for weight,idx in lst[-100:][::-1]:
                        highest_prio.append(idx)
                    high_prio[j] = highest_prio
                returnable.append({'keylist_id':i, 'sniplen':SNIPPET_LENS[k], 'c':pair['c'], 'tol':pair['tol'], 'f1':f1, 'important_feats_7-8':high_prio['7-8'], 'important_feats_9-12':high_prio['9-12'], 'important_feats_13+':high_prio['13+']})
        with open(filename, 'w') as f:
            f.write('\n'.join(map(json.dumps, returnable)))
        shutil.rmtree(hf_cache_dir)

def testParamResults(permutations: int, sniplen: int, keylists: list):
    pool = mp.Pool(mp.cpu_count())
    SNIPPET_LENS = [sniplen]
    pbar = tqdm(total=permutations*len(SNIPPET_LENS))
    def update(*a):
     pbar.update(1)
    #Generate temporary cache dir to manage memory
    cache_dir = "cache_dir/"
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    for i in range(permutations):
        #Add to list the test results of our 'manual' study
        for k in range(len(SNIPPET_LENS)):
            pool.apply_async(manualStudy, [CHOSEN_PARAMS, SNIPPET_LENS, keylists, i, k, cache_dir, False], callback=update)
    #print("All running!")
    pool.close()
    #print("Pool closed!")
    pool.join()
    #print("Waiting done!")
    
    

#Main function
def main(cmd_args):
    #Fetch keylists
    keylists = []
    with open(KEYLISTS, 'r') as f:
        for line in f:
            keylists.append(json.loads(line))
    testParamResults(int(cmd_args[0]), cmd_args[1], keylists)
#Pass cmd args to main function
if __name__ == "__main__":
    main(sys.argv[1:])
