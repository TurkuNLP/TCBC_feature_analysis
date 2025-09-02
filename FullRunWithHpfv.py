#Imports
from scripts import corpusMLfunctions as cmf
from datasets import logging, disable_progress_bars
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, confusion_matrix
import json
import sys
import multiprocessing as mp
import os
import numpy as np
from datasets import Dataset
from scripts import bookdatafunctions as bdf
import optuna
import warnings
import scipy as sp
from sklearn.preprocessing import normalize
#Constants
BASE_BEG = "SnippetDatasets/"
BASE_MID = "sniplen_"
BASE_END = ".jsonl"
KEYLISTS = "Keylists.jsonl"
SNIPPET_LENS = ['5','10','25','50','75','100']
#Set logging to not be as annoying
logging.set_verbosity(40)

#Defining functions for the program
def do_nothing(ex):
    return ex

def whitespace_tokenizer(ex):
    return ex.split(" ")

def assignLabel(ex):
    age = int(bdf.findAgeFromID(ex))
    if age < 9:
        return '7-8'
    elif age < 13:
        return '9-12'
    else:
        return '13+'

def conllu2RawLemmas(conllu_text):
    conllu_lines = conllu_text.split("\n")
    return " ".join([x.split('\t')[2] for x in conllu_lines if len(x) > 0])


def mapLabels(ex):
    return {'label':[assignLabel(x) for x in ex['book_id']]}

def mapConlluData2RawLemmas(ex):
    return {'data':[conllu2RawLemmas(x) for x in ex['data']]}   

#Version for only using TfIdfVectorizer with raw text as input
def manualStudy(SNIPPET_LENS, keylists, i, k, base_dataset, overwrite: bool=True, multiple_jobs: int=1):
    disable_progress_bars()
    filename = "TestResults/FullResultOnlyHPFVNew/OnlyHpfv_List_"+str(i)+"_SnipLen_"+str(SNIPPET_LENS[k])+"_Results.jsonl"
    if overwrite or not os.path.exists(filename):
        train_keys = keylists[i]['train_keys']
        eval_keys = keylists[i]['eval_keys']
        test_keys = keylists[i]['test_keys']
        #Training dataset
        train_ds = base_dataset.filter(lambda x: x['book_id'] in train_keys).shuffle()
        train_ds = train_ds.map(mapLabels, batched=True, batch_size=32, load_from_cache_file=False)
        #Evaluation dataset
        eval_ds = base_dataset.filter(lambda x: x['book_id'] in eval_keys).shuffle()
        eval_ds = eval_ds.map(mapLabels, batched=True, batch_size=32, load_from_cache_file=False)
        #Test dataset
        test_ds = base_dataset.filter(lambda x: x['book_id'] in test_keys).shuffle()
        test_ds = test_ds.map(mapLabels, batched=True, batch_size=32, load_from_cache_file=False)
        
        #Vectorize data
        vectorized_train = sp.sparse.coo_array(np.array(train_ds['data']))
        vectorized_eval = sp.sparse.coo_array(np.array(eval_ds['data']))
        vectorized_test = sp.sparse.coo_array(np.array(test_ds['data']))

        returnable = {}
        c_eval_pairs= []

        #Very quick hyperparam optimization as we have computational resources
        def objective(trial):
            #Defining hyperparameters to tune
            c = trial.suggest_float('c', 1e-10, 1e+0, log=True)
            pen = trial.suggest_categorical('pen', ['l2'])
            tol = trial.suggest_float('tol', 1e-10, 1e-3, log=True)
            clf = LinearSVC(
                random_state=42,
                C=c,
                tol=tol,
                penalty=pen
            )
            clf.fit(vectorized_train,np.array(train_ds['label']))
            predicted = clf.predict(vectorized_eval)
            f1 = f1_score(np.array(eval_ds['label']), predicted, average="macro")
            c_eval_pairs.append([c, f1])
            return f1

        # Your code for hyperparameter optimization here
        study = optuna.create_study(direction='maximize')
        optuna.logging.disable_default_handler()
        study.optimize(objective, n_trials=25, n_jobs=multiple_jobs)

        #Run with best params
        clf = LinearSVC(
            penalty=study.best_trial.params['pen'],
            random_state=42,
            C=study.best_trial.params['c'],
            tol=study.best_trial.params['tol'],
        )
        clf.fit(vectorized_train, np.array(train_ds['label']))
        test_predict = clf.predict(vectorized_test)

        #Assign returnble values
        returnable['keylist_id'] = i
        returnable['f1'] = f1_score(test_ds['label'], test_predict, average="macro")
        returnable['labels'] = clf.classes_.tolist()
        returnable['conf_matrix'] = confusion_matrix(test_ds['label'], test_predict).tolist()
        returnable['c'] = study.best_trial.params['c']
        returnable['tol'] = study.best_trial.params['tol']
        returnable['penalty'] = study.best_trial.params['pen']
        returnable['c_eval_scores'] = c_eval_pairs
        #Add important features        
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
        for h in high_prio:
            returnable['important_features_for_'+h] = high_prio[h]
        with open(filename, 'w') as f:
            f.write(json.dumps(returnable))

def testParamResults(permutations: int, sniplen: int, keylists: list):
    warnings.filterwarnings('ignore') 
    os.environ['PYTHONWARNINGS']='ignore'
    disable_progress_bars()
    pbar = tqdm(total=permutations)
    def update(*a):
     pbar.update(1)
    #For CSC environments
    pool = mp.Pool(len(os.sched_getaffinity(0)))
    for i in range(permutations):
        #Add to list the test results of our 'manual' study
            base_dataset = Dataset.load_from_disk("TCBC_datasets/sniplen"+str(SNIPPET_LENS[sniplen])+"_hpfv")
            #manualStudy(SNIPPET_LENS, keylists, i, k, base_dataset, True)
            #pbar.update(1)
            pool.apply_async(manualStudy, [SNIPPET_LENS, keylists, i, sniplen, base_dataset, True], callback=update)
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
    testParamResults(int(cmd_args[0]), int(cmd_args[1]), keylists)
#Pass cmd args to main function
if __name__ == "__main__":
    main(sys.argv[1:])
