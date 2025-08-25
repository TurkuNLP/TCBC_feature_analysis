#Imports
from scripts import bookdatafunctions as bdf
from scripts import corpusMLfunctions as cmf
import pandas as pd
import numpy as np
from datasets import Dataset, disable_progress_bars
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import f1_score
import optuna
import json
import multiprocessing as mp
import shutil
import warnings
from tqdm import tqdm

#Constants
AGES = ['5','6','7','8','9','10','11','12','13','14','15']
BASE_BEG = "SnippetDatasets/"
BASE_MID = "sniplen_"
BASE_END = ".jsonl"
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


def generateIntervals(ages: list[str]):
    intervals = []
    #Powersets of 2 intervals
    for i in range(2, len(ages)-1):
        #temp = intervals[2]
        intervals.append((ages[:i], ages[i:]))
        #intervals[2] = temp
    #Powersets for 3 intervals
    for i in range(2, len(ages)-3):
        for j in range(i+2, len(ages)-1):
            #temp = intervals[3]
            intervals.append((ages[:i], ages[i:j], ages[j:]))
            #intervals[3] = temp
    #Powersets for 4 intervals
    for i in range(2, len(ages)-5):
        for j in range(i+2, len(ages)-3):
            for k in range(j+2, len(ages)-1):
                #temp = intervals[4]
                intervals.append((ages[:i], ages[i:j], ages[j:k], ages[k:]))
                #intervals[4] = temp
    return intervals

def reMapLabels(ex, intervals):
    age = ex['age']
    if age > 14:
        age = 15
    age = str(age)
    for n in intervals:
        if age in n:
            ex['label'] = n[0]+'-'+n[-1]
    return ex

def initDatasets():
    train_keys = keylists[0]['train_keys']
    eval_keys = keylists[0]['eval_keys']
    test_keys = keylists[0]['test_keys']
    #Also makes it easier to clean cache files and use space more efficiently
    cache_dir = "cache_dir/temp/"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.mkdir(cache_dir)
    train_ds = cmf.combineSnippedBooksToDS(train_keys, '50', cache_dir, cache_file=cache_dir+"0_50_train.jsonl", folder=BASE_BEG, inc_raw_text=True)
    eval_ds = cmf.combineSnippedBooksToDS(eval_keys, '50', cache_dir, cache_file=cache_dir+"0_50_eval.jsonl", folder=BASE_BEG, inc_raw_text=True)
    test_ds = cmf.combineSnippedBooksToDS(test_keys, '50', cache_dir, cache_file=cache_dir+"0_50_test.jsonl", folder=BASE_BEG, inc_raw_text=True)
    return train_ds, eval_ds, test_ds

#Code to run in parallel
def evaluateGroups(train_ds: Dataset, eval_ds: Dataset, test_ds: Dataset, intervals):
    #Map labels to match the intervals given (aka re-assign labels)
    train_ds = train_ds.map(reMapLabels, fn_kwargs={"intervals":intervals})
    eval_ds = eval_ds.map(reMapLabels, fn_kwargs={"intervals":intervals})
    test_ds = test_ds.map(reMapLabels, fn_kwargs={"intervals":intervals})
    #Initialize and fir our vectorizer
    vectorizer = TfidfVectorizer(norm='l2', tokenizer=whitespace_tokenizer, preprocessor=do_nothing, max_features=2000).fit(train_ds['raw_text'])
    #Vectorize datasets
    vectorized_train = vectorizer.transform(train_ds['raw_text'])
    vectorized_eval = vectorizer.transform(eval_ds['raw_text'])
    vectorized_test = vectorizer.transform(test_ds['raw_text'])

    returnable = {}
    c_eval_pairs= []
    
    #Very quick hyperparam optimization as we have computational resources
    def objective(trial):
        #Defining hyperparameters to tune
        c = trial.suggest_float('c', 1e-10, 1e+0, log=True)
        pen = trial.suggest_categorical('pen', ['l1', 'l2'])
        tol = trial.suggest_float('tol', 1e-10, 1e-3, log=True)
        clf = LinearSVC(
            random_state=42,
            C=c,
            tol=tol,
            penalty=pen
        )
        clf.fit(vectorized_train, train_ds['label'])
        predicted = clf.predict(vectorized_eval)
        f1 = f1_score(eval_ds['label'], predicted, average="macro")
        c_eval_pairs.append([c, f1])
        return f1

    # Your code for hyperparameter optimization here
    study = optuna.create_study(direction='maximize')
    optuna.logging.disable_default_handler()
    study.optimize(objective, n_trials=50)

    #Run with best params
    clf = LinearSVC(
        penalty=study.best_trial.params['pen'],
        random_state=42,
        C=study.best_trial.params['c'],
        tol=study.best_trial.params['tol'],
    )
    clf.fit(vectorized_train, train_ds['label'])
    test_predict = clf.predict(vectorized_test)

    #Generate unique id
    id = "_".join([x[0]+"-"+x[-1] for x in intervals])

    #Assign returnble values
    returnable['f1'] = f1_score(test_ds['label'], test_predict, average="macro")
    returnable['id'] = id
    returnable['labels'] = clf.classes_.tolist()
    returnable['conf_matrix'] = metrics.confusion_matrix(test_ds['label'], test_predict).tolist()
    returnable['c'] = study.best_trial.params['c']
    returnable['tol'] = study.best_trial.params['tol']
    returnable['penalty'] = study.best_trial.params['pen']
    returnable['c_eval_scores'] = c_eval_pairs

    #Write JSON-file
    with open("TestResults/DifferentAgeGroups_tfidf/"+id+".json", 'w') as f:
        f.write(json.dumps(returnable))

    return None

#Main function
def main():
    warnings.filterwarnings('ignore') 
    os.environ['PYTHONWARNINGS']='ignore'
    disable_progress_bars()
    train_ds_base, eval_ds_base, test_ds_base = initDatasets()
    interval_splits = generateIntervals(AGES)
    pbar = tqdm(total=len(interval_splits))
    def update(*a):
     pbar.update(1)
    pool = mp.Pool(len(os.sched_getaffinity(0)))
    for i in interval_splits:
        pool.apply_async(evaluateGroups, [train_ds_base, eval_ds_base, test_ds_base, i], callback=update)
    #print("All running!")
    pool.close()
    #print("Pool closed!")
    pool.join()
    print("Waiting done!")

    
#Pass cmd args to main function
if __name__ == "__main__":
    main()