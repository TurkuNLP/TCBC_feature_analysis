from TCBC_tools import Structure
from scripts import CustomOVOClassifier as covoc
from datasets import Dataset, logging, disable_progress_bars
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, confusion_matrix
import optuna
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy as sp
import sys
import os
import warnings
from pprint import pprint
import multiprocessing as mp
from tqdm import tqdm

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)



#Constants
keylists = []
#Helper functions
def do_nothing(ex):
    return ex

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
    
def mapLabels(ex):
    return {'label':[assignLabel(x) for x in ex['book_id']]}

def conllu2RawLemmas(conllu_text):
    conllu_lines = conllu_text.split("\n")
    return " ".join([x.split('\t')[2] for x in conllu_lines if len(x) > 0])

def mapConlluData2RawLemmas(ex):
    return {'data':[conllu2RawLemmas(x) for x in ex['data']]}

def getFeaturesForEachBinaryPair(clf, vocab=None):
        labels = clf.classes_
        estimator_keys = [str(labels[i])+" vs. "+str(labels[j]) for i in range(3) for j in range(i+1, 3)]
        estimators = {estimator_keys[i]:clf.estimators_[i] for i in range(3)}
        returnable = {}
        for e in estimators:
            lst=[]
            #If tfidf and we cannot reverse engineer feature-indices later, do it now
            if vocab:
                index2feature = {}
                for feature,idx in vocab.items():
                    assert idx not in index2feature #This really should hold
                    index2feature[idx]=feature
                for idx,weight in enumerate(estimators[e].coef_[0]):
                    lst.append((weight,index2feature[idx]))
            #If hpfv, meanining we can get all of this later on, don't waste time here :)
            else:
                for idx,weight in enumerate(estimators[e].coef_[0]):
                    lst.append((weight,idx))
            returnable[e] = lst
        return returnable
    

def _resolve_dataset_path(sniplen: int, tfidf: bool, keylist_type: str) -> str:
    """Build the dataset path string once, pass it to workers (not the dataset itself)."""
    if tfidf:
        return f"TCBC_datasets/sniplen{sniplen}"
    elif keylist_type == "novels":
        return f"TCBC_datasets/sniplen{sniplen}_hpfv_novels"
    else:
        return f"TCBC_datasets/sniplen{sniplen}_hpfv"


def _resolve_hpo_path(sniplen: int, keylist_type: str, tfidf: bool) -> str:
    """Single shared HPO result file per sniplen + keylist_type + feature mode."""
    suffix = "_tfidf" if tfidf else "_hpfv"
    return (
        f"TestResults/COVOC_hyperparams/"
        f"hpo_results_sniplen_{sniplen}_{keylist_type}{suffix}.jsonl"
    )
    

#Important functions

from concurrent.futures import ProcessPoolExecutor, as_completed

def optimize_pair(pair, train_pair_ds, eval_pair_ds, tfidf, num_of_rounds, n_jobs_per_pair):
    """Runs optimization for a single label pair."""
    if tfidf:
        vectorizer = TfidfVectorizer(
            norm='l2', tokenizer=whitespace_tokenizer,
            preprocessor=do_nothing, max_features=2000
        ).fit(train_pair_ds['data'])
        vectorized_train = vectorizer.transform(train_pair_ds['data'])
        vectorized_eval = vectorizer.transform(eval_pair_ds['data'])
    else:
        dense = np.array(train_pair_ds['data'])
        vectorized_train = sp.sparse.csr_matrix(dense)
        del dense
        dense = np.array(eval_pair_ds['data'])
        vectorized_eval = sp.sparse.csr_matrix(dense)
        del dense

    train_labels = train_pair_ds['label']
    eval_labels = eval_pair_ds['label']

    def objective(trial):
        c   = trial.suggest_float('c',   1e-10, 1e+0, log=True)
        tol = trial.suggest_float('tol', 1e-10, 1e-3, log=True)
        clf = LinearSVC(random_state=42, C=c, tol=tol, penalty='l2')
        clf.fit(vectorized_train, train_labels)
        predicted = clf.predict(vectorized_eval)
        return f1_score(eval_labels, predicted, average="macro")

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=15)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=num_of_rounds, n_jobs=n_jobs_per_pair)

    return {
        'id':  pair,
        'f1':  study.best_value,
        'c':   study.best_params['c'],
        'pen': 'l2',
        'tol': study.best_params['tol'],
    }

def getOptimHyperparam(base_dataset: Dataset, num_of_rounds: int, tfidf: bool):
    """Run HPO using keylist 0 as the representative train/eval split."""
    n_cpus = len(os.sched_getaffinity(0))
    labels = ['13+', '7-8', '9-12']
    train_keys = keylists[0]['train_keys']
    eval_keys = keylists[0]['eval_keys']

    logger.info('Now arrived at param optimization (shared across all keylists)')

    #Training dataset
    train_ds = base_dataset.filter(lambda x: x['book_id'] in train_keys, num_proc=n_cpus).shuffle()
    train_ds = train_ds.map(mapLabels, batched=True, num_proc=n_cpus)
    if tfidf:
        train_ds = train_ds.map(mapConlluData2RawLemmas, batched=True, num_proc=n_cpus)
    #Evaluation dataset
    eval_ds = base_dataset.filter(lambda x: x['book_id'] in eval_keys, num_proc=n_cpus).shuffle()
    eval_ds = eval_ds.map(mapLabels, batched=True, num_proc=n_cpus)
    if tfidf:
        eval_ds = eval_ds.map(mapConlluData2RawLemmas, batched=True, num_proc=n_cpus)

    logger.info('Datasets have been loaded for HPO')

    #Filter datasets to fit our needs
    label_pairs = {
        f"{i}_{j}": {labels[i], labels[j]}
        for i in range(3) for j in range(i+1, 3)
    }
    train_dss = {
        key: train_ds.filter(
            lambda x, lset=lset: x['label'] in lset,
            num_proc=n_cpus
        ) for key, lset in label_pairs.items()
    }
    eval_dss = {
        key: eval_ds.filter(
            lambda x, lset=lset: x['label'] in lset,
            num_proc=n_cpus
        ) for key, lset in label_pairs.items()
    }
    logger.info('Datasets have been filtered for HPO')
    n_pairs = len(train_dss)  # 3
    n_jobs_per_pair = max(1, n_cpus // n_pairs)

    with ProcessPoolExecutor(max_workers=n_pairs) as executor:
        futures = {
            executor.submit(
                optimize_pair, pair, train_dss[pair], eval_dss[pair],
                tfidf, num_of_rounds, n_jobs_per_pair
            ): pair for pair in train_dss
        }
        best_params = [f.result() for f in as_completed(futures)]
    
    return best_params


def testCOVOC(
    dataset_path: str,
    sniplen: int,
    keylist_num: int,
    tfidf: bool,
    keylist_type: str,
    keylist: dict,
):
    """
    Each invocation runs in its own process.  All heavy operations
    (filter / map) run single-threaded (num_proc=1) because the
    outer pool already saturates all CPUs.
    """
    
    # 1.  Load dataset inside the worker → memory-mapped, zero-copy
    base_dataset = Dataset.load_from_disk(dataset_path)

    train_keys: set = keylist["train_keys"]
    test_keys: set = keylist["test_keys"]

    
    # 2.  Filter + transform with num_proc=1
    train_ds = (
        base_dataset
        .filter(lambda x: x['book_id'] in train_keys, num_proc=1)
        .shuffle()
    )
    train_ds = train_ds.map(mapLabels, batched=True, num_proc=1)

    test_ds = (
        base_dataset
        .filter(lambda x: x['book_id'] in test_keys, num_proc=1)
        .shuffle()
    )
    test_ds = test_ds.map(mapLabels, batched=True, num_proc=1)

    if tfidf:
        train_ds = train_ds.map(
            mapConlluData2RawLemmas, batched=True, num_proc=1
        )
        test_ds = test_ds.map(
            mapConlluData2RawLemmas, batched=True, num_proc=1
        )

    
    # 3.  Load hyperparameters — now from the SHARED file
    
    param_file = _resolve_hpo_path(sniplen, keylist_type, tfidf)
    with open(param_file, 'r') as reader:
        best_params = [json.loads(line) for line in reader]

    lsvcs = [LinearSVC(C=p['c'], tol=p['tol']) for p in best_params]
    estimators = {"0_1": lsvcs[0], "0_2": lsvcs[1], "1_2": lsvcs[2]}

    
    # 4.  Vectorize
    
    if tfidf:
        vectorizer = TfidfVectorizer(
            norm='l2',
            tokenizer=whitespace_tokenizer,
            preprocessor=do_nothing,
            max_features=2000,
        ).fit(train_ds['data'])
        vectorized_train = vectorizer.transform(train_ds['data'])
        vectorized_test = vectorizer.transform(test_ds['data'])
    else:
        vectorized_train = sp.sparse.vstack(
            [sp.sparse.csr_array(np.asarray(row, dtype=np.float32).reshape(1, -1))
             for row in train_ds['data']],
            format='csr',
        )
        vectorized_test = sp.sparse.vstack(
            [sp.sparse.csr_array(np.asarray(row, dtype=np.float32).reshape(1, -1))
             for row in test_ds['data']],
            format='csr',
        )

    
    # 5.  Train & evaluate
    
    clf = covoc.CustomOneVsOneClassifier(estimators)
    clf.fit(vectorized_train, train_ds['label'])
    test_predict = clf.predict(vectorized_test)

    returnable = {
        'sniplen': sniplen,
        'keylist_id': keylist_num,
        'f1': f1_score(test_ds['label'], test_predict, average="macro"),
        'labels': clf.classes_.tolist(),
        'conf_matrix': confusion_matrix(test_ds['label'], test_predict).tolist(),
        'bookid_prediction': list(zip(test_ds['book_id'], test_predict.tolist())),
    }

    # Feature importances
    if tfidf:
        feature_importance = getFeaturesForEachBinaryPair(clf, vectorizer.vocabulary_)
    else:
        feature_importance = getFeaturesForEachBinaryPair(clf)
    returnable.update(feature_importance)

    # Write individual result
    out_suffix = "_tfidf.json" if tfidf else "_hpfv.json"
    out_path = (
        f"TestResults/COVOC_FullResult/Sniplen_{sniplen}_Keylist_"
        f"{keylist_num}_{keylist_type}{out_suffix}"
    )
    with open(out_path, 'w') as f:
        json.dump(returnable, f)

    return returnable
    
def hyperparamOptimize(sniplen: int, num_of_rounds: int, tfidf: bool, keylist_type: str):
    """
    Run HPO once for a given sniplen + keylist_type + feature mode.
    Uses keylist 0 as the representative train/eval split.
    Skips entirely if the result file already exists.
    """
    warnings.filterwarnings('ignore') 
    os.environ['PYTHONWARNINGS'] = 'ignore'
    disable_progress_bars()

    filename = _resolve_hpo_path(sniplen, keylist_type, tfidf)

    if os.path.exists(filename):
        logger.info(
            f"HPO already done for sniplen={sniplen}, "
            f"keylist_type={keylist_type}, tfidf={tfidf} — skipping. "
            f"({filename})"
        )
        return

    if tfidf:
        base_dataset = Dataset.load_from_disk("TCBC_datasets/sniplen" + str(sniplen))
    elif keylist_type == "novels":
        base_dataset = Dataset.load_from_disk("TCBC_datasets/sniplen" + str(sniplen) + "_hpfv_novels")
    else:
        base_dataset = Dataset.load_from_disk("TCBC_datasets/sniplen" + str(sniplen) + "_hpfv")

    results = getOptimHyperparam(base_dataset, num_of_rounds, tfidf)
    with open(filename, 'w') as f:
        f.write('\n'.join(map(json.dumps, results)))

    logger.info(f"HPO results written to {filename}")


def doFullRun(sniplen: int, tfidf: bool, keylist_type: str, keylists: list):
    warnings.filterwarnings('ignore')
    disable_progress_bars()

    dataset_path = _resolve_dataset_path(sniplen, tfidf, keylist_type)

    # Pre-validate the dataset exists
    _ = Dataset.load_from_disk(dataset_path)

    n_cpus = len(os.sched_getaffinity(0))
    
    max_workers = n_cpus

    # Pre-convert key lists to sets so every worker, so we do the conversion once instead of 100 times.
    prepared_keylists = [
        {
            "train_keys": set(kl["train_keys"]),
            "test_keys": set(kl["test_keys"]),
        }
        for kl in keylists
    ]

    results = [None] * 100
    completed_count = 0
    failed_count = 0
    mode = "hpfv"

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                testCOVOC,
                dataset_path,
                sniplen,
                i,
                tfidf,
                keylist_type,
                prepared_keylists[i],
            ): i
            for i in range(100)
        }

        # tqdm progress bar over completed futures
        with tqdm(
            total=100,
            desc=f"sniplen={sniplen} {mode}",
            unit="task",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        ) as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                    f1 = results[idx]['f1']
                    completed_count += 1
                    pbar.set_postfix(
                        last_task=idx,
                        f1=f"{f1:.4f}",
                        failed=failed_count,
                    )
                    logger.debug(
                        "Task %d completed — F1=%.4f", idx, f1
                    )
                except Exception:
                    failed_count += 1
                    pbar.set_postfix(last_task=idx, failed=failed_count)
                    logger.exception("Task %d FAILED", idx)
                pbar.update(1)

    with open("TestResults/all_results_"+str(sniplen)+"_"+keylist_type+"_.jsonl", 'w', encoding='utf-8') as writer:
        for res in results:
            writer.write(json.dumps(res) + '\n')


#Main function
def main(cmd_args):
    #Check which keylist we use!
    sniplen = int(cmd_args[0])
    num_of_rounds = int(cmd_args[1])
    tfidf = bool(int(cmd_args[2]))
    keylist_type = cmd_args[3]
    optimize = bool(int(cmd_args[4]))
    if keylist_type == 'novels':
        with open("NewKeylists_only_novels.jsonl", 'r') as f:
            for line in f:
                keylists.append(json.loads(line))
    elif keylist_type == 'without_author':
        with open("NewKeylists.jsonl", 'r') as f:
            for line in f:
                keylists.append(json.loads(line))
    else:
        with open("Keylists.jsonl", 'r') as f:
            for line in f:
                keylists.append(json.loads(line))
    #For performing the hyperparam optimization
    if optimize:
        hyperparamOptimize(sniplen, num_of_rounds, tfidf, keylist_type)
    #For performing the tests
    else:
        doFullRun(sniplen, tfidf, keylist_type, keylists)
    
#Pass cmd args to main function
if __name__ == "__main__":
    main(sys.argv[1:])