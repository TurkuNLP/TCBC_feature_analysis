from scripts import bookdatafunctions as bdf
from scripts import corpusMLfunctions as cmf
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

#Constants
keylists = []
with open("NewKeylists_only_novels.jsonl", 'r') as f:
    for line in f:
        keylists.append(json.loads(line))

#Helper functions

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
    
#Important functions

def getOptimHyperparam(base_dataset: Dataset, keylist_num: int, num_of_rounds: int, tfidf: bool):
    labels = ['13+', '7-8', '9-12']
    train_keys = keylists[keylist_num]['train_keys']
    eval_keys = keylists[keylist_num]['eval_keys']

    #Training dataset
    train_ds = base_dataset.filter(lambda x: x['book_id'] in train_keys).shuffle()
    train_ds = train_ds.map(mapLabels, batched=True)
    if tfidf:
        train_ds = train_ds.map(mapConlluData2RawLemmas, batched=True)
    #Evaluation dataset
    eval_ds = base_dataset.filter(lambda x: x['book_id'] in eval_keys).shuffle()
    eval_ds = eval_ds.map(mapLabels, batched=True)
    if tfidf:
        eval_ds = eval_ds.map(mapConlluData2RawLemmas, batched=True)

    #Filter datasets to fit our needs
    train_dss = {str(i)+"_"+str(j):train_ds.filter(lambda x: x['label'] == labels[i] or x['label'] == labels[j]) for i in range(3) for j in range(i+1, 3)}
    eval_dss = {str(i)+"_"+str(j):eval_ds.filter(lambda x: x['label'] == labels[i] or x['label'] == labels[j]) for i in range(3) for j in range(i+1, 3)}

    best_params = []
    
    for pair in train_dss:
        optimal = {}

        train_ds = train_dss[pair]
        eval_ds = eval_dss[pair]

        #If using tfidf
        if tfidf:
            vectorizer = TfidfVectorizer(norm='l2', tokenizer=whitespace_tokenizer, preprocessor=do_nothing, max_features=2000).fit(train_ds['data'])
            #Vectorize data
            vectorized_train = vectorizer.transform(train_ds['data'])
            vectorized_eval = vectorizer.transform(eval_ds['data'])
        #If using hpfv
        else:
            #Transform into sparse arrays
            vectorized_train = sp.sparse.coo_array(np.array(train_ds['data'])).tocsr()
            vectorized_eval = sp.sparse.coo_array(np.array(eval_ds['data'])).tocsr()
        #Hyperparam optimization as we have computational resources
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
            clf.fit(vectorized_train,train_ds['label'])
            predicted = clf.predict(vectorized_eval)
            f1 = f1_score(eval_ds['label'], predicted, average="macro")
            return f1

        # Your code for hyperparameter optimization here
        study = optuna.create_study(direction='maximize')
        #optuna.logging.disable_default_handler()
        study.optimize(objective, n_trials=num_of_rounds, n_jobs=len(os.sched_getaffinity(0)))

        optimal['id'] = pair
        optimal['f1'] = study.best_value
        optimal['c'] = study.best_params['c']
        optimal['pen'] = study.best_params['pen']
        optimal['tol'] = study.best_params['tol']

        best_params.append(optimal)
    
    return best_params

def testCOVOC(base_dataset: Dataset, sniplen: int, keylist_num: int, tfidf: bool, keylist_type):
    train_keys = keylists[keylist_num]['train_keys']
    test_keys = keylists[keylist_num]['test_keys']

    #Training dataset
    train_ds = base_dataset.filter(lambda x: x['book_id'] in train_keys).shuffle()
    train_ds = train_ds.map(mapLabels, batched=True)
    if tfidf:
        train_ds = train_ds.map(mapConlluData2RawLemmas, batched=True)
    #Test dataset
    test_ds = base_dataset.filter(lambda x: x['book_id'] in test_keys).shuffle()
    test_ds = test_ds.map(mapLabels, batched=True)
    if tfidf:
        test_ds = test_ds.map(mapConlluData2RawLemmas, batched=True)

    #Load optimized parameters
    best_params = []
    filename = "TestResults/COVOC_hyperparams/COVOC_hyperparams_sniplen_"+str(sniplen)+"_keylist_"+str(keylist_num)+"_"+keylist_type
    if tfidf:
        filename += "_tfidf"
    else:
        filename += "_hpfv"
    filename += ".jsonl"
    with open(filename, 'r') as reader:
        for line in reader:
            best_params.append(json.loads(line))
    #Initialize LinearSVC models used in our customized OneVsOneClassifier
    lsvcs = [LinearSVC(C=x['c'], tol=x['tol']) for x in best_params]
    estimators = {"0_1":lsvcs[0], "0_2":lsvcs[1], "1_2":lsvcs[2]}

    #Vectorize our data
    #If using tfidf
    if tfidf:
        vectorizer = TfidfVectorizer(norm='l2', tokenizer=whitespace_tokenizer, preprocessor=do_nothing, max_features=2000).fit(train_ds['data'])
        #Vectorize data
        vectorized_train = vectorizer.transform(train_ds['data'])
        vectorized_test = vectorizer.transform(test_ds['data'])
    #If using hpfv
    else:
        #Transform into sparse arrays
        vectorized_train = sp.sparse.coo_array(np.array(train_ds['data'])).tocsr()
        vectorized_test = sp.sparse.coo_array(np.array(test_ds['data'])).tocsr()

    #Train our classifier and evaluate performance
    returnable = {}
    #Run with best params
    clf = covoc.CustomOneVsOneClassifier(estimators)
    clf.fit(vectorized_train, train_ds['label'])
    test_predict = clf.predict(vectorized_test)

    #Assign returnable values
    returnable['sniplen'] = sniplen
    returnable['keylist_id'] = keylist_num
    returnable['f1'] = f1_score(test_ds['label'], test_predict, average="macro")
    returnable['labels'] = clf.classes_.tolist()
    returnable['conf_matrix'] = confusion_matrix(test_ds['label'], test_predict).tolist()
    returnable['bookid_prediction'] = [[test_ds['book_id'][x], test_predict.tolist()[x]] for x in range(len(test_ds['book_id']))]

    #Grab feature importances from the individual estimators (100 features with the most positive weights)
    if tfidf:
        feature_importance = getFeaturesForEachBinaryPair(clf, vectorizer.vocabulary_)
    else:
        feature_importance = getFeaturesForEachBinaryPair(clf)
    for est in feature_importance:
        returnable[est] = feature_importance[est]


    filename = "TestResults/COVOC_FullResult/Sniplen_"+str(sniplen)+"_Keylist_"+str(keylist_num)+"_"+keylist_type
    if tfidf:
        filename += "_tfidf.json"
    else:
        filename += "_hpfv.json"
    with open(filename, 'w') as f:
        f.write(json.dumps(returnable))
    
def hyperparamOptimize(sniplen: int, keylist_num: int, num_of_rounds: int, tfidf: bool, keylist_type):
    #Function for coordinating the hyperparameter optimization for each estimator to be used in testing COVOC
    warnings.filterwarnings('ignore') 
    os.environ['PYTHONWARNINGS']='ignore'
    disable_progress_bars()
    if tfidf:
        base_dataset = Dataset.load_from_disk("TCBC_datasets/sniplen"+str(sniplen))
    elif keylist_type == "novels":
        base_dataset = Dataset.load_from_disk("TCBC_datasets/sniplen"+str(sniplen)+"_hpfv_novels")
    else:
        base_dataset = Dataset.load_from_disk("TCBC_datasets/sniplen"+str(sniplen)+"_hpfv")
    results = getOptimHyperparam(base_dataset, keylist_num, num_of_rounds, tfidf, keylist_type)
    filename = "TestResults/COVOC_hyperparams/COVOC_hyperparams_sniplen_"+str(sniplen)+"_keylist_"+str(keylist_num)+"_"+keylist_type
    if tfidf:
        filename += "_tfidf"
    else:
        filename += "_hpfv"
    filename += ".jsonl"
    with open(filename, 'w') as f:
        f.write('\n'.join(map(json.dumps, results)))


def doFullRun(sniplen: int, tfidf: bool, keylist_type):
    warnings.filterwarnings('ignore') 
    os.environ['PYTHONWARNINGS']='ignore'
    disable_progress_bars()
    if tfidf:
        base_dataset = Dataset.load_from_disk("TCBC_datasets/sniplen"+str(sniplen))
    elif keylist_type == "novels":
        base_dataset = Dataset.load_from_disk("TCBC_datasets/sniplen"+str(sniplen)+"_hpfv_novels")
    else:
        base_dataset = Dataset.load_from_disk("TCBC_datasets/sniplen"+str(sniplen)+"_hpfv")
    pbar = tqdm(total=100)
    def update(*a):
     pbar.update(1)
    #For CSC environments
    pool = mp.Pool(len(os.sched_getaffinity(0)))
    #For each keylist
    for i in range(100):
        pool.apply_async(testCOVOC, [base_dataset, sniplen, i, tfidf, keylist_type], callback=update)
        #Manual test case
        #testCOVOC(base_dataset, sniplen, i, tfidf)
    #print("All running!")
    pool.close()
    #print("Pool closed!")
    pool.join()
    #print("Waiting done!")


#Main function
def main(cmd_args):
    #For performing the hyperparam optimization
    #hyperparamOptimize(int(cmd_args[0]), int(cmd_args[1]), int(cmd_args[2]), bool(int(cmd_args[3])))
    #For performing the tests
    doFullRun(int(cmd_args[0]), bool(int(cmd_args[1])))
    #Manual testing
    #doFullRun(100, 1)
    
#Pass cmd args to main function
if __name__ == "__main__":
    main(sys.argv[1:])

