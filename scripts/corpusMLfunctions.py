#Functions for prepping corpus data for ML purposes
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, logging
from scripts import bookdatafunctions as bdf
import numpy as np
import random
import math
from tqdm import tqdm
import os
import json

#Constants
#In the deprel column
#Dependency relation types in Finnish UD
DEPRELS = ['root', 'nsubj', 'advmod', 'obl', 'obj', 'conj', 'aux', 'cc', 'amod', 'nmod:poss', 'mark', 'cop', 'nsubj:cop', 'advcl', 'xcomp', 'case', 'det', 'ccomp', 'nmod', 'parataxis', 'acl:relcl', 'acl', 'xcomp:ds', 'discourse', 'nummod', 'fixed', 'cop:own', 'appos', 'flat:name', 'compound:nn', 'aux:pass', 'vocative', 'nmod:gobj', 'nmod:gsubj', 'compound:prt', 'csubj:cop', 'flat:foreign', 'orphan', 'cc:preconj', 'csubj', 'compound', 'flat', 'goeswith']

#In the feats column
#Cases in Finnihs UD
CASES = ['Case=Nom', 'Case=Gen', 'Case=Par', 'Case=Ill', 'Case=Ine', 'Case=Ela', 'Case=Ade', 'Case=All', 'Case=Ess', 'Case=Abl', 'Case=Tra', 'Case=Acc', 'Case=Ins', 'Case=Abe', 'Case=Com']
#Verb forms in Finnish UD
VERBFORMS = ['VerbForm=Fin', 'VerbForm=Inf', 'VerbForm=Part']
#Verb tenses in Finnish UD
VERBTENSES = ['Tense=Pres', 'Tense=Past']
#Verb voices in Finnish UD
VERBVOICES = ['Voice=Act', 'Voice=Pass']
#Verb moods in Finnish UD
VERBMOODS = ['Mood=Ind', 'Mood=Cnd', 'Mood=Imp']
#Verb 'person' in Finnish UD (aka first person, second person and so on)
PERSONS = ['Person=0', 'Person=1', 'Person=2', 'Person=3']
#Verb 'number' in Finnish UD (aka first singluar person [me] or first plural person [we] and so on)
NUMBERS = ['Number=Sing', 'Number=Plur']
#Connegative (aka verb that has been given a negative meaning by 'ei')
CONNEGATIVE = ['Connegative=Yes']
#Degrees in Finnish UD (positive, comparative, and superlative)
DEGREES = ['Degree=Pos','Degree=Cmp','Degree=Sup']
#Syles in Finnish UD
STYLES = ['Style=Arch', 'Style=Coll']
#Reflex pronouns in Finnish UD
REFS = ['Reflex=Yes']
#PronTypes in Finnish UD
PRONTYPES = ['PronType=Dem', 'PronType=Ind', 'PronType=Int', 'PronType=Prs', 'PronType=Rcp', 'PronType=Rel']
#Verb polarity in Finnish UD
POLARITY = ['Polarity=Neg']
#Person possessor in Finnish UD (e.g. luu VS. luumme)
PPSORS = ['Person[psor]=1', 'Person[psor]=2', 'Person[psor]=3']
#Partforms in Finnish UD
PARTFORMS = ['PartForm=Agt', 'PartForm=Neg', 'PartForm=Past', 'PartForm=Pres']
#Number types in Finnish UD
NUMTYPES = ['NumType=Card', 'NumType=Ord']
#Numeral posessor in Finnish UD (e.g. aikani VS. aikanamme)
NPSORS = ['Number[psor]=Plur', 'Number[psor]=Sing']
#Infinitive forms for verbs in Finnish UD
INFFORMS = ['InfForm=1', 'InfForm=2', 'InfForm=3']
#Marks foreign words in Finnish UD
FOREIGN = ['Foreign=Yes']
#Derivations of words in Finnish UD
DERIVATIONS = ['Derivation=Inen', 'Derivation=Ja', 'Derivation=Lainen', 'Derivation=Llinen', 'Derivation=Tar', 'Derivation=Ton', 'Derivation=Ttain', 'Derivation=U', 'Derivation=Vs', 'Derivation=Inen|Vs' 'Derivation=Ja|Tar', 'Derivation=Lainen|Vs', 'Derivation=Llinen|Vs', 'Derivation=Ton|Vs']
#Clitics of words in Finnish UD
CLITICS = ['Cilitic=Han', 'Cilitic=Ka', 'Cilitic=Kaan', 'Cilitic=Kin', 'Cilitic=Ko', 'Cilitic=Pa', 'Cilitic=S', 'Cilitic=Han|Kin', 'Cilitic=Han|Ko', 'Cilitic=Han|Pa', 'Cilitic=Ko|S', 'Cilitic=Pa|S']
#AdpTypes in Finnish UD
ADPTYPES = ['AdpType=Post', 'AdpType=Prep']
#Marks if words are abbrevations in Finnish UD
ABBR = ['Abbr=Yes']

FEATS = CASES + VERBFORMS + VERBTENSES + VERBVOICES + VERBMOODS + PERSONS + NUMBERS + CONNEGATIVE + DEGREES + STYLES + REFS + PRONTYPES + POLARITY + PPSORS + PARTFORMS + NUMTYPES + NPSORS + INFFORMS + FOREIGN + DERIVATIONS + CLITICS + ADPTYPES + ABBR
#In the upos column
#POS tags in Finnish UD
POS = ['NOUN', 'VERB', 'PRON', 'ADV', 'AUX', 'ADJ', 'PROPN', 'CCONJ', 'SCONJ', 'ADP', 'NUM', 'INTJ', 'PUNCT']

#Bigrams and trigrams
POS_BIGRAMS = []
POS_TRIGRAMS = []
for x in POS:
    for y in POS:
            for z in POS:
                  POS_TRIGRAMS.append((x,y,z))
            POS_BIGRAMS.append((x,y))

#Can be used if we want to look at all possible deprel bigrams and trigrams (not too computationally expensive, but a bit slow and can cause memory issues for a normal working laptop)
"""
DEPREL_BIGRAMS = []
DEPREL_TRIGRAMS = []
for x in DEPRELS:
    for y in DEPRELS:
            for z in DEPRELS:
                  DEPREL_TRIGRAMS.append((x,y,z))
            DEPREL_BIGRAMS.append((x,y))
"""

#NOT IN TCBC 1.0
#Uneccessary features and bi/trigrams, which don't occur even once in any of the books
#We remove them to save time when creating feature vectors
#THIS SHOULD BE UPDATED WITH NEW VERSIONS OF THE CORPUS

FEATS_TO_DEL = ['Derivation=Inen|VsDerivation=Ja|Tar', 'Derivation=Lainen|Vs', 'Derivation=Llinen|Vs', 'Derivation=Ton|Vs', 'Cilitic=Han', 'Cilitic=Ka', 'Cilitic=Kaan', 'Cilitic=Kin', 'Cilitic=Ko', 'Cilitic=Pa', 'Cilitic=S', 'Cilitic=Han|Kin', 'Cilitic=Han|Ko', 'Cilitic=Han|Pa', 'Cilitic=Ko|S', 'Cilitic=Pa|S']
POS_TRIGRAMS_TO_DEL = [('NOUN', 'INTJ', 'ADP'), ('VERB', 'ADP', 'SCONJ'), ('VERB', 'ADP', 'INTJ'), ('VERB', 'NUM', 'INTJ'), ('VERB', 'INTJ', 'ADP'), ('PRON', 'INTJ', 'CCONJ'), ('PRON', 'INTJ', 'NUM'), ('ADV', 'NUM', 'INTJ'), ('ADV', 'INTJ', 'ADP'), ('AUX', 'CCONJ', 'ADP'), ('AUX', 'SCONJ', 'CCONJ'), ('AUX', 'SCONJ', 'ADP'), ('AUX', 'SCONJ', 'INTJ'), ('AUX', 'ADP', 'INTJ'), ('AUX', 'INTJ', 'CCONJ'), ('AUX', 'INTJ', 'ADP'), ('ADJ', 'PRON', 'INTJ'), ('ADJ', 'SCONJ', 'CCONJ'), ('ADJ', 'ADP', 'INTJ'), ('ADJ', 'NUM', 'INTJ'), ('ADJ', 'INTJ', 'PRON'), ('ADJ', 'INTJ', 'AUX'), ('ADJ', 'INTJ', 'ADP'), ('ADJ', 'INTJ', 'NUM'), ('PROPN', 'ADJ', 'INTJ'), ('PROPN', 'SCONJ', 'CCONJ'), ('PROPN', 'SCONJ', 'ADP'), ('PROPN', 'SCONJ', 'INTJ'), ('PROPN', 'ADP', 'INTJ'), ('PROPN', 'INTJ', 'CCONJ'), ('PROPN', 'INTJ', 'SCONJ'), ('PROPN', 'INTJ', 'ADP'), ('PROPN', 'INTJ', 'NUM'), ('CCONJ', 'CCONJ', 'ADP'), ('CCONJ', 'CCONJ', 'NUM'), ('CCONJ', 'ADP', 'SCONJ'), ('CCONJ', 'ADP', 'INTJ'), ('CCONJ', 'NUM', 'SCONJ'), ('CCONJ', 'NUM', 'INTJ'), ('CCONJ', 'INTJ', 'ADP'), ('SCONJ', 'PROPN', 'INTJ'), ('SCONJ', 'CCONJ', 'AUX'), ('SCONJ', 'CCONJ', 'ADP'), ('SCONJ', 'CCONJ', 'NUM'), ('SCONJ', 'CCONJ', 'INTJ'), ('SCONJ', 'CCONJ', 'PUNCT'), ('SCONJ', 'SCONJ', 'ADP'), ('SCONJ', 'SCONJ', 'INTJ'), ('SCONJ', 'ADP', 'AUX'), ('SCONJ', 'ADP', 'CCONJ'), ('SCONJ', 'ADP', 'SCONJ'), ('SCONJ', 'ADP', 'ADP'), ('SCONJ', 'ADP', 'INTJ'), ('SCONJ', 'ADP', 'PUNCT'), ('SCONJ', 'NUM', 'SCONJ'), ('SCONJ', 'NUM', 'INTJ'), ('SCONJ', 'INTJ', 'ADJ'), ('SCONJ', 'INTJ', 'CCONJ'), ('SCONJ', 'INTJ', 'ADP'), ('SCONJ', 'INTJ', 'NUM'), ('ADP', 'VERB', 'INTJ'), ('ADP', 'ADV', 'INTJ'), ('ADP', 'ADJ', 'INTJ'), ('ADP', 'PROPN', 'INTJ'), ('ADP', 'SCONJ', 'CCONJ'), ('ADP', 'ADP', 'SCONJ'), ('ADP', 'ADP', 'INTJ'), ('ADP', 'NUM', 'SCONJ'), ('ADP', 'NUM', 'INTJ'), ('ADP', 'INTJ', 'PRON'), ('ADP', 'INTJ', 'ADV'), ('ADP', 'INTJ', 'AUX'), ('ADP', 'INTJ', 'ADJ'), ('ADP', 'INTJ', 'CCONJ'), ('ADP', 'INTJ', 'SCONJ'), ('ADP', 'INTJ', 'ADP'), ('ADP', 'INTJ', 'NUM'), ('NUM', 'PRON', 'INTJ'), ('NUM', 'ADV', 'INTJ'), ('NUM', 'AUX', 'CCONJ'), ('NUM', 'AUX', 'INTJ'), ('NUM', 'ADJ', 'INTJ'), ('NUM', 'CCONJ', 'ADP'), ('NUM', 'SCONJ', 'ADV'), ('NUM', 'SCONJ', 'CCONJ'), ('NUM', 'SCONJ', 'SCONJ'), ('NUM', 'SCONJ', 'ADP'), ('NUM', 'SCONJ', 'INTJ'), ('NUM', 'SCONJ', 'PUNCT'), ('NUM', 'ADP', 'ADP'), ('NUM', 'ADP', 'INTJ'), ('NUM', 'INTJ', 'VERB'), ('NUM', 'INTJ', 'AUX'), ('NUM', 'INTJ', 'ADP'), ('NUM', 'INTJ', 'INTJ'), ('INTJ', 'VERB', 'ADP'), ('INTJ', 'ADV', 'ADP'), ('INTJ', 'AUX', 'CCONJ'), ('INTJ', 'AUX', 'SCONJ'), ('INTJ', 'AUX', 'ADP'), ('INTJ', 'AUX', 'NUM'), ('INTJ', 'ADJ', 'ADP'), ('INTJ', 'ADJ', 'NUM'), ('INTJ', 'PROPN', 'SCONJ'), ('INTJ', 'CCONJ', 'CCONJ'), ('INTJ', 'CCONJ', 'ADP'), ('INTJ', 'CCONJ', 'NUM'), ('INTJ', 'SCONJ', 'CCONJ'), ('INTJ', 'SCONJ', 'ADP'), ('INTJ', 'SCONJ', 'NUM'), ('INTJ', 'SCONJ', 'INTJ'), ('INTJ', 'ADP', 'VERB'), ('INTJ', 'ADP', 'AUX'), ('INTJ', 'ADP', 'ADJ'), ('INTJ', 'ADP', 'CCONJ'), ('INTJ', 'ADP', 'SCONJ'), ('INTJ', 'ADP', 'ADP'), ('INTJ', 'ADP', 'NUM'), ('INTJ', 'ADP', 'INTJ'), ('INTJ', 'NUM', 'PRON'), ('INTJ', 'NUM', 'AUX'), ('INTJ', 'NUM', 'PROPN'), ('INTJ', 'NUM', 'CCONJ'), ('INTJ', 'NUM', 'SCONJ'), ('INTJ', 'NUM', 'ADP'), ('INTJ', 'NUM', 'NUM'), ('INTJ', 'NUM', 'INTJ'), ('INTJ', 'INTJ', 'ADP'), ('PUNCT', 'ADP', 'SCONJ'), ('PUNCT', 'ADP', 'INTJ')]

FEATS = [x for x in FEATS if x not in FEATS_TO_DEL]
POS_TRIGRAMS = [x for x in POS_TRIGRAMS if x not in POS_TRIGRAMS_TO_DEL]

#These lists can be generated by generating feature vectors for all books in TCBC and running the following code on the feature vectors:
"""
def zeroedFeatures(arr):
    zeroed_indices = []
    res = np.count_nonzero(arr, axis=0)
    for i in index_dict:
        if res[i] == 0:
            zeroed_indices.append(i)
    return zeroed_indices

to_del = zeroedFeatures(np.array([x[1] for x in results]))

def ngramToTuple(ngram_str):
    return tuple(ngram_str.split('_'))

to_del_feats = []
to_del_pos_bigram = []
to_del_pos_trigram = []
to_del_deprel_bigram = []
to_del_deprel_trigram = []
for i in to_del:
    if index_dict[i] in FEATS:
        to_del_feats.append(index_dict[i])
    test = ngramToTuple(index_dict[i])
    if test in POS_BIGRAMS:
        to_del_pos_bigram.append(test)
    if test in POS_TRIGRAMS:
        to_del_pos_trigram.append(test)
    if test in DEPREL_BIGRAMS:
        to_del_deprel_bigram.append(test)
    if test in DEPREL_TRIGRAMS:
        to_del_deprel_trigram.append(test)
"""

#Combine UD features together
CONLLU_FEATS = DEPRELS + FEATS + POS 

def splitKeysTrainTestEval(corpus_with_ages: dict[str, pd.DataFrame]):
    """
    Splits the corpus' keys into train, eval, and test datasets, where the dataset contains:
    key - identifier of the book
    age - exact target age of the book in question
    Returned is a tuple with three datasets, in the order of: (train, test, eval)
    """

    temp_for_ds_building = []
    for i in corpus_with_ages:
            temp_for_ds_building.append({'key':i, 'age':bdf.findAgeFromID(i)})

    temp_ds = Dataset.from_list(temp_for_ds_building)
    temp_ds = temp_ds.class_encode_column("age")
    train_test_keys = temp_ds.train_test_split(test_size=0.3, shuffle=True, stratify_by_column='age')
    test_eval_keys = train_test_keys['test'].train_test_split(test_size=0.5, shuffle=True, stratify_by_column='age')

    train_keys_ds = train_test_keys['train']
    test_keys_ds = test_eval_keys['train']
    eval_keys_ds = test_eval_keys['test']
    return (train_keys_ds, test_keys_ds, eval_keys_ds)

def splitCorpusTrainTestEval(corpus, train_keys: list[str], test_keys: list[str], eval_keys: list[str]) -> tuple[dict,dict,dict]:
    """
    Function that splits the corpus into three 'subcorpora', for NLP purposes
    Return tuple of dictionaries in the order: (train, test, eval)
    """
    train_corp = { k: corpus[k] for k in train_keys }
    test_corp = { k: corpus[k] for k in test_keys }
    eval_corp = { k: corpus[k] for k in eval_keys }

    return (train_corp, test_corp, eval_corp)

def prepBooksForVectorizer(columns: list[str], corpus: dict[str,pd.DataFrame]) -> dict[str,str]:
    """
    Function which takes in a list of columns to be included in fitting a classifier and joins all its members together
    Returns a dictionary with ids as keys and raw strings with the wanted features separated by whitespaces
    """
    returnable = {}
    for book in corpus:
        df = corpus[book]
        temp = ""
        for column in columns:
            temp_list = df[column].values.astype(str)
            if column == 'feats':
                temp_list = [x.split('|') for x in temp_list]
                temp_list = [x for xs in temp_list for x in xs]
            #If for some reason we get a nested list with singluar items instead of a flat list...
            if type(temp_list[0]) == np.ndarray:
                flatList = list(np.array(df[column].values.astype(str)).flat)
            else:
                flatList = temp_list
            temp += " ".join(flatList)
        temp = temp.lower()
        returnable[book] = temp
    return returnable

def getVocabularyFromPreppedCorpus(prepped_corpus: dict[str,str]):
    lists = [i.split(" ") for i in list(prepped_corpus.values())]
    flatList = [x for xs in lists for x in xs]
    temp_ser = pd.Series(flatList).apply(lambda x: str(x).lower()).drop_duplicates()
    temp_list = temp_ser.to_numpy(dtype=str).tolist()
    
    return 

def vectorizeCorpus(corpus: dict[str, str], vectorizer) -> dict:
    """
    Function which vectorizes the books in a corpus with some given vectorizer (tested with SKLearn's TfIdfVectorizer
    Returns a dict where keys are the ids of books and values are the vectorized results
    """
    returnable = {}
    
    vectorized_books = vectorizer.fit_transform(corpus.values())
    keys = list(corpus.keys())
    for i in range(len(keys)):
        key = keys[i]
        returnable[key] = vectorized_books[i]
    return returnable


def buildDatasetDictFromCorpus(preppedCorpus: dict[str,list[float]], train_keys: list[str], test_keys: list[str], exact_age_labels=False) -> DatasetDict:
    """ 
    Function which takes in a corpus that's been prepped by prepBooksForVectorizer, and sets of keys with a train-test split
    Returns a HuggingFace DatasetDict with ['train'] and ['test'] sets
    """
    train_ds = []
    test_ds = []
    #Add training data
    for key in train_keys:
        if exact_age_labels:
            label = int(bdf.findAgeFromID(key))
        else:
            label = int(str(key)[14])-1
        data = preppedCorpus[key]
        train_ds.append({"data":data,"label":label})
    #Add test data
    for key in test_keys:
        if exact_age_labels:
            label = int(bdf.findAgeFromID(key))
        else:
            label = int(str(key)[14])-1
        data = preppedCorpus[key]
        test_ds.append({"data":data,"label":label})
    return DatasetDict({"train":Dataset.from_list(train_ds), "test":Dataset.from_list(test_ds)})   

def buildManualDatasetFromKeys(keys_ds : Dataset, vectorized_corpus: dict, exact_age_labels=True) -> tuple[dict, dict]:
    """ 
    Function which takes in a corpus that's been prepped by prepBooksForVectorizer, a set of keys (e.g. keys dedicated to the training set) 
    and returns a tuple which has two dicts: First one contains the sparse matrix that we got from the vectorizer and the second one the age corrseponding age labels. 
    This function exists, because datasets does not like using sparse matrices as data, so we achieve the same result 'manually'. 
    Obviously not as neat and requires a bit more coding to use, but it works :)
    """
    data_dict = {}
    label_dict = {}
    #Add data and labels to corresponding dicts
    for key in keys_ds['key']:
        if exact_age_labels:
            label = int(bdf.findAgeFromID(key))
        else:
            label = int(str(key)[14])-1
        data = vectorized_corpus[key]
        data_dict[key] = data
        label_dict[key] = label
    return (data_dict, label_dict)

def splitOnAuthorLevel(corpus_keys: list[str], sheet_path: str) -> dict[str,list[str]]:
    """
    Function for distributing train/test-eval keys so that books from the same author only end up in one of the two splits
    """

    books_per_author = {}
    temp_df = pd.read_excel(sheet_path).fillna('Unknown')
    isbn2auth_series = pd.Series(temp_df[temp_df.columns[1]].values, index=temp_df[temp_df.columns[0]].values)
    for key in corpus_keys:
        author = str(isbn2auth_series.at[int(key[:13])])
        if not author in list(books_per_author.keys()):
            books_per_author[author] = [key]
        else:
            temp_list = books_per_author[author]
            temp_list.append(key)
            books_per_author[author] = temp_list
    return books_per_author

def generateAgeStratificationAmounts(corpus_with_ages: dict[str,pd.DataFrame], train_size: float) -> tuple[dict[int,int],dict[int,int],dict[int,int]]:
    train = {}
    test = {}
    eval = {}
    ages = bdf.getAvailableAges(corpus_with_ages)
    for age in ages:
        raw_amount = len([x for x in list(corpus_with_ages.keys()) if bdf.findAgeFromID(x)==str(age)])
        train[age] = int(raw_amount*train_size)
        test[age] = int((raw_amount-train[age])/2)
        eval[age] = int((raw_amount-train[age])/2)
    return train, test, eval

def getNumOfEntriesPerAge(keys: list[str]):
    ages_to_add = [int(bdf.findAgeFromID(x)) for x in keys]
    return {x : ages_to_add.count(x) for x in list(set(ages_to_add))}

def doTrainTestEvalSplitSeriesLevel(author_level_split: dict[str,list[str]], train_target_amounts: dict[int,int], test_target_amounts: dict[int,int], eval_target_amounts: dict[int,int]):
    
    """
    Function which splits a corpus into (roughly) stratified datasets for training, evaluation, and testing
    """
    train_keys = []
    test_keys = []
    eval_keys = []
    authors = list(author_level_split.keys())
    #Shuffle the authors so we don't always end up with the same sets
    random.shuffle(authors)
    for author in authors:
        keys_to_add = author_level_split[author]
        #Get dicts for age:number of entries
        to_add = getNumOfEntriesPerAge(keys_to_add)
        current_train = getNumOfEntriesPerAge(train_keys)
        current_test = getNumOfEntriesPerAge(test_keys)
        current_eval = getNumOfEntriesPerAge(eval_keys)
        #Use a flag to determine whether to return true or false
        toset = 'train'
        for age in to_add:
            #If age not yet present in the train set, then immediately return True as we want to add the batch to Train 
            if not age in list(current_train.keys()):
                toset = 'train'
                break
            #If the target age has not yet been met, then continue to check rest of the ages before making any conclusions
            if current_train[age] < train_target_amounts[age]:
                continue
            #If the target amount has been reached in Train, then if an age is not present in Eval, immediately add the batch there
            if not age in list(current_eval.keys()):
                toset = 'eval'
                break
            #If the target amount has been reached in Train and Eval, then if an age is not present in Test, immediately add the batch there
            if not age in list(current_test.keys()):
                toset = 'test'
                break
            #If all previous checks pass and Eval is underpopulated, then tentatively add the batch to TE, but see the remaining ages if stronger conditions are met
            if current_eval[age] < eval_target_amounts[age]:
                toset = 'eval'
            #If all previous checks pass and Test is underpopulated, then tentatively add the batch to TE, but see the remaining ages if stronger conditions are met
            elif current_test[age] < test_target_amounts[age]:
                toset = 'test'
            #If all else passes, then just tentatively add the batch to train
            else:
                toset = 'train'
        
        if toset == 'train':
            train_keys += keys_to_add
        elif toset == 'eval':
            eval_keys += keys_to_add
        else:
            test_keys += keys_to_add
    return train_keys, test_keys, eval_keys

def simpleFeatureVectorizer(corpus: dict[str,pd.DataFrame], deprels: list[str], feats: list[str], pos: list[str], other_features: bool=False) -> tuple[dict[str,list[float]],dict[str,list[int]]]:
    """
    Simple custom vectorizer for FCBLex that uses proportions of simple features appearing in different columns of Trankit output
    Returns a dictionary which contains corpus keys mapped with the feature vectors and a dictionary, which maps feature names to their index in the feature vectors
    """
    featurized_books = {}
    feature_indices = {}
    feat_dicts = []
    index = 0
    for deprel in deprels:
        feat_dicts.append(bdf.getDeprelFeaturePerBook(corpus, deprel, True))
        feature_indices[index] = deprel
        index += 1
    for feat in feats:
        feat_dicts.append(bdf.getFeatsFeaturePerBook(corpus, feat, True))
        feature_indices[index] = feat
        index += 1
    for p in pos:
        feat_dicts.append(bdf.getPosFeaturePerBook(corpus, p, True))
        feature_indices[index] = p
        index += 1
    if other_features:
        #Other features
        word_freqs = bdf.getWordFrequencies(corpus)
        word_amounts = bdf.getTokenAmounts(corpus)
        sent_amounts = bdf.getNumOfSentences(corpus)

        feat_dicts.append(bdf.getTypeTokenRatios(word_freqs, word_amounts).to_dict())
        feature_indices[index] = 'TTR'
        index += 1
        feat_dicts.append(bdf.scaleCorpusData(bdf.getPosPhraseCounts(corpus, 'NOUN'), sent_amounts))
        feature_indices[index] = 'NPS'
        index += 1
        feat_dicts.append(bdf.scaleCorpusData(bdf.getPosPhraseCounts(corpus, 'VERB'), sent_amounts))
        feature_indices[index] = 'VPS'
        index += 1
        feat_dicts.append(bdf.scaleCorpusData(bdf.getPosPhraseCounts(corpus, 'ADJ'), sent_amounts))
        feature_indices[index] = 'ADJPS'
        index += 1
        feat_dicts.append(bdf.scaleCorpusData(bdf.getPosPhraseCounts(corpus, 'ADV'), sent_amounts))
        feature_indices[index] = 'ADVPS'
        index += 1

    for key in corpus:
        feature_vector = []
        for ind in feature_indices:
            feature_vector.append(feat_dicts[ind][key])
        #Normalize the feature vectors using l2
        l2 = math.sqrt(sum([x**2 for x in feature_vector]))
        if not l2 == 0:
            feature_vector = [x/l2 for x in feature_vector]
        featurized_books[key] = feature_vector
    
    return featurized_books, feature_indices


#Split the books into snippets
def splitBooksToSnippetsDataset(corpus: dict[str,pd.DataFrame], keys: list[str], snipLength: int, columns: list[str], whitelist: list[str]=None) -> Dataset:
    dict_list = []
    for key in keys:
        df = corpus[key]
        sent_counter = 0
        text = ""
        label = bdf.findAgeFromID(key)
        for i in range(len(df)):
            if str(df['id'].iloc[i]) == '1':
                sent_counter += 1
            if sent_counter == snipLength-1:
                if whitelist:
                    temp_list = text.split(" ")
                    text = " ".join([x for x in temp_list if x in whitelist])
                dict_list.append({"data":text,"label":label, "book_id":key})
                sent_counter = 0
                text = ""
            for column in columns:
                if column=='feats':
                    text += str(df[column].iloc[i]).replace('|'," ")+" "
                else:
                    text += str(df[column].iloc[i])+" "
        #Add in the remaining few sentences
        if sent_counter != 0:
            if whitelist:
                temp_list = text.split(" ")
                text = " ".join([x for x in temp_list if x in whitelist])
            dict_list.append({"data":text,"label":label, "book_id":key})
    
    return Dataset.from_list(dict_list)


def writeKeysToFile(keys: list[str], name: str):
    """
    Save list of keys to a file
    """
    keys = [str(x)+'\n' for x in keys]
    with open(name+"_keys.txt", 'w') as writer:
        writer.writelines(keys)

def readKeysToMemory(file_path: str) -> list[str]:
    """
    Read a list of keys from file to memory
    """
    with open(file_path, 'r') as reader:
        keys = reader.readlines()
    keys = [x.replace('\n','') for x in keys]
    return keys[:len(keys)-1]

def map2Group(i: int):
    if i < 9:
        return "7-8"
    elif i < 13:
        return "9-12"
    return "13+"

def replaceWithMin(list1, list2):
    new_list = []
    for i in range(len(list1)):
        if list1[i] < list2[i]:
            new_list.append(list1[i])
        else:
            new_list.append(list2[i])
    return new_list

def replaceWithMax(list1, list2):
    new_list = []
    for i in range(len(list1)):
        if list1[i] > list2[i]:
            new_list.append(list1[i])
        else:
            new_list.append(list2[i])
    return new_list

def snippetConllu2DF(conllu_lines: str):
    df = pd.DataFrame([line.split('\t') for line in conllu_lines.split('\n')])
    df.columns = ['id', 'text', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']
    
    return df.dropna()

def getFleschKincaidGradeLevel(corpus: dict):
    returnable = {}
    ASL = bdf.getAvgSentenceLens(corpus)
    for id in corpus:
        df = corpus[id]
        ASW = np.mean(pd.Series(data=df['text'].apply(bdf.countSyllablesFinnish).to_numpy(), index=df['text'].to_numpy(dtype='str')).to_numpy(na_value=0))
        returnable[id] = 0.39*ASL[id] + 11.8*ASW - 15.59
    
    return returnable

#Alternative way of using the custom vectorizer to enable min-max normalization of the data

def customConlluVectorizer(df: pd.DataFrame, generate_key_dictionary:bool=False):
    """
    Custom vecotrizer used to create feature vectors from hand-picked features.
    If passed a flag, will generate a dictionary that maps feature vector indices to the names of the features.
    Recommend only generating to key dictionary once to save some time, allthough it is not computationally super expensive.
    """
    feature_vector = []
    if generate_key_dictionary:
        feature_indices = {}
        index = 0
    temp_corp = {'1':df}
    syntactic_tree = bdf.buildIdTreeFromConllu(df)
    word_freqs = bdf.getWordFrequencies(temp_corp)
    word_amounts = bdf.getTokenAmounts(temp_corp)
    sent_amounts = bdf.getNumOfSentences(temp_corp)

    #Deprel per main clause for each deprel
    for deprel in DEPRELS:
        feature_vector.append(bdf.getDeprelFeaturePerBook(temp_corp, deprel, True)['1'])
        if generate_key_dictionary:
            feature_indices[index] = deprel
            index += 1
    #Feat per main clause for each feature
    for feat in FEATS:
        feature_vector.append(bdf.getFeatsFeaturePerBook(temp_corp, feat, True)['1'])
        if generate_key_dictionary:
            feature_indices[index] = feat
            index += 1
    #POS related (simple) features
    for pos in POS:
        #POS per main clause for each pos-tag
        feature_vector.append(bdf.getPosFeaturePerBook(temp_corp, pos, True)['1'])
        if generate_key_dictionary:
            feature_indices[index] = pos
            index += 1
        #POS pharses per main clause
        feature_vector.append(bdf.scaleCorpusData(bdf.getPosPhraseCounts(temp_corp, pos), sent_amounts)['1'])
        if generate_key_dictionary:
            feature_indices[index] = pos+"_Phrase"
            index += 1
        #POS variation
        feature_vector.append(bdf.getPOSVariation(df, pos))
        if generate_key_dictionary:
            feature_indices[index] = pos+"_Variation"
            index += 1
        #Corrected POS variation
        feature_vector.append(bdf.getCorrectedPOSVariation(df, pos))
        if generate_key_dictionary:
            feature_indices[index] = pos+"_Variation_Corrected"
            index += 1
        #POS ratios
        pos2 = POS.copy()
        pos2.remove(pos)
        for pos_2 in pos2:
            #Check that we don't divide by 0 by accident!
            divider = bdf.getPosFeaturePerBook(temp_corp, pos_2)['1']
            if divider == 0:
                feature_vector.append(0)
            else:
                feature_vector.append(bdf.getPosFeaturePerBook(temp_corp, pos)['1'] / divider)
            if generate_key_dictionary:
                feature_indices[index] = pos+"_To_"+pos_2+"_Ratio"
                index += 1
    """
    #Flat POS bigrams per main clause
    pos_bigrams = bdf.getPosNGramForCorpus(temp_corp, 2)['1']
    for pb in POS_BIGRAMS:
         feature_vector.append(pos_bigrams.get(pb, 0) / sent_amounts['1'])
         if generate_key_dictionary:
              feature_indices[index] = 'flat_'+pb[0]+'_'+pb[1]
              index += 1
    #Flat POS trigrams per main clause
    pos_trigrams = bdf.getPosNGramForCorpus(temp_corp, 3)['1']
    for pt in POS_TRIGRAMS:
         feature_vector.append(pos_trigrams.get(pt, 0) / sent_amounts['1'])
         if generate_key_dictionary:
              feature_indices[index] = 'flat_'+pt[0]+'_'+pt[1]+'_'+pt[2]
              index += 1
    #Tree POS bigrams per main clause
    pos_bigrams = bdf.getSyntacticTreeNGram(df, syntactic_tree, 2, 'upos')
    for pb in POS_BIGRAMS:
         feature_vector.append(pos_bigrams.get(pb, 0) / sent_amounts['1'])
         if generate_key_dictionary:
              feature_indices[index] = 'tree_'+pb[0]+'_'+pb[1]
              index += 1
    #Tree POS trigrams per main clause
    pos_trigrams = bdf.getSyntacticTreeNGram(df, syntactic_tree, 3, 'upos')
    for pt in POS_TRIGRAMS:
         feature_vector.append(pos_trigrams.get(pt, 0) / sent_amounts['1'])
         if generate_key_dictionary:
              feature_indices[index] = 'tree_'+pt[0]+'_'+pt[1]+'_'+pt[2]
              index += 1
    """

        

    #Other features
    #TTR
    feature_vector.append(bdf.getTypeTokenRatios(word_freqs, word_amounts)['1'])
    if generate_key_dictionary:
            feature_indices[index] = "TTR"
            index += 1
    #MLS
    feature_vector.append(bdf.getAvgSentenceLens(temp_corp)['1'])
    if generate_key_dictionary:
            feature_indices[index] = "MLS"
            index += 1
    #Average number of syllables per sentence
    feature_vector.append(bdf.getAverageSyllablesPerSentence(df, syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "AvgSylPerSent"
            index += 1
    #CONJ2Sent
    feature_vector.append(bdf.getConjPerSentence(temp_corp)['1'])
    if generate_key_dictionary:
            feature_indices[index] = "ConjPerSent"
            index += 1
    #Flesch-Kincaid grade level
    feature_vector.append(bdf.getFleschKincaidGradeLevel(temp_corp)['1'])
    if generate_key_dictionary:
            feature_indices[index] = "F-K-GradeLevel"
            index += 1
    #Get modified SMOG Index
    feature_vector.append(bdf.getModifiedSmogIndexForFinnish(df, syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "SMOG"
            index += 1
    #Get Coleman-Liau index
    feature_vector.append(bdf.getColemanLiauIndex(df, syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "ColemanLiau"
            index += 1
    #Get Automated readability index
    feature_vector.append(bdf.getARI(df, syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "ARI"
            index += 1

    #Preposing adverbial clauses
    feature_vector.append(bdf.getPreposingAdverbialClauses(temp_corp)['1'])
    if generate_key_dictionary:
            feature_indices[index] = "PrepAdvcl"
            index += 1
    #Ratio between function words and content words
    feature_vector.append(bdf.getRatioOfFunctionWords(df))
    if generate_key_dictionary:
            feature_indices[index] = "Func2ContWordRatio"
            index += 1
    #Features that require parsing the syntactic tree structure
    #Average depth of syntactic tree
    feature_vector.append(bdf.getMeanSyntacticTreeDepth(syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "MeanTreeDepth"
            index += 1
    #Maximum depth of syntactic tree
    feature_vector.append(bdf.getMaxSyntacticTreeDepth(syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "MaxTreeDepth"
            index += 1
    #Nesting of clauses
    feature_vector.append(len(bdf.findNestingSubclauses(df, syntactic_tree)))
    if generate_key_dictionary:
            feature_indices[index] = "NestingOfClauses"
            index += 1
    #Stacking of clauses
    feature_vector.append(len(bdf.findStackingClauses(df, syntactic_tree)))
    if generate_key_dictionary:
            feature_indices[index] = "StackingOfClauses"
            index += 1
    #Mean length of clauses
    feature_vector.append(bdf.findMeanLengthOfClause(df, syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "MLC"
            index += 1
    #deprel bigrams per main clause
    """
    deprel_bigrams = bdf.getSyntacticTreeNGram(df, syntactic_tree, 2)
    for db in DEPREL_BIGRAMS:
         feature_vector.append(deprel_bigrams.get(db, 0) / sent_amounts['1'])
         if generate_key_dictionary:
              feature_indices[index] = db[0]+'_'+db[1]
              index += 1
    """
    #deprel trigrams per main clause
    """
    deprel_trigrams = bdf.getSyntacticTreeNGram(df, syntactic_tree, 3)
    for dt in DEPREL_TRIGRAMS:
         feature_vector.append(deprel_trigrams.get(dt, 0) / sent_amounts['1'])
         if generate_key_dictionary:
              feature_indices[index] = dt[0]+'_'+dt[1]+'_'+dt[2]
              index += 1
    """
    


    if generate_key_dictionary:
         return feature_vector, feature_indices
    return feature_vector

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

def splitBooksToSnippets(key: str, df: pd.DataFrame, snip_lens: list[int], folder:str=None):
    """
    Function for splitting a book in TCBC into snippet datasets.
    This means taking each book, and splitting it into snippets of x sentences, where x is some integer.
    Needs the key (id) of the book in question and the dataframe that has its conllu data.
    Adjusted to be usable a parallel code.
    Outputs folders for each book in the corpus and creates a (HF) Dataset for each snippet length.
    To combine these individual snippet datasets into a more usable form, please use combineSnippetBooksToDS()
    """
    #Skip if folder already exists (don't do unecessary work twice!)
    #Remember to archive or empty the chosen folder if changing the custom vectorizer function and wanting to re-do the snippet datasets
    if not os.path.exists(folder+key):
        os.mkdir(folder+key)
    placement_folder = folder+key+"/"
    #Init log
    with open(placement_folder+"log.txt", 'w') as log:
         log.write("Starting log\n")
    #Start program
    with open(placement_folder+"log.txt", 'a') as log:
        mins = {x:[] for x in snip_lens}
        maxs = {x:[] for x in snip_lens}
        pre_norming = {}
        base = snip_lens[0]
        dict_lists = {x:[] for x in snip_lens}
        sent_counter = 0
        raw_text = ""
        conllu_format = ""
        sent_counters = {x:0 for x in snip_lens}
        raw_texts = {x:"" for x in snip_lens}
        conllu_formats = {x:"" for x in snip_lens}
        label = int(bdf.findAgeFromID(key))
        group = map2Group(label)
        log.write("Initial setups done!\n")

        #Build syntactic tree for more reliable sentence splitting
        id_tree = bdf.buildIdTreeFromConllu(df)
        log.write("Syntactic tree built!\n")
        sentence_heads = list(id_tree.keys())
        for sent in sentence_heads:
                for j in id_tree[sent]:
                    raw_text += str(df['text'].iloc[j])+" "
                    conllu_format += "\t".join(df.iloc[j].to_numpy("str"))+"\n"
                sent_counter += 1
                if sent_counter == base:
                    for x in snip_lens:
                        raw_texts[x] = raw_texts[x]+raw_text
                        conllu_formats[x] = conllu_formats[x]+conllu_format
                        sent_counters[x] = sent_counters[x]+base
                        if x == sent_counters[x]:
                            dict_lists[x].append({"book_id":key, "age":label, "label":group, "raw_text":raw_texts[x], "conllu":conllu_formats[x]})
                            raw_texts[x] = ""
                            conllu_formats[x] = ""
                            sent_counters[x] = 0
                    sent_counter = 0
                    raw_text = ""
                    conllu_format = ""
        if sent_counter!=0:
            for x in snip_lens:
                raw_texts[x] = raw_texts[x]+raw_text
                conllu_formats[x] = conllu_formats[x]+conllu_format
                dict_lists[x].append({"book_id":key, "age":label, "label":group, "raw_text":raw_texts[x], "conllu":conllu_formats[x]})
        log.write("Dataset creations done properly!\n")
        #Add initial feature vectors for hand picked features and keep track of mins and maxs
        for d in dict_lists:
            for i in range(len(dict_lists[d])):
                hp_fv = customConlluVectorizer(snippetConllu2DF(dict_lists[d][i]['conllu']))
                if len(mins[d]) == 0:
                    mins[d] = hp_fv
                else:
                    mins[d] = replaceWithMin(mins[d], hp_fv)
                if len(maxs[d]) == 0:
                    maxs[d] = hp_fv
                else:
                    maxs[d] = replaceWithMax(maxs[d], hp_fv)
                dict_lists[d][i]['hp_fv'] = hp_fv
            log.write("Feature vector initialized for sniplen "+str(d))
        pre_norming[key] = dict_lists
        log.write("Feature vectors initialized correctly!\n")
        #Scale hand-picked feature vectors and write snippet datasets
        dict_lists = pre_norming[key]
        for d in dict_lists:
            for i in range(len(dict_lists[d])):
                dict_lists[d][i]['hp_fv'] = minMaxNormalization(mins[d], maxs[d], dict_lists[d][i]['hp_fv'])
                log.write("Min-max normalization for sniplen "+str(d)+" done successfully")
            Dataset.from_list(dict_lists[d]).to_json(placement_folder+"sniplen_"+str(d)+".jsonl")
                 
def combineSnippedBooksToDS(keys: list[str], snip_len: str, cache_dir: str, cache_file:str, inc_raw_text: bool=False, inc_conllu: bool=False, inc_hpfv: bool=False, folder:str=None):
    #logging.set_verbosity(40)
    #Helper function to parse json-lines
    def jsonlReader(key: str):
        with open(folder+key+"/sniplen_"+snip_len+".jsonl") as reader:
            with open(cache_file, 'a') as tt:
                 #Only include the information we need for our specific purposes to save up on cache space
                 for line in reader:
                        if not inc_raw_text:
                            line = line[:line.find(",\"raw_text\":")] + line[line.find(",\"conllu\":"):]
                        if not inc_conllu:
                            line = line[:line.find(",\"conllu\":")] + line[line.find(",\"hp_fv\":"):]
                        if not inc_hpfv:
                            line = line[:line.find(",\"hp_fv\":")] + line[line.find("}\n"):]
                        tt.write(line)
    #Generate list of dicts, where each dict is a json-line
    for k in range(len(keys)):
        jsonlReader(keys[k])
    #Return a shuffled dataset
    return Dataset.from_json(cache_file, cache_dir=cache_dir).shuffle()