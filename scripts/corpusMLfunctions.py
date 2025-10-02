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
        train[age] = math.ceil(int(raw_amount*train_size))
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



def getFleschKincaidGradeLevel(corpus: dict):
    returnable = {}
    ASL = bdf.getAvgSentenceLens(corpus)
    for id in corpus:
        df = corpus[id]
        ASW = np.mean(pd.Series(data=df['text'].apply(bdf.countSyllablesFinnish).to_numpy(), index=df['text'].to_numpy(dtype='str')).to_numpy(na_value=0))
        returnable[id] = 0.39*ASL[id] + 11.8*ASW - 15.59
    
    return returnable

#Alternative way of using the custom vectorizer to enable min-max normalization of the data


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