#Imports
import pandas as pd
import os
import numpy as np
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import itertools
import string
from collections import Counter
import random

#CONSTANTS

CONSONANTS = ['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','z']
VOCALS = ['a','e','i','o','u','y','å','ä','ö']
DIFTONGS_1 = ['ai','ei','oi','ui','yi','äi','öi','ey','iy','äy','öy','au','eu','iu','ou', 'aa', 'ee', 'ii', 'oo', 'uu', 'yy', 'ää', 'öö']
DIFTONGS_2 = ['ie','uo','yö']
SYLL_SET_1 = set(''.join(i) for i in itertools.product(CONSONANTS, VOCALS))
SYLL_SET_2 = set(''.join(i) for i in itertools.product(VOCALS, VOCALS)) - set(DIFTONGS_1+DIFTONGS_2)
SYLL_SET_3 = set(DIFTONGS_2)

def initBooksFromJsons(json_path: str) -> dict:
    """
    Function which takes in a path to a folder containing .json files produced by Trankit and creates python dicts from them
    :JSON_PATH: path to folder as str
    :return: python dict with form [book_name, pandas.DataFrame]
    """

    books = {}
    #Loading the conllus (jsons) as Dataframes

    for file in os.listdir(json_path):
        #Opening json contents
        with open(json_path+"/"+file) as json_file:
            #Transform into dataframe
            df = pd.read_json(json_file)
            #Append as dict juuuuust in case we need the metadata
            #Clip at 17 as the format for the filenames are standardized
            books[file[:17]] = df
    return books

def initBooksFromConllus(conllu_path: str) -> dict:
    """
    Function which takes in a path to a folder containing conllu files and returns a dict with pd.DataFrames of sentence data
    :conllu_path: path to folder of conllus as str
    :return: dict of form [book_name, pd.DataFrame], df is sentence data of a book
    """

    books = {}
    #Loading the conllus as Dataframes
    for file in os.listdir(conllu_path):
        #Opening conllus contents
        with open(conllu_path+"/"+file) as conllu_file:
            #Transform into dataframe
            df = pd.read_csv(conllu_file, sep="[\t]",skip_blank_lines=True,  header=None, on_bad_lines='error', engine='python')
            #Set names for columns
            df.columns = ['id', 'text', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']
            #Append as dict juuuuust in case we need the metadata
            #Clip at 17 as the format for the filenames are standardized
            books[file[:17]] = df
    return books

#Functions which extract data from a dictionary to get data on sentences

#Function that removes PUNCT
def getNoPunct(sentences: dict) -> dict:
    """
    Function which takes in a dict created by getTokenData and spits out a version with the upos tag PUNCT removed from the DataFrames
    Recommend only editing sentence data
    :sentences: dict of form [book_name, pandas.DataFrame]
    :return: dict of form [book_name, pandas.DataFrame] with upos PUNCT removed from df
    """
    no_punct = {}
    #Remove the rows from each dataframe what are classified as PUNCT
    for key in sentences:
        df = sentences[key]
        no_punct[key] = df[df.upos != "PUNCT"]
    return no_punct

#Function that takes in a dictionary of [book_name, conllu_dataframe] and returns a dictionary with [book_name, sentences_dataframe]
def getSentenceData(books: dict) -> dict:
    """
    Function which takes in a dict created by initBooks and spits out a version where DataFrames only contain sentences data (see CoNLLU file format for more info)
    :books: dict of form [book_name, pandas.DataFrame]
    :return: dict of form [book_name, pandas.DataFrame] with only sentence data
    """
    return_dict = {}
    with tqdm(range(len(books.keys())), desc="Extracting sentences...") as pbar:
        #For key-value pair in dict
        for key in books:
            #Init a new array for sentences
            sentence_dfs = []
            df = books[key]
            
            #Only care about the sentences
            for sentence in df['sentences']:
                #Add dfs created from sentences to list
                sentence_dfs.append(pd.DataFrame.from_dict(sentence['tokens']))
            #Map book_name to a dataframe from all its sentences
            sentece_df = pd.concat(sentence_dfs, ignore_index=True)
            return_dict[key]=sentece_df
            #Update pbar
            pbar.update(1)
    return return_dict

def maskPropn(corpus: dict[str,pd.DataFrame]) -> dict:
    """
    Function which masks all the proper nouns from the dict for classifier training purposes
    """
    returnable = {}
    for key in corpus:
        df = corpus[key]
        df.loc[df['upos'] == 'PROPN', 'lemma'] = ""
        df.loc[df['upos'] == 'PROPN', 'text'] = ""
        returnable[key] = df
    return returnable


#Function which returns a dictionary [book_name, lemma_freq_series]
def getLemmaFrequencies(sentences: dict) -> dict:
    """
    Function which takes in sentence data and creates a dict with the lemma frequencies of each book
    :sentences: dict of form [book_name, pandas.DataFrame]
    :return: dict of form [book_name, pandas.Series] series contains lemmas and frequencies
    """
    lemma_freqs = {}
    for key in sentences:
        lemma_freqs[key] = sentences[key]['lemma'].value_counts()
    return lemma_freqs

def getOnlyAlnums(sentences: dict, column: str) -> dict:
    """
    Function which takes in sentence data and cleans punctuation and other non-alnum characters
    :param sentences:dict of form [book_name, pd.DataFrame], df is sentence data
    :param column: name of the column which to clean (recommend 'text' or 'lemma')
    :return: dict of the same form
    """
    clean = {}
    for key in sentences:
        df = sentences[key]
        #Get rid of PUNCT
        no_punct = df[df.upos != "PUNCT"].copy()
        #Make words lowercase
        no_punct[column] = no_punct[column].apply(lambda x: x.lower())
        #Remove non-alnums
        no_punct[column] = no_punct[column].apply(lambda x: ''.join(filter(str.isalnum, x)))
        #Filter rows with nothing in them
        no_punct = no_punct[no_punct.text != '']
        clean[key] = no_punct
    return clean

#Get frequencies of words)
def getWordFrequencies(sentences: dict) -> dict:
    """
    Function which takes in a dict of sentence data and spits out a dict with pd.Series containing word frequencies of the books
    """
    word_freqs = {}
    for key in sentences:
        df = sentences[key]
        #Map book_name to pivot table
        word_freqs[key] = sentences[key]['text'].value_counts()
    return word_freqs

def getColumnFrequencies(corpus: dict[str,pd.DataFrame], columns=list[str]) -> dict[str,pd.DataFrame]:
    """
    A more general function for calculating frequencies of rows in our corpus format
    """
    freqs = {}
    for key in corpus:
        book = corpus[key]
        freqs[key] = book[columns].value_counts()
    return freqs


def getTokenAmounts(sentences: dict) -> dict:
    """
    Get amount of tokens in sentences
    """
    word_amounts = {}
    for key in sentences:
        df = sentences[key]
        word_amounts[key] = len(df)
    return word_amounts

#Get PoS frequencies
def getPOSFrequencies(sentences: dict, scaler_sentences: bool=None) -> dict:
    """
    Function which gets the POS frequencies of sentences of books
    """
    pos_freqs = {}

    if scaler_sentences:
        sentences_sizes = getNumOfSentences(sentences)

    for key in sentences:
        #Map book_name to pivot table
        if scaler_sentences:
            freqs = sentences[key]['upos'].value_counts()
            pos_freqs[key]=freqs/sentences_sizes[key]
        else:
            pos_freqs[key] = sentences[key]['upos'].value_counts()
        #pd.DataFrame.pivot_table(sentences[key], columns='upos', aggfunc='size').sort_values(ascending=False).reset_index().rename(columns={0: "frequency"})

    return pos_freqs


#Functions to get metrics from sentences

#Function the get the average length of the unique lemmas in the sentenes
def getAvgLen(data: dict, column: str=None) -> dict:
    """
    Get the average length of either words or lemmas from sentence data. Works for both original sentence data (pd.DataFrame) and processed ones,
    such as frequency data (pd.Series)
    :data: dict of form [book_name, pd.DataFrame], df should contain sentence data
    :column: name of the CoNLLU column for which to calculate the average. Recommend either 'text' for words and 'lemma' for lemmas
    :return: dict of [book_name, avg_len], where avg_len is float
    """
    avg_lens = {}
    for key in data:
        i = 1
        total_len = 0
        df = data[key]
        if type(df) is pd.DataFrame:
            #For each lemma count the length and add one to counter
            for lemma in df[column]:
                #Only care about strings
                if type(lemma) is str:
                    total_len += len(lemma)
                    i += 1
        elif type(df) is pd.Series:
            #For each lemma count the length and add one to counter
            for lemma in list(df.index):
                #Only care about strings
                if type(lemma) is str:
                    total_len += len(lemma)
                    i += 1
        #If no lemmas were found (should never happen but just in case), we make the avg_len be 0
        if i==1:
            avg_lens[key] = 0
        else:
            #Map book_name to avg lemma length
            avg_lens[key] = total_len/(i-1.0)
    return avg_lens

def getAvgSentenceLens(books: dict) -> dict:
    """
    Functon for getting the average length of sentences in each book
    :param books: dict of form [id, pd.DataFrame] like in the other methods
    :return: dict of form [id, double], where the double is the average sentence length of the corresponding book
    """
    lens = {}
    for key in books:
        df = books[key]
        help1 = len(df)
        nums = df.id.value_counts()
        num_of_sents = nums.iloc[0]
        lens[key] = (help1/num_of_sents)
    return lens

#Function to calculate DP (deviation of proportions) of all the words in the corpus
def getDP(v: dict, f_series: pd.Series, s: dict) -> tuple:
    """
    Function which calculates the dispersion (DP) based on the formula by Gries
    :v: dict of form [book_name, pd.Series], series has frequencies per book
    :f_df: pd.Series that includes the total frequencies of words/lemmas in the whole corpus
    :s: dict of form [book_name, ratio], where ratio is how much of the whole corpus a book takes
    :return: tuple, where the first member is a pd.Series with DP, the second is a series with DP_norm
    """
    #First get the minimum s
    min_s = 1
    for key in s:
        if s[key] < min_s:
            min_s = s[key]
    #For corpus parts that are length 1
    if min_s == 1:
        min_s = 0

    words = [x[0] for x in f_series.index]
    DP = []
    DP_norm = []
    with tqdm(range(len(f_series)), desc="DP calculations") as pbar:
        #Loop through every single word in the corpus
        for word in words:
            #Get the freq of the word in the whole corpus
            f = f_series[word]
            #Calculations according to the dispersion measure by Gries 2020
            abs_sum_i = [((v[key].get(word, 0.0))/f)-s[key] for key in v]
            #Calculate and append DP
            dp = np.sum(np.absolute(abs_sum_i))*0.5
            DP.append(dp)
            #Append DP_norm to list (alltho with how many documents we have, the normalization doesn't work very well at all)
            DP_norm.append(dp/(1-min_s))
            #Update pbar
            pbar.update(1)
    return pd.Series(DP, words), pd.Series(DP_norm, words)

#Function to calculate D (dispersion) of all the words in the corpus
def getDispersion(v: dict, f_series: pd.Series) -> pd.Series:
    """
    Function which calculates the dispersion of a word in a (sub)corpus
    :v: dict of form [book_name, pd.Series], series has frequencies per book
    :f_df: pd.Series that includes the total frequencies of words/lemmas in the whole corpus
    :return: pd.Series, with words/lemmas as indices and dispersions as values
    """

    words = [x[0] for x in f_series.index]
    #v_prepped = {key:pd.Series(data=np.multiply(v[key].to_numpy(), np.log(v[key].to_numpy())),index=v[key].index) for key in v}
    mass_frame = pd.concat([pd.Series(data=np.multiply(v[key].to_numpy(), np.log(v[key].to_numpy())),index=v[key].index) for key in v]).groupby(level=0).sum().fillna(0)
    corp_len = np.log(len(list(v.keys())))
    D = []
    with tqdm(range(len(f_series)), desc="D calculations") as pbar:
        #Loop through every single word in the corpus
        for word in words:
            #Get the freq of the word in the whole corpus
            f = f_series[word]
            # D = [log(p) * sum(p_i*log(p_i))/p]/log(n)
            #Calculate and append D
            D.append((np.log(f) - (mass_frame[word] / f))/corp_len)
            #Update pbar
            pbar.update(1)
    return pd.Series(D, words)

def getU(v: dict, f: pd.Series, D: pd.Series) -> pd.Series:
    """
    Function for calculating the Estimated frequency per 1 million words (U) for words/lemmas
    """
    corp_size = f.sum()
    base_scaler = 1000000 / corp_size
    words = [x[0] for x in f.index]
    #To speed up calculations, prepare everything that we can with numpy operations before doing anything in a for-loop
    #Here we calculate the f_min value for each word (frequency of word inside a book * total length of said book)
    v_prepped = {key:pd.Series(data=np.multiply(v[key].to_numpy(), v[key].sum()),index=v[key].index) for key in v}
    #Sum all f_mins together (essentailly flatten the dictionary)
    f_mins = list(v_prepped.values())[0]
    if len(v) != 1:
        for i in range(1, len(v_prepped)):
            f_mins = f_mins.add(list(v_prepped.values())[i], fill_value=0)
    #Scale with 1/N as per formula
    f_mins = f_mins * (1/corp_size)
    #Calculate U-values for each word
    U_data = [base_scaler * (f[word]*D.get(word, 0.0) + (1-D.get(word, 0.0)) * f_mins[word]) for word in words]
    return pd.Series(U_data, index=words)

def getSFI(U: pd.Series) -> pd.Series:
    """
    Function for calculating the Standardized Frequency Index (SFI) for the estimated frequency per 1 million words (U)
    """
    values = 10*(np.log10(U.to_numpy())+4)
    return pd.Series(values, U.index)

#Function to get contextual diversity
def getCD(v: dict) -> pd.Series:
    """
    Function which gets the contextual diversity of words/lemmas based on frequency data
    """
    #Get number of books
    books_num = len(v.keys())
    word_series = []
    #For each series attached to a book, look for a frequency list and gather all the words in a list
    for key in v:
        v_series = v[key]
        word_series.append([x[0] for x in v_series.index])
    #Add all words to a new series
    series = pd.Series(word_series)
    #Create series to count in how many books does a word appear in (explode the series comprised of lists)
    CD_raw = series.explode().value_counts()
    #Return Contextual Diversity by dividing the number of appearances by the total number of books
    return CD_raw/books_num

#Functions for getting values for different variables used in metrics


def getL(word_amounts: dict) -> int:
    """
    Function for getting the total length of the corpus in terms of the number of words
    """
    l = 0
    for key in word_amounts:
        l += word_amounts[key]
    return l

def getS(word_amounts: dict, l: int) -> dict:
    """
    Function for getting how big each part is in relation to the total size of the corpus
    """
    s = {}
    for key in word_amounts:
        s[key] = (word_amounts[key]*1.0)/l
    return s


def combineFrequencies(freq_data: dict) -> pd.Series:
    """
    Get the total frequencies of passed freq_data in the corpus
    """
    series = []
    #Add all series to list
    for key in freq_data:
        series.append(freq_data[key])
    #Concat all series together
    ser = pd.concat(series)
    to_return = ser.groupby(ser.index).sum()
    #Retain multi-indexes if used more than one column in getting frequency data
    if type(to_return.index[0]) == tuple:
        to_return.index = pd.MultiIndex.from_tuples(to_return.index)
    #Return a series containing text as index and total freq in collection in the other
    return to_return


#Functions to do with sub-corpora

def getAvailableAges(corpus: dict) -> list[int]:
    """
    Function which returns the ages that are currently available as sub corpora
    """
    return list(map(int,list(set([findAgeFromID(x) for x in list(corpus.keys())]))))


def getRangeSubCorp(corp: dict, num: int) -> dict:
    """
    Simple function to get sub_corpora from the whole package based on the target age, such that a book will go to +-1 range of its target age
    Naming conventions are ISBN_age-group_register, where age is an int [5,16]
    """
    sub_corp = {}
    for key in corp:
        age = int(findAgeFromID(key))
        if (num - age < 2 and num - age > -2):
            df = corp[key]
            sub_corp[key] = df
    return sub_corp

def getDistinctSubCorp(corp: dict, num: int) -> dict:
    """
    Simple function to get sub_corpora from the whole package based on the target age exactly, so eahc book will only be included once
    Naming conventions are ISBN_age-group_register, where age is an int [5,16]
    """
    sub_corp = {}
    for key in corp:
        if key.find('_'+str(num)+'_') != -1:
            sub_corp[key] = corp[key]
    return sub_corp


def combineSubCorpDicts(corps: list) -> dict:
    """
    Combine a list of sub-corp dicts into one dict
    """
    whole = corps[0].copy()
    for i in range(1, len(corps)):
        whole.update(corps[i])
    return whole

def combineSubCorpsData(corps: list, sum_together: bool):
    """
    Takes in a list of dataframes (or series) and combines them together
    """
    dfs = []
    for df in corps:
        dfs.append(df)
    combined = pd.concat(dfs)
    if sum_together:
        if type(combined) is pd.DataFrame:
            return combined.groupby(combined.columns[0])[combined.columns[1]].sum().reset_index()
        else:
            return combined.groupby(level=0).sum()
    return combined
    

def getTypeTokenRatios(v: dict, word_amounts: dict) -> pd.Series:
    """
    Function which gets the type-token ratios of each book that's in the corpus
    :param v:frequency data per book
    :param word_amounts:token amounts per book
    :return: pd.Series with book names being indexes and ttr being values 
    """
    names = []
    ttrs = []
    for key in word_amounts:
        v_df = v[key]
        #Get the number of unique entities in freq data
        types = len(v_df)
        #Get the number of token in book
        tokens = word_amounts[key]
        #Add ttr to lis
        ttrs.append(types/tokens)
        #Add key to list
        names.append(key)
    return pd.Series(ttrs, names)

def getZipfValues(l: int, f: pd.Series) -> pd.Series:
    """
    Function for calculating the Zipf values of words/lemmas in a corpus
    Zipf = ( (raw_freq + 1) / (Tokens per million + Types per million) )+3.0
    :param l: total length of corpus (token amount)
    :param f: series containing frequency data of words/lemmas for the corpus
    :return: pd.Series, where indexes are words/lemmas and values the Zipf values
    """
    indexes = f.index
    types_per_mil = len(indexes)/1000000
    tokens_per_mil = l/1000000
    zipfs = f.values+1
    zipfs = zipfs / (tokens_per_mil + types_per_mil)
    zipfs = np.log10(zipfs)
    zipfs = zipfs + 3.0
    #zipfs_ser = pd.Series(zipfs, indexes)
    return pd.Series(zipfs, indexes)

def cleanLemmas(sentences: dict) -> dict:
    """
    Function which takes in sentence data and cleans rows based on a number of filters on lemma forms to guarantee better quality data
    Does get rid of PUNCT
    :param sentences:dict of form [book_name, pd.DataFrame], df is sentence data
    :return: dict of the same form, but better data
    """
    clean = {}
    for key in sentences:
        df = sentences[key]
        no_punct = df.copy()

        #Make words lowercase
        no_punct['lemma'] = no_punct['lemma'].apply(lambda x: str(x).lower())
        #First mask
        #Remove lemmas which are not alnum or have '-' but no weird chars at start or end, length >1, has no ' ', and has no ','
        m = no_punct.lemma.apply(lambda x: (x.isalnum() 
                                            or (not x.isalnum() and '-' in x and x[0].isalnum() and x[len(x)-1].isalnum())
                                            or (not x.isalnum() and '#' in x and x[0].isalnum() and x[len(x)-1].isalnum())
                                            and len(x)>1 
                                            and not ' ' in x
                                            and not ',' in x))
        filtered = no_punct[m]
        #Second mask
        #Remove lemmas that have the same character more than thrice consecutively at the start (Finnish doesn't work like this)
        m_2 = no_punct.lemma.apply(lambda x: conseqChars(x)
                                   and not (x.isnumeric() and len(x)>4)
                                   )
        filtered_2 = filtered[m_2] 
        clean[key] = filtered_2
    return clean

def conseqChars(x: str):
    if len(x)>2:
        return not x[0]==x[1]==x[2]
    else:
        return True
    
def cleanWords(books: dict) -> dict:
    """
    Function for cleaning non-alnum characters from the beginning and ending of words and lemmas in sentence data
    :param books: dict of the sentence data of books
    :return: dictionary, where the dataframes have been cleaned
    """
    #Clean words
    clean = {}
    for key in books:
        df = books[key].copy()
        df['text'] = df['text'].apply(lambda x: delNonAlnumStart(str(x)))
        df['lemma'] = df['lemma'].apply(lambda x: delNonAlnumStart(str(x)))
        df['text'] = df['text'].apply(lambda x: delNonAlnumEnd(str(x)))
        df['lemma'] = df['lemma'].apply(lambda x: delNonAlnumEnd(str(x)))
        clean[key] = df.dropna()
    return clean

def delNonAlnumStart(x: str) -> str:
    '''
    Function for deleting non-alnum sequences of words from Conllu-files
    :param x: string that is at least 2 characters long
    :return: the same string, but with non-alnum characters removed from the start until the first alnum-character
    '''
    if not x[0].isalnum() and len(x)>1:
        ind = 0
        for i in range(len(x)):
            if x[i].isalnum():
                ind=i
                break
        return x[ind:]
    return x    

def delNonAlnumEnd(x: str) -> str:
    '''
    Function for deleting non-alnum sequences of words from Conllu-files
    :param x: string that is at least 2 characters long
    :return: the same string, but with non-alnum characters removed from the start until the first alnum-character
    '''
    if not x[-1].isalnum() and len(x)>1:
        ind = 0
        for i in range(1,len(x)):
            if x[-i].isalnum():
                ind=-i
                break
        return x[:ind+1]
    return x 

def ignoreOtherAlphabets(sentences: dict) -> dict:
    """
    Function which takes in sentence data and cleans rows based on if words contain characters not in the Finnish alphabe (e.g. cyrillic)
    :param sentences:dict of form [book_name, pd.DataFrame], df is sentence data
    :return: dict of the same form, but better data
    """
    clean = {}
    for key in sentences:
        df = sentences[key]
        no_punct = df.copy()

        #Make words lowercase
        no_punct['text'] = no_punct['text'].apply(lambda x: str(x).lower())
        #First mask
        #Remove words which are not in the Finnish alphabet
        m = no_punct.text.apply(lambda x: (
                len(x.encode("ascii", "ignore")) == len(x)
                or x.find('ä') != -1
                or x.find('ö') != -1 
                or x.find('å') != -1
            )
        )
        filtered = no_punct[m]
        clean[key] = filtered
    return clean

def getSharedWords(wordFrequencies1: dict, wordFrequencies2: dict) -> pd.Series:
    """
    Gives a pd.DataFrame object where there are two columns: first contains those words/lemmas which are shared and the second their combined frequencies
    """
    sub1 = combineFrequencies(wordFrequencies1)
    sub2 = combineFrequencies(wordFrequencies2)

    shared = pd.concat([sub1, sub2])
    mask = shared.index.duplicated(keep=False)
    shared = shared[mask]
    return shared.groupby(shared.index).sum()

def getTaivutusperheSize(corpus: dict) -> pd.Series:
    """
    Returns a series that contains the unique lemmas of the corpus and their 'inflection family size' (taivutusperheen koko)
    """
    #First, combine the data of separate books into one, massive df
    dfs = []
    for book in corpus:
        dfs.append(corpus[book])
    #Then limit to just words and lemmas
    combined_df = pd.concat(dfs, ignore_index=True)[['lemma','feats']]
    #Drop duplicate words
    mask = combined_df.drop_duplicates()
    #Get the counts of lemmas, aka the number of different inflections
    tper = mask.value_counts('lemma')
    return tper

def dfToLowercase(df):
    """
    Simple function which maps all fields into lowercase letters
    """
    return df.copy().applymap(lambda x: str(x).lower())

def getNumOfSentences(corpus: dict) -> dict:
    """
    Function for returning the amount of main clauses for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :return:dict of form [id, int]
    """

    sentences_sizes = {}
    for id in corpus:
        book = corpus[id]
        #Each sentences should have one word with deprel=='root' means the start of a new sentence (lause)
        sentences_sizes[id] = len(book[book['deprel'].astype(str)=='root'])
    return sentences_sizes

def getConjPerSentence(corpus: dict) -> dict:
    """
    Function for calculating the conjuction-to-sentence ratio for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :return:dict of form [id, int]
    """
    conj_sentences_ratio = {}
    sentences_sizes = getNumOfSentences(corpus)
    for id in corpus:
        book = corpus[id]
        conj_num = len(book[book['upos'] == ('CCONJ' or 'SCONJ')])
        conj_sentences_ratio[id] = conj_num/sentences_sizes[id]
    return conj_sentences_ratio

def getPosFeaturePerBook(corpus: dict, feature: str, scaler_sentences: bool=None) -> dict:
    """
    Function for calculating the amount of wanted POS features for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :param feature: str that maps to some dependency relation in the CoNLLU format (https://universaldependencies.org/u/dep/index.html)
    :param scaler_sentences: optional bool that forces the use of sentence amounts for scaling
    :return:dict of form [id, float]
    """

    returnable = {}
    if scaler_sentences:
        sentences_sizes = getNumOfSentences(corpus)
    for key in corpus:
        book = corpus[key]
        if scaler_sentences:
            num = len(book[book['upos'] == feature])/sentences_sizes[key]
        else:
            num = len(book[book['upos'] == feature])
        returnable[key] = num
    return returnable

def getDeprelFeaturePerBook(corpus: dict, feature: str, scaler_sentences: bool=None) -> dict:
    """
    Function for calculating the amount of wanted deprel features words for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :param feature: str that maps to some dependency relation in the CoNLLU format (https://universaldependencies.org/u/dep/index.html)
    :param scaler_sentences: optional bool that forces the use of sentence amounts for scaling
    :return:dict of form [id, float]
    """

    returnable = {}
    if scaler_sentences:
        sentences_sizes = getNumOfSentences(corpus)
    for key in corpus:
        book = corpus[key]
        if scaler_sentences:
            num = len(book[book['deprel'] == feature])/sentences_sizes[key]
        else:
            num = len(book[book['deprel'] == feature])
        returnable[key] = num
    return returnable

def getFeatsFeaturePerBook(corpus: dict, feature: str, scaler_sentences: bool=None) -> dict:
    """
    Function for calculating the amount of wanted feats feature for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :param feature: str that maps to some dependency relation in the CoNLLU format (https://universaldependencies.org/u/dep/index.html)
    :param scaler_sentences: optional bool that forces the use of sentence amounts for scaling
    :return:dict of form [id, float]
    """

    returnable = {}
    if scaler_sentences:
        sentences_sizes = getNumOfSentences(corpus)
    for key in corpus:
        book = corpus[key]
        #Mask those rows that don't have the wanted feature
        m = book.copy().feats.apply(lambda x: (
            x.find(feature) != -1
                )
            )
        if scaler_sentences:
            num = len(book[m])/sentences_sizes[key]
        else:
            num = len(book[m])
        returnable[key] = num
    return returnable

def cohensdForSubcorps(subcorp1: dict, subcorp2: dict) -> float:
    """
    Function for calculating the effect size using Cohen's d for some feature values of two subcorpora
    :param subcorp1: dictionary of form [id, float], calculated with e.g. getDeprelFeaturePerBook()
    :param subcorp2: dict of the same form as above
    :return: flaot measuring the effect size
    """
    data1 = list(subcorp1.values())
    data2 = list(subcorp2.values())
    #Sample size
    n1, n2 = len(data1), len(data2)
    #Variance
    s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    #Pooled standard deviation
    s = math.sqrt( ((n1-1)*s1 + (n2-1)*s2)/(n1+n2-2) )
    #Return Cohen's d
    return ((np.mean(data1)-np.mean(data2)) / s)

def getMultiVerbConstrNumPerSentence(corpus: dict) -> dict:
    """
    Function for calculating the conjuction-to-sentence ratio for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :return:dict of form [id, int]
    """
    multiverb_sentences_ratio = {}
    sentences_sizes = getNumOfSentences(corpus)
    for id in corpus:
        book = corpus[id]
        modal_verb_num = len(book[((book['upos'] == 'AUX') & (book['xpos'] == 'V') & (book['deprel'] == 'aux')) | ((book['upos'] == 'VERB') & (book['deprel'] == 'xcomp'))])
        multiverb_sentences_ratio[id] = modal_verb_num/sentences_sizes[id]
    return multiverb_sentences_ratio

def getPreposingAdverbialClauses(corpus: dict[str,pd.DataFrame]) -> dict:
    """
    Function for calculating the number of preposing adverbial clauses in a conllu file
    """
    returnable = {}
    for key in corpus:
        df = corpus[key]
        df['head'] = df['head'].apply(lambda x: int(x))
        df['id'] = df['id'].apply(lambda x: int(x))
        prep_advcl = 0
        advcl_id = 1000
        root_id = 1000
        for i in range(len(df)):
            if df.loc[i]['id'] == 1:
                advcl_id = 1000
                root_id = 1000
            if df.loc[i]['deprel'] == 'root':
                root_id = i
            if df.loc[i]['deprel'] == 'advcl':
                advcl_id = i
                if advcl_id < root_id:
                    prep_advcl += 1
        returnable[key] = prep_advcl
    return returnable

def getDictAverage(corp_data: dict) -> float:
    """
    Simple function for calculating the average value of a dict containing book ids and some numerical values
    """
    return sum(list((corp_data.values())))/len(list(corp_data.keys()))

def getBookLemmaCosineSimilarities(corpus: dict[str,pd.DataFrame], f_lemma: pd.Series) -> pd.DataFrame:
    """
    Calculating cosine similarities of all lemmas between the books in the corpus. Inspired by Korochkina et el. 2024
    """
    tf_idf_scores = {}

    #Sort the books so that we get groupings by age group
    sorted_keys = list(corpus.keys())
    sorted_keys.sort(key=lambda x:int(findAgeFromID(x)))

    #Get all corpus' lemmas from lemma frequency data
    all_lemmas = list(f_lemma.index)
    book_vectorizer = TfidfVectorizer(vocabulary=all_lemmas)
    for book in sorted_keys:
        #Tf-idf scores from lemma data of a book
        book_lemmas = " ".join(corpus[book]['lemma'].to_numpy(dtype=str))
        #print(book_lemmas.values)
        tf_idf_scores[book] = book_vectorizer.fit_transform([book_lemmas])
    similarity_scores = {}
    for book in sorted_keys:
        #Compare current book to every other book
        scores = []
        for comp in sorted_keys:
            scores.append(cosine_similarity(tf_idf_scores[book], tf_idf_scores[comp]))
        similarity_scores[book] = scores
    #Create df
    matrix_df = pd.DataFrame.from_dict(similarity_scores, orient='index').transpose()
    #Set indexes correctly
    matrix_df.index = tf_idf_scores.keys()
    #Dig out the values from nd.array
    matrix_df_2 = matrix_df.copy().applymap(lambda x: x[0][0])
    return matrix_df_2
    return addAgeGroupSeparatorsToDF(matrix_df_2)

def addAgeGroupSeparatorsToDF(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function for adding separator lines to a df that's meant to be shown as a heatmap!
    """
    indices = list(df.index)
    one2two = 0
    while int(findAgeFromID(indices[one2two]))<9:
        one2two += 1
    two2three = one2two
    while int(findAgeFromID(indices[two2three]))<13:
        two2three += 1
    df.insert(one2two, 'one2two', pd.Series([1]*len(indices)))
    df.insert(two2three+1, 'two2three', pd.Series([1]*len(indices)))
    temp_dict = dict(zip(df.columns, ([1]*(len(indices)+2))))
    row1 = pd.DataFrame(temp_dict, index=['one2two'])
    row2 = pd.DataFrame(temp_dict, index=['two2three'])
    df_2 = pd.concat([df.iloc[:one2two], row1, df.iloc[one2two:]])
    df_2 = pd.concat([df_2.iloc[:two2three+1], row2, df_2.iloc[two2three+1:]])
    return df_2

def combineSeriesForExcelWriter(f_lemmas, corpus, lemma_DP, lemma_CD, lemma_zipfs, f_words, word_DP, word_CD, word_zipfs, pos):
    """
    Helper function for combining various Series containing lemma/word data into compact dataframes
    """
    lemma_data = pd.concat([f_lemmas, getTaivutusperheSize(corpus), lemma_DP, lemma_CD], axis=1)
    lemma_data.columns = ['frequency','t_perh_size', 'DP', 'CD']

    word_data = pd.concat([f_words, word_DP, word_CD], axis=1)
    word_data.columns = ['frequency', 'DP', 'CD']


    return lemma_data, word_data


def filterRegisters(corpus: dict[str,pd.DataFrame], registers: list[int]) -> dict[str,pd.DataFrame]:
    """
    Function for creating a register sepcific subcorpus. Valid registers are:
    1 = Fiction
    2 = Non-fiction, non-textbook
    3 = Textbook
    You can pass as many registers as you want (any valid subset of [1,2,3])
    """

    returnable = {}
    for key in corpus:
        if int(key[-1]) in registers:
            df = corpus[key]
            returnable[key] = df
    return returnable

#Moving to a regression task instead of hard age groups

def findAgeFromID(key: str) -> str:
    "Function that returns the age information embedded in a book id"
    return key[key.find('_')+1:key.find('_')+1+key[key.find('_')+1:].find('_')]

def mapGroup2Age(corpus: dict[str,pd.DataFrame], sheet_path: str) -> dict[str,pd.DataFrame]:
    """
    Function for changing the file keys to use exact ages instead of age groups [1,3]
    """

    returnable = {}
    isbn2age_series = pd.DataFrame(pd.read_excel(sheet_path, index_col=0))
    for key in corpus:
        df = corpus[key]
        new_key = key
        new_key = key[:14] +  str(isbn2age_series.at[int(key[:13]),isbn2age_series.columns[0]]) + key[15:]
        returnable[new_key] = df
    return returnable

def mapExactAgeToMean(corpus: dict[str,pd.DataFrame]) -> dict[str,pd.DataFrame]:
    """
    Function for taking exact ages in ids and mapping them to age groups/means of age intervals
    """
    returnable = {}
    for key in corpus:
        if int(findAgeFromID(key))<9:
            new_key = key[:key.find('_')]+'_7_'+key[-1]
        elif 8<int(findAgeFromID(key))<13:
            new_key = key[:key.find('_')]+'_10_'+key[-1]
        else:
            new_key = key[:key.find('_')]+'_14_'+key[-1]
        df = corpus[key]
        returnable[new_key] = df
    return returnable

def getPosPhraseCounts(corpus: dict[str,pd.DataFrame], upos: str) -> dict[str,float]:
    """
    Function for calculating the number of POS phrases for each book in (sub)corpus.
    Working POS-tags are those listed in the CoNLLU file format
    """
    returnable = {}
    for key in corpus:
        df = corpus[key]
        returnable[key] = len(df[(df['upos'] == upos) & (df['deprel'] == 'root')])
    return returnable

def scaleCorpusData(corpus_data: dict[str,float], scaling_data: dict[str,float]):
    """
    Function for scaling some previously calculated data (e.g. word counts) with some other measure (e.g. number of sentences)
    """
    returnable = {}
    for key in corpus_data:
        returnable[key] = corpus_data[key]/scaling_data[key]
    return returnable

def getPosNGramForCorpus(corpus: dict[str,pd.DataFrame], n: int) -> dict[str, Counter]:
    """
    Function for getting the number of wanted length POS n-grams for each book in corpus
    """
    returnable = {}
    for key in corpus:
        df = corpus[key]['upos']
        n_grams = []
        for i in range(len(df)-(n-1)):
            n_grams += [list(df.iloc[i:i+n].to_numpy(str))]
        n_grams = map(tuple, n_grams)
        returnable[key] = Counter(n_grams)
    return returnable

def getFleschKincaidGradeLevel(corpus: dict):
    returnable = {}
    ASL = getAvgSentenceLens(corpus)
    for id in corpus:
        df = corpus[id]
        ASW = np.mean(pd.Series(data=df['text'].apply(countSyllablesFinnish).to_numpy(), index=df['text'].to_numpy(dtype='str')).to_numpy(na_value=0))
        returnable[id] = 0.39*ASL[id] + 11.8*ASW - 15.59
    
    return returnable

def getModifiedSmogIndexForFinnish(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    """
    Calculate the SMOG (Simple Measure of Gobbledygook) index for a conllu-file.
    Using the version modified for Finnish as presented by Geoff Taylor (https://doi.org/10.1016/j.ssci.2012.01.016)
    It can only be calculated for texts that are at least 30 sentences long, so if the number of sentences is lower, we return 0.
    """
    #The modification is to increase the number of syllables from 3 in the original SMOG index to 5 for Finnish
    poly_syllable_cutoff = 5

    if len(id_tree) < 31:
        return 0
    
    smog_sentences = []
    sentences_to_pick_from = list(id_tree.keys())
    cutoff = math.floor((len(sentences_to_pick_from)/3))
    #Randomly sample 10 sentences from the beginning, middle, and end of the text
    for i in range(3):
        start_index = random.randint(0,cutoff-10)
        smog_sentences += sentences_to_pick_from[start_index:start_index+10]
        sentences_to_pick_from = sentences_to_pick_from[cutoff:]
    #Count number of polysyllable words in the sampled sentences
    num_of_polysyllables = 0
    for sentence_head in smog_sentences:
        tree = id_tree[sentence_head]
        for leaf in tree:
            if countSyllablesFinnish(conllu['text'].iloc[leaf]) > poly_syllable_cutoff-1:
                num_of_polysyllables += 1
    #Return SMOG index formula result
    return 1.0430*np.sqrt(num_of_polysyllables)+3.1291

#Coleman-Liau index
def getColemanLiauIndex(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    num_of_words = len(conllu)
    num_of_sentences = len(id_tree)
    num_of_letters = np.sum(list(map(len, conllu['text'].to_numpy(str))))
    return 0.0588 * (num_of_letters / num_of_words) * 100 - 0.296 * (num_of_sentences / num_of_words) * 100 - 15.8

#Automated readability index
def getARI(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    num_of_words = len(conllu)
    num_of_sentences = len(id_tree)
    num_of_letters = np.sum(list(map(len, conllu['text'].to_numpy(str))))
    return 4.71 * (num_of_letters / num_of_words) + 0.5 * (num_of_words / num_of_sentences) - 21.43

def buildIdTreeFromConllu(conllu: pd.DataFrame) -> dict[int,list[int]]:
    """
    Build a tree for each sentence in a conllu file, where each node points to the corresponding DataFrame row of a line in the conllu-file
    """
    #conllu['id'] = conllu['id'].apply(lambda x: int(x))
    id_tree = {}
    #First fetch ids marking boundaries for each sentence
    sentence_ids = []
    start = 0
    for i in range(1,len(conllu)):
        if conllu.iloc[i]['id'] == '1':
            sentence_ids.append((start,i-1))
            start = i
    sentence_ids.append((start, len(conllu)-1))
    #Build trees for each sentence
    for sentence in sentence_ids:
        root = 0
        sent_locs = range(sentence[0],sentence[1]+1)
        heads = conllu.iloc[sentence[0]:sentence[1]+1]['head'].to_numpy(int)-1
        sent_tree = {x:[] for x in sent_locs}
        for j in range(len(heads)):
            if heads[j] == -1:
                root = sent_locs[j]
            else:
                children = sent_tree[sent_locs[heads[j]]]
                children.append(sent_locs[j])
                sent_tree[sent_locs[heads[j]]] = children            
        id_tree[root] = sent_tree
    return id_tree

def getDepthOfTree(head:int, tree: dict[int,list[int]], depth=0) -> int:
    """
    Get syntactic tree depth recursively
    """
    next = depth+1
    children = tree[head]
    if len(children) > 0:
        for child in children:
            depth = max(depth, getDepthOfTree(child, tree, next))
    return depth


def getMeanSyntacticTreeDepth(tree):
    """
    Get the average (mean) depth of the syntactic tree of a conllu-file
    """
    depths = []
    for head in tree:
        depths.append(getDepthOfTree(head, tree[head]))
    return np.mean(depths)

def getMaxSyntacticTreeDepth(tree):
    """
    Get the average (mean) depth of the syntactic tree of a conllu-file
    """
    depths = []
    for head in tree:
        depths.append(getDepthOfTree(head, tree[head]))
    return max(depths)

def getIdTreeNGram(tree, prev_round=[]):
    """
    Recursively add layers to the n-grams.
    Returns a list of lists, which consist of ids. Amount per list is the specified n.
    """
    current_round = []
    for gram in prev_round:
        if len(tree[gram[-1]]) > 0:
            for leaf in tree[gram[-1]]:
                current_round.append(gram+[leaf])
    return current_round

def getSyntacticTreeNGram(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]], n: int, feature_column: str):
    """
    Function for getting n-grams from a conllu-file of the wanted feature (UD output) column
    Most often you will have either 'deprel' for dependency relations or 'upos' for POS tags
    Expects you to have built the id-tree beforehand using buildIdTreeFromConllu()
    Returns the n-grams as a dictionary of tuple-count pairs
    """
    feat_col = conllu[feature_column]
    all_id_grams = []
    for root in id_tree:
        tree = id_tree[root]
        init_grams = [[x] for x in list(tree.keys())]
        for i in range(1, n):
            init_grams = getIdTreeNGram(tree, init_grams)
        all_id_grams += init_grams
    all_n_grams = map(tuple, [[feat_col[y] for y in x] for x in all_id_grams])
    return Counter(all_n_grams)

def findNestingSubclauses(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    """
    Finds the subordinate clauses that have nesting clauses (clause within a clause)
    Only looks to find at least one nesting clause and return a list of dicts with the following keys:
    sentence_head:id of sentence head, clause_head:id of clause with nesting clauses, clause_type:deprel type of head 
    """
    deprel_conllu = conllu['deprel']
    clauses = ['csubj', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl', 'xcomp:ds']
    nesting_clauses = []
    #Check children of sentence heads
    for head in id_tree:
        tree = id_tree[head]
        head_children = tree[head]
        for child in head_children:
            child_deprel = deprel_conllu.iloc[child]
            #If children deprel is a clause
            if child_deprel in clauses:
                #Check all grandchildren of the child that is a clause
                for grandchild in tree[child]:
                    grandchild_deprel = deprel_conllu.iloc[grandchild]
                    #If a nesting clause is found, append child's data to list and move to the next child
                    if grandchild_deprel in clauses:
                        nesting_clauses.append({'sentence_head':head, 'clause_head':child, 'clause_type':child_deprel})
                        break
                
    return nesting_clauses

def findStackingClauses(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    """
    Finds the subordinate clauses that have stacking clauses (coordinating clause within subordinating clause)
    Only looks to find at least one stacking clause and return a list of dicts with the following keys:
    sentence_head:id of sentence head, clause_head:id of clause with stacking clauses, clause_type:deprel type of head 
    """
    deprel_conllu = conllu['deprel']
    clauses = ['csubj', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl', 'xcomp:ds']
    coordinating = 'conj'
    stacking_clauses = []
    #Check children of sentence heads
    for head in id_tree:
        tree = id_tree[head]
        head_children = tree[head]
        for child in head_children:
            child_deprel = deprel_conllu.iloc[child]
            #If children deprel is a clause
            if child_deprel in clauses:
                #Check all grandchildren of the child that is a clause
                for grandchild in tree[child]:
                    grandchild_deprel = deprel_conllu.iloc[grandchild]
                    #If a stacking clause is found, append child's data to list and move to the next child
                    if grandchild_deprel == coordinating:
                        stacking_clauses.append({'sentence_head':head, 'clause_head':child, 'clause_type':child_deprel})
                        break
    
    return stacking_clauses
                

def getNonClausalChildrenAmount(deprel_conllu: pd.Series, tree: dict[int, list[int]], head):
    clauses = ['csubj', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl', 'xcomp:ds']
    leaves = tree[head]
    non_clausal = [head]
    #While there are leaves yet to be explored
    while len(leaves) > 0:
        #Pop first member and select it
        leaf = leaves.pop(0)
        #If leaf starts a new clause, move on
        if deprel_conllu.iloc[leaf] in clauses:
            continue
        #If leaf did not start a new clause, add it to non_clausal
        non_clausal.append(leaf)
        #Get the children of leaf and if there are any, append them to the list of leaves to explore
        children = tree[leaf]
        if len(children) > 0:
            leaves += children
    #return the amount of non_clausal members of the clause explored
    return len(non_clausal)

def findMeanLengthOfClause(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    """
    Function that finds the mean length of clauses in a given conllu-snippet.
    Calculated by traversing the syntactic tree and seeing how many children start new clauses and how many belong to the head (obviosuly also including grandchildren etc.)
    """
    deprel_conllu = conllu['deprel']
    clauses = ['csubj', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl', 'xcomp:ds', 'root']
    clause_lengths = []
    for head in id_tree:
        tree = id_tree[head]
        clausal_heads = []
        for i in tree:
            if deprel_conllu.iloc[i] in clauses:
                clausal_heads.append(i)
        #clausal_heads = [i for i in tree if deprel_conllu.iloc[i] in clauses]
        for ch in clausal_heads:
            clause_lengths.append(getNonClausalChildrenAmount(deprel_conllu, tree, ch))
    return np.mean(clause_lengths)

def getPOSVariation(conllu: pd.DataFrame, pos: str):
    """
    Function for getting the variation of words belonging to a sepcific POS category.
    Measures the ratio between unique words and all words, but in specific POS categories.
    """
    all_pos_tags_present = conllu['upos'].drop_duplicates().to_numpy(str)
    if pos not in all_pos_tags_present:
        return 0
    all_specific_pos = conllu[conllu['upos'] == pos]
    reduced_df = all_specific_pos[['text','upos']]
    uniq_words = reduced_df['text'].apply(lambda x: str(x).lower()).drop_duplicates()
    return len(uniq_words) / len(all_specific_pos)

def getCorrectedPOSVariation(conllu: pd.DataFrame, pos: str):
    """
    Same as POS variation, except we divide the amount of unique words by a 'corrected' term, rather than the raw number of words.
    The corrected term is equal to sqrt(2 * total number of words with specified POS tag)
    """
    all_pos_tags_present = conllu['upos'].drop_duplicates().to_numpy(str)
    if pos not in all_pos_tags_present:
        return 0
    all_specific_pos = conllu[conllu['upos'] == pos]
    reduced_df = all_specific_pos[['text','upos']]
    uniq_words = reduced_df['text'].apply(lambda x: str(x).lower()).drop_duplicates()
    return len(uniq_words) / np.sqrt(2*len(all_specific_pos))

def getRatioOfFunctionWords(conllu: pd.DataFrame):
    """
    Function for calculating the ratio between function words and content words.
    Essentially means dividing the number of function words (all other POS tags) by the number of content words (NOUN, PROPN, ADJ, NUM, and VERB)
    """
    num_of_content_words = len(conllu[(conllu['upos'] == 'NOUN') | (conllu['upos'] == 'PROPN') | (conllu['upos'] == 'ADJ') | (conllu['upos'] == 'NUM') | (conllu['upos'] == 'VERB')])
    #Don't divide by 0...
    if num_of_content_words == 0:
        return 1.0
    return (len(conllu)-num_of_content_words)/num_of_content_words

def getPreposingAdverbialClauses(corpus: dict[str,pd.DataFrame]) -> dict:
    """
    Function for calculating the number of preposing adverbial clauses in a conllu file
    """
    returnable = {}
    for key in corpus:
        df = corpus[key]
        #df['head'] = df['head'].apply(lambda x: int(x))
        #df['id'] = df['id'].apply(lambda x: int(x))
        prep_advcl = 0
        advcl_id = 1000
        root_id = 1000
        for i in range(len(df)):
            if df.loc[i]['id'] == '1':
                advcl_id = 1000
                root_id = 1000
            if df.loc[i]['deprel'] == 'root':
                root_id = i
            if df.loc[i]['deprel'] == 'advcl':
                advcl_id = i
                if advcl_id < root_id:
                    prep_advcl += 1
        returnable[key] = prep_advcl
    return returnable

def countSyllablesFinnish(word: str) -> int:
    """
    Function for calculating syllables of words in Finnish.
    Is not perfect, as cases like "Aie" will get marked as having one syllable instead of two, 
    but these edge cases are very hard to code into rules and due to limited resources, have been left 'unfinished'
    """
    syll_count = 0
    first_round_word = ""

    word = str(word).capitalize()
    #If more than one char AND does not start with punctuation/numerals
    if len(word) > 1 and not word[0] in string.punctuation and not word[0].isnumeric():
        temp_ind = 0
        for i in range(1,len(word)):
            bigram = word[i-1]+word[i]
            #If consonant+vowel
            if bigram in SYLL_SET_1:
                first_round_word += word[temp_ind:i-1]+"#"
                temp_ind = i-1
        first_round_word += word[temp_ind:]
    if first_round_word.count('#') > 0:
        candidates = first_round_word.split('#')
        for i in range(len(candidates)):
            candidate = candidates[i]
            if len(candidate) == 0:
                continue
            syll_count += 1
            for j in range(1,len(candidate)):
                bigram = candidate[j-1]+candidate[j]
                #If not recognized diftong
                if bigram in SYLL_SET_2:
                    syll_count += 1
                #If special diftong that counts as a syllable after the first syllable
                if i!=0 and bigram in SYLL_SET_3:
                    syll_count += 1
    elif len(word) == 0:
        return 0
    #If comprises of only one character and not punct/sym/num
    elif not word[0] in string.punctuation and not word[0].isnumeric():
        syll_count += 1
    return syll_count

def getSyllableAmountsForWords(words: pd.Series) -> pd.Series:
    "Helper function to apply syllable counting to all words"
    uniq_words = words.drop_duplicates()
    return pd.Series(data=uniq_words.apply(countSyllablesFinnish).to_numpy(), index=uniq_words.to_numpy(dtype='str'))

def getAverageSyllablesPerSentence(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    """
    Calculate the average amount of syllables per sentence (total number of syllables / number of sentences)
    """
    syllables = np.sum([countSyllablesFinnish(x) for x in conllu['text'].to_numpy(str)])
    return syllables / len(id_tree)

def getStatisticsForDatabaseOnlyPos(sub_corpora, word_age_appearances=None, lemma_age_appearances=None):
    #If wfsa or lfsa included
    flag = (word_age_appearances or lemma_age_appearances) == None
    returnable = {}
    if flag:
        word_age_appearances = {}
        lemma_age_appearances = {}

    #First sort out the subcorpora
    for s in sub_corpora:
        if flag:
            sub_corp = s
        else:
            sub_corp = sub_corpora[s]
        combined_data = pd.concat(sub_corp.values()).reset_index()
        filtered_data = combined_data[['text','lemma','upos']]
        filtered_data = filtered_data.drop_duplicates(['text','lemma','upos'], ignore_index=True)
        #Add word-pos frequencies
        v_words_pos = getColumnFrequencies(sub_corp, ['text','upos'])
        word_pos_freqs = combineFrequencies(v_words_pos)
        filtered_data['Word-POS F'] = [word_pos_freqs[x[0]][x[1]] for x in filtered_data[['text','upos']].to_numpy(dtype='str')]
        #Add word frequencies
        v_words = getColumnFrequencies(sub_corp, ['text'])
        word_freqs = combineFrequencies(v_words)
        filtered_data['Word F'] = [word_freqs[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word CD
        word_CD = getCD(v_words)
        filtered_data['Word CD'] = [word_CD[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word D
        word_D = getDispersion(v_words, word_freqs)
        filtered_data['Word D'] = [word_D[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word U
        word_U = getU(v_words, word_freqs, word_D)
        filtered_data['Word U'] = [word_U[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word SFI
        word_SFI = getSFI(word_U)
        filtered_data['Word SFI'] = [word_SFI[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word zipf-values
        l = getL(getTokenAmounts(sub_corp))
        word_zipfs = getZipfValues(l, word_freqs)
        filtered_data['Word Zipf'] = [word_zipfs[x] for x in filtered_data['text'].to_numpy(dtype='str')]


        #Add lemma frequencies
        v_lemmas = getColumnFrequencies(sub_corp, ['lemma'])
        lemma_freqs = combineFrequencies(v_lemmas)
        filtered_data['Lemma F'] = [lemma_freqs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma CD
        lemma_CD = getCD(v_lemmas)
        filtered_data['Lemma CD'] = [lemma_CD[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma D
        lemma_D = getDispersion(v_lemmas, lemma_freqs)
        filtered_data['Lemma D'] = [lemma_D[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma U
        lemma_U = getU(v_lemmas, lemma_freqs, lemma_D)
        filtered_data['Lemma U'] = [lemma_U[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma SFI
        lemma_SFI = getSFI(lemma_U)
        filtered_data['Lemma SFI'] = [lemma_SFI[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma zipf-values
        lemma_zipfs = getZipfValues(l, lemma_freqs)
        filtered_data['Lemma Zipf'] = [lemma_zipfs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add taivutusperhe size
        tv_sizes = getTaivutusperheSize(sub_corp)
        filtered_data['Lemma MPS'] = [tv_sizes[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add syllables per word
        syllable_amount = getSyllableAmountsForWords(filtered_data['text'])
        filtered_data['Word Syllables'] = [syllable_amount[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add length of word
        filtered_data['Word Length'] = filtered_data['text'].str.len()
        #Add length of lemma
        filtered_data['Lemma Length']= filtered_data['lemma'].str.len()
        key = s
        #Slow but steady way of adding words and first appearance ages...
        if flag:
            key = findAgeFromID(list(sub_corp.keys())[0])
            for w in word_freqs.index:
                word_age_appearances.setdefault(w[0],key)
            for le in lemma_freqs.index:
                lemma_age_appearances.setdefault(le[0],key)
        #Add first appearance
        filtered_data['Word FSA'] = [word_age_appearances[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        filtered_data['Lemma FSA'] = [lemma_age_appearances[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add to dictionary
        returnable[key] = filtered_data.sort_values('text')

    if flag:
        return returnable, word_age_appearances, lemma_age_appearances
    else:
        return returnable

def getStatisticsForDatabasePosFeats(sub_corpora, word_age_appearances=None, lemma_age_appearances=None):
    #If wfsa or lfsa included
    flag = (word_age_appearances or lemma_age_appearances) == None
    returnable = {}
    if flag:
        word_age_appearances = {}
        lemma_age_appearances = {}

    #First sort out the subcorpora
    for s in sub_corpora:
        if flag:
            sub_corp = s
        else:
            sub_corp = sub_corpora[s]
        combined_data = pd.concat(sub_corp.values()).reset_index()

        filtered_data = combined_data[['text','lemma']]
        filtered_data["upos+features"] = combined_data[['upos','feats']].agg('+'.join, axis=1)
        filtered_data = filtered_data.drop_duplicates(['text','lemma','upos+features'], ignore_index=True)
        #Add word-pos frequencies
        freqs = {}
        for key in sub_corp:
            book = sub_corp[key]
            book["upos+features"] = book[['upos','feats']].agg('+'.join, axis=1)
            freqs[key] = book[["text","upos+features"]].value_counts()
        v_words_pos = freqs
        word_pos_freqs = combineFrequencies(v_words_pos)
        filtered_data['Word-POS+FEATS F'] = [word_pos_freqs[x[0]][x[1]] for x in filtered_data[['text','upos+features']].to_numpy(dtype='str')]
        #Add word frequencies
        v_words = getColumnFrequencies(sub_corp, ['text'])
        word_freqs = combineFrequencies(v_words)
        filtered_data['Word F'] = [word_freqs[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word CD
        word_CD = getCD(v_words)
        filtered_data['Word CD'] = [word_CD[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word D
        word_D = getDispersion(v_words, word_freqs)
        filtered_data['Word D'] = [word_D[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word U
        word_U = getU(v_words, word_freqs, word_D)
        filtered_data['Word U'] = [word_U[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word SFI
        word_SFI = getSFI(word_U)
        filtered_data['Word SFI'] = [word_SFI[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word zipf-values
        l = getL(getTokenAmounts(sub_corp))
        word_zipfs = getZipfValues(l, word_freqs)
        filtered_data['Word Zipf'] = [word_zipfs[x] for x in filtered_data['text'].to_numpy(dtype='str')]


        #Add lemma frequencies
        v_lemmas = getColumnFrequencies(sub_corp, ['lemma'])
        lemma_freqs = combineFrequencies(v_lemmas)
        filtered_data['Lemma F'] = [lemma_freqs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma CD
        lemma_CD = getCD(v_lemmas)
        filtered_data['Lemma CD'] = [lemma_CD[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma D
        lemma_D = getDispersion(v_lemmas, lemma_freqs)
        filtered_data['Lemma D'] = [lemma_D[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma U
        lemma_U = getU(v_lemmas, lemma_freqs, lemma_D)
        filtered_data['Lemma U'] = [lemma_U[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma SFI
        lemma_SFI = getSFI(lemma_U)
        filtered_data['Lemma SFI'] = [lemma_SFI[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add lemma zipf-values
        lemma_zipfs = getZipfValues(l, lemma_freqs)
        filtered_data['Lemma Zipf'] = [lemma_zipfs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add taivutusperhe size
        tv_sizes = getTaivutusperheSize(sub_corp)
        filtered_data['Lemma MPS'] = [tv_sizes[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add syllables per word
        syllable_amount = getSyllableAmountsForWords(filtered_data['text'])
        filtered_data['Word Syllables'] = [syllable_amount[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add length of word
        filtered_data['Word Length'] = filtered_data['text'].str.len()
        #Add length of lemma
        filtered_data['Lemma Length']= filtered_data['lemma'].str.len()
        key = s
        #Slow but steady way of adding words and first appearance ages...
        if flag:
            key = findAgeFromID(list(sub_corp.keys())[0])
            for w in word_freqs.index:
                word_age_appearances.setdefault(w[0],key)
            for le in lemma_freqs.index:
                lemma_age_appearances.setdefault(le[0],key)
        #Add first appearance
        filtered_data['Word FSA'] = [word_age_appearances[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        filtered_data['Lemma FSA'] = [lemma_age_appearances[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add to dictionary
        returnable[key] = filtered_data.sort_values('text')

    if flag:
        return returnable, word_age_appearances, lemma_age_appearances
    else:
        return returnable


def formatDataForPaperOutputBasic(corpus: dict[str,pd.DataFrame]):
    """
    Function which takes in a corpus and provides four sets of dictionaries as sets of csv-files:
    1. contains data for exact ages as subcorpora
    2. contains data for age groups as subcorpora
    3. contains data for registers as subcorpora
    4. contains data for the whole corpus
    """
    ages = sorted(getAvailableAges(corpus))

    ready_dfs_ages = {}
    ready_dfs_groups = {}
    ready_dfs_whole = {}

    #Subcorpora based on the target age groups
    sub_corpora = []
    #Combine books aged 15 and up into one sub-corpus as there are very few entries in 16,17,18
    over_15 = []
    for i in ages:
        if i<15:
            sub_corpora.append(cleanWords(getDistinctSubCorp(corpus, i)))
        else:
            over_15.append(cleanWords(getDistinctSubCorp(corpus, i)))
    #Sort the aged 15 and over sub-corpora from lowest age to highest
    over_15.sort(key=lambda x:int(findAgeFromID(list(x.keys())[0])))
    #Combine 15+ aged books into one sub-corpus
    sub_corpora.append(combineSubCorpDicts(over_15))
    #Sort the sub-corpora from lowest age to highest
    sub_corpora.sort(key=lambda x:int(findAgeFromID(list(x.keys())[0])))
    #Keep track of when words and lemmas first appear in terms of intended reading age
    ready_dfs_ages, word_age_appearances, lemma_age_appearances = getStatisticsForDatabaseOnlyPos(sub_corpora)

    writePaperOutputCsv(ready_dfs_ages, 'ages_csv')
    print("Ages outputted!")
    #Define age group sub-corpora

    
    #Generate correct keys/ids
    group_1 = [5,6,7,8]
    group_2 = [9,10,11,12]
    group_3 = ages[ages.index(13):]
    #Distinct subcorpora
    sub_corp_1= combineSubCorpDicts([getDistinctSubCorp(corpus, x) for x in group_1])
    sub_corp_2= combineSubCorpDicts([getDistinctSubCorp(corpus, x) for x in group_2])
    sub_corp_3= combineSubCorpDicts([getDistinctSubCorp(corpus, x) for x in group_3])
    sub_corps = dict(zip(['7-8','9-12','13+'],[sub_corp_1, sub_corp_2, sub_corp_3]))

    ready_dfs_groups = getStatisticsForDatabaseOnlyPos(sub_corps, word_age_appearances, lemma_age_appearances)

    writePaperOutputCsv(ready_dfs_groups, 'groups_csv')
    print("Groups done!")

    print("Start registers")
    #Work with registers
    reg1 = {key:corpus[key] for key in corpus if key[-1]=='1'}
    reg2 = {key:corpus[key] for key in corpus if key[-1]=='2'}
    reg3 = {key:corpus[key] for key in corpus if key[-1]=='3'}

    ready_dfs_registers = {}

    sub_corps = dict(zip(['Fiction','Nonfiction','Textbook'],[reg1, reg2, reg3]))

    ready_dfs_registers = getStatisticsForDatabaseOnlyPos(sub_corps, word_age_appearances, lemma_age_appearances)

    writePaperOutputCsv(ready_dfs_registers, 'genres_csv')
    print("Registers done!")

    temp_whole = {"Whole":corpus}
    ready_dfs_whole = getStatisticsForDatabaseOnlyPos(temp_whole, word_age_appearances, lemma_age_appearances)

    writePaperOutputCsv(ready_dfs_whole, 'whole_csv')
    
    print("All done!!")
    #return ready_dfs_ages, ready_dfs_groups, ready_dfs_registers, ready_dfs_whole

def formatDataForPaperOutputWithFeats(corpus: dict[str,pd.DataFrame]):
    """
    Function which takes in a corpus and provides four sets of dictionaries as sets of csv-files:
    1. contains data for exact ages as subcorpora
    2. contains data for age groups as subcorpora
    3. contains data for registers as subcorpora
    4. contains data for the whole corpus
    """
    ages = sorted(getAvailableAges(corpus))

    ready_dfs_ages = {}
    ready_dfs_groups = {}
    ready_dfs_whole = {}

    #Subcorpora based on the target age groups
    sub_corpora = []
    #Combine books aged 15 and up into one sub-corpus as there are very few entries in 16,17,18
    over_15 = []
    for i in ages:
        if i<15:
            sub_corpora.append(cleanWords(getDistinctSubCorp(corpus, i)))
        else:
            over_15.append(cleanWords(getDistinctSubCorp(corpus, i)))
    #Sort the aged 15 and over sub-corpora from lowest age to highest
    over_15.sort(key=lambda x:int(findAgeFromID(list(x.keys())[0])))
    #Combine 15+ aged books into one sub-corpus
    sub_corpora.append(combineSubCorpDicts(over_15))
    #Sort the sub-corpora from lowest age to highest
    sub_corpora.sort(key=lambda x:int(findAgeFromID(list(x.keys())[0])))
    #Keep track of when words and lemmas first appear in terms of intended reading age
    ready_dfs_ages, word_age_appearances, lemma_age_appearances = getStatisticsForDatabasePosFeats(sub_corpora)

    writePaperOutputCsv(ready_dfs_ages, 'ages_with_features_csv')
    print("Ages outputted!")
    #Define age group sub-corpora

    
    #Generate correct keys/ids
    group_1 = [5,6,7,8]
    group_2 = [9,10,11,12]
    group_3 = ages[ages.index(13):]
    #Distinct subcorpora
    sub_corp_1= combineSubCorpDicts([getDistinctSubCorp(corpus, x) for x in group_1])
    sub_corp_2= combineSubCorpDicts([getDistinctSubCorp(corpus, x) for x in group_2])
    sub_corp_3= combineSubCorpDicts([getDistinctSubCorp(corpus, x) for x in group_3])
    sub_corps = dict(zip(['7-8','9-12','13+'],[sub_corp_1, sub_corp_2, sub_corp_3]))

    ready_dfs_groups = getStatisticsForDatabasePosFeats(sub_corps, word_age_appearances, lemma_age_appearances)

    writePaperOutputCsv(ready_dfs_groups, 'groups_with_features_csv')
    print("Groups done!")

    print("Start registers")
    #Work with registers
    reg1 = {key:corpus[key] for key in corpus if key[-1]=='1'}
    reg2 = {key:corpus[key] for key in corpus if key[-1]=='2'}
    reg3 = {key:corpus[key] for key in corpus if key[-1]=='3'}

    ready_dfs_registers = {}

    sub_corps = dict(zip(['Fiction','Nonfiction','Textbook'],[reg1, reg2, reg3]))

    ready_dfs_registers = getStatisticsForDatabasePosFeats(sub_corps, word_age_appearances, lemma_age_appearances)

    writePaperOutputCsv(ready_dfs_registers, 'genres_with_features_csv')
    print("Registers done!")

    temp_whole = {"Whole":corpus}
    ready_dfs_whole = getStatisticsForDatabasePosFeats(temp_whole, word_age_appearances, lemma_age_appearances)

    writePaperOutputCsv(ready_dfs_whole, 'whole_with_features_csv')
    
    print("All done!!")
    #return ready_dfs_ages, ready_dfs_groups, ready_dfs_registers, ready_dfs_whole
        
def writePaperOutputXlsx(ready_dfs: dict[str:pd.DataFrame], name: str):
    """
    Simple function for writing xslx-files based on a list of dictionaries
    Name is the name of the xlsx-file
    """
    with pd.ExcelWriter("Data/TCBLex_data_output_"+name+".xlsx") as writer:
        for df in ready_dfs:
            ready_dfs[df].to_excel(writer, sheet_name=df, index=False)
            print(df+" done!")

def writePaperOutputCsv(ready_dfs: dict[str:pd.DataFrame], name: str):
    """
    Simple function for writing csv-files based on a list of dictionaries
    Name is the name of the folder containing the csv-files
    """
    path = "Data/"+name
    if not os.path.exists(path):
        os.mkdir(path)
    for df in ready_dfs:
        name = path+"/"+str(df)+".csv"
        ready_dfs[df].to_csv(name, index=False, sep=';')
        print(df+" done!")