#Imports
from scripts import bookdatafunctions as bdf
from scripts import corpusMLfunctions as cmf
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import os
import sys
import multiprocessing as mp

#Constants
AGE_SHEET = "ISBN_MAPS/ISBN2AGE.xlsx"
AUTH_SHEET = "ISBN_MAPS/ISBN2AUTH.xlsx"
CONLLUS_FOLDER = "Conllus"
SNIPPET_LENS = [5,10,25,50,75,100]


def performSnipping():

    #Load corpus
    corpus = bdf.mapGroup2Age(bdf.maskPropn(bdf.initBooksFromConllus(CONLLUS_FOLDER)), AGE_SHEET)
    pool = mp.Pool(mp.cpu_count())
    pbar = tqdm(total=len(corpus))
    def update(*a):
     pbar.update(1)
    #Perform the splitting for each book asynchronously, as it doesn't matter in which order the books get done
    for key in corpus:
        df = corpus[key]
        pool.apply_async(cmf.splitBooksToSnippets, [key, df, SNIPPET_LENS, "SnippetDatasets/"], callback=update)
        
    #print("All running!")
    pool.close()
    #print("Pool closed!")
    pool.join()
    #print("Waiting done!")

#Main function
def main():
    performSnipping()
#Launch main
if __name__ == "__main__":
    main()
