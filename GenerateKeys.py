# %%
#Imports
from scripts import bookdatafunctions as bdf
from scripts import corpusMLfunctions as cmf
import json

# %%
#Constants
AGE_SHEET = "ISBN_MAPS/ISBN2AGE.xlsx"
AUTH_SHEET = "ISBN_MAPS/ISBN2AUTH.xlsx"
CONLLUS_FOLDER = "Conllus"
SNIPPET_LENS = [5,10,25,50,75,100]

# %%
#Load corpus
corpus = bdf.mapGroup2Age(bdf.maskPropn(bdf.initBooksFromConllus(CONLLUS_FOLDER)), AGE_SHEET)

# %%
#Generate train-test-eval split for keys
author_level_split = cmf.splitOnAuthorLevel(list(corpus.keys()), AUTH_SHEET)
train_target, test_target, eval_target = cmf.generateAgeStratificationAmounts(corpus, 0.7)
print(train_target)
train_keys, test_keys, eval_keys = cmf.doTrainTestEvalSplitSeriesLevel(author_level_split, train_target, test_target, eval_target)

# %%
trainkeys_straps = []
testkeys_straps = []
evalkeys_straps = []

while len(trainkeys_straps) != 100:
    train_keys, test_keys, eval_keys = cmf.doTrainTestEvalSplitSeriesLevel(author_level_split, train_target, test_target, eval_target)
    if len(train_keys) == 210:
        print(len(trainkeys_straps))
        trainkeys_straps.append(train_keys)
        testkeys_straps.append(test_keys)
        evalkeys_straps.append(eval_keys)

# %%
counter = 0
for x in trainkeys_straps:
    for y in trainkeys_straps:
        if x==y:
            counter += 1
print(counter)

# %%
temp_dict = []
for i in range(len(trainkeys_straps)):
    temp_dict.append({'id':i, 'train_keys':trainkeys_straps[i], 'eval_keys':evalkeys_straps[i], 'test_keys':testkeys_straps[i]})
with open("Keylists.jsonl", 'w') as f:
    f.write('\n'.join(map(json.dumps, temp_dict)))


