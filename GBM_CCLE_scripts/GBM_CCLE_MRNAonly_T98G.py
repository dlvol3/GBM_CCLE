# Using gdsc exp data -> TMZ resistance in GBM cell lines
# NERF V0.3
# Yue Zhang <yue.zhang@lih.lu>

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
import platform
import time
import mygene
from sklearn.utils.multiclass import unique_labels
import os
import networkx as nx
import pickle
from scipy import stats
import re


#%% GBM cell lines in GDSC

gdsc = pd.read_csv('C:/Users/Yue/PycharmProjects/data/GBM/readyforpython_sub.csv')
gdsc = gdsc.rename(columns = {'Cancer.Type..matching.TCGA.label.':'cancertype'})
gbmindex = gdsc.loc[gdsc.cancertype == 'GBM', gdsc.columns[0:3]].index.tolist()
gbmindex
gbmname = gdsc.loc[gbmindex, 'Sample.Name']

gdsc = gdsc.iloc[0:201,]

# join GDSC exp data with miRNA

allinone = gdsc
# Create list for subset
flist = list(range(2, (len(allinone.columns)), 1))
# ciLapa.insert(0, 1)

# subset two sets
tmz= allinone.iloc[:, flist]   # Almost ready, need to remove the unmatched NAs

tmz_ready = tmz.dropna(axis = 0) # 110 samples left ready

# Check GBM cell lines left

GBM_test = tmz_ready.loc[gbmindex]

#%% model training

# -------------------------###
# Get familiar with python DS
# ref: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

# Label the class
le = LabelEncoder()
le_count = 0

# iterate through columns
for col in tmz_ready:
    if tmz_ready.loc[:, col].dtype == 'object':
        # if less than 2 classes(which is better to use one-hot coding if not)
        if len(list(tmz_ready.loc[:, col].unique())) <= 2:
            # 'train' the label encoder with the training data
            le.fit(tmz_ready.loc[:, col])
            # Transform both training and testing
            tmz_ready.loc[:, col] = le.transform(tmz_ready.loc[:, col])
            # pdC.loc[:, col] = le.transform(pdC.loc[:, col])

            # Keep track of how many columns were labeled
            le_count += 1

print('%d columns were label encoded.' % le_count)

# %%
# Exploratory Data Analysis(EDA)

# Distribution of the target classes(columns)
tmz_ready['Temozolomide'].value_counts()
tmz_ready['Temozolomide'].head(4)

tmz_ready['Temozolomide'].plot.hist()
plt.show()

# %%
# Examine Missing values
def missing_value_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing values', 1: '% of Total Values'}
    )

    # Sort the table by percentage of the missing values
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print the summary
    print("Your selected data frame has " + str(df.shape[1]) + " columns.\n"
                                                               "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the result
    return mis_val_table_ren_columns


# Check the missing value in the dataset

Missing_values = missing_value_table(tmz_ready)
Missing_values.head(10)

# %%
# Column Types
# Number of each type of column
tmz_ready.dtypes.value_counts()

# Check the number of the unique classes in each object column
tmz_ready.select_dtypes('object').apply(pd.Series.nunique, axis=0)

# %%
# Random forest 1st

# Define the RF
random_forest = RandomForestClassifier(n_estimators=600, random_state=49, max_features="sqrt",
                                       criterion="gini", oob_score=True, n_jobs=10, max_depth=12,
                                       verbose=0)

#%%
# Drop SENRES

tmz_ready_label_all = tmz_ready.loc[:, "Temozolomide"]
test_98G_U251 = tmz_ready.iloc[[84,87],:]
train_tmz = tmz_ready.drop([84,87],axis = 0)

train_labels = train_tmz.loc[:, "Temozolomide"]
test_98G_U251_label = test_98G_U251.loc[:, "Temozolomide"]
#
#
if 'Temozolomide' in train_tmz.columns:
    train = train_tmz.drop(['Temozolomide'], axis=1)
else:
    train = train_tmz.copy()
# # train.iloc[0:3,0:3]
if 'Temozolomide' in test_98G_U251.columns:
    test = test_98G_U251.drop(['Temozolomide'], axis=1)
else:
    test = test_98G_U251.copy()

features = list(train.columns)
# train["SENRES"] = train_labels


random_forest.fit(train, train_labels)
print(random_forest.oob_score_)

testP = random_forest.predict(test)


# featurelist = train.columns.values.tolist()
# index = list(range(len(featurelist)))
#%%
# Create feature list, convert ENSG into gene symbols
featurelist = train.columns.values.tolist()
# Mygene convertion
mg = mygene.MyGeneInfo()
mg.metadata('available_fields')
con = mg.querymany(featurelist, scopes='ensembl.gene', fields='symbol', species="human", as_dataframe=True)
# replace Nan unmapped with original ENSGZ
con['symbol'] = np.where(con['notfound'] == True, con.index.values, con['symbol'])

featurelist_g = con.iloc[:, 2].reset_index()
feag = featurelist_g.iloc[:, 1]
# featurelist_g.loc[featurelist_g['query'] == 'ENSG00000229425'].index[0]


# POP out those duplicates
feag = list(feag)

index = list(range(len(featurelist)))

sl = random_forest.feature_importances_
fl = pd.DataFrame({
    'feature_name': feag,
    'score': sl,
    'index': index
})


fls = fl.sort_values('score', ascending=False)
fls.to_csv("feature_importances_mRNAonly.csv")
#%%
# feature_importance_values = random_forest.feature_importances_
# feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
# feature_importances.to_csv("feature_importances.csv")
#%% NERF for T98G
pd_ff = flatforest(random_forest, test)
pd_f = extarget(random_forest, test, pd_ff)
pd_nt = nerftab(pd_f)

t98g_localnerf = localnerf(pd_nt, 1)
t98g_g_twonets = twonets(t98g_localnerf, str('T98G_mRNAonly'), index, feag,index1=8, index2 = 8)

#%% BRO ranking
g_pagerank = t98g_localnerf.replace(index, feag)
xg_pagerank = nx.from_pandas_edgelist(g_pagerank, "feature_i", "feature_j", "EI")
DG = xg_pagerank.to_directed()
rktest = nx.pagerank(DG, weight='EI')
rkdata = pd.Series(rktest, name='position')
rkdata.index.name = 'PR'
rkrank = sorted(rktest, key=rktest.get, reverse=True)
fea_corr = rkrank[0:100]
# top100[i] = fea_corr
# rkrank = [featurelist[i] for i in rkrank]
fn = "pagerank_sample_T98G_mRNAonly.txt"
with open(os.getcwd() + '/output/pagerank/' + fn, 'w') as f:
    for item in rkrank:
        f.write("%s\n" % item)
#%%
import pickle
object = pd_ff
filehandler = open('pd_ff_t98Gmrna_index1', 'wb')
pickle.dump(object, filehandler)

object = pd_f
filehandler = open('pd_f_t98Gmrna_index1', 'wb')
pickle.dump(object, filehandler)