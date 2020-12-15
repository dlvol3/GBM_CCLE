######


#%%
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
#%%

# import all the data

# RNA COAD

coadrna_raw = pd.read_table("C:\\Users\\Yue\\PycharmProjects\\data\\coad\\HiSeqV2")
coadrna_t = coadrna_raw.T

# reassign column and row names

coadrna_t.columns = coadrna_t.iloc[0,:]

coadrna_t["sampleID"] =coadrna_t.index   # will be removed later?> store in sample names

coadrna_t = coadrna_t.iloc[1::,::]

# COAD phenotypes

coadpheno = pd.read_table("C:\\Users\\Yue\\PycharmProjects\\data\\coad\\COAD_clinicalMatrix")

coadhisto = coadpheno.loc[:, ["histological_type", "sampleID"]]
coadhisto = coadhisto.loc[(coadhisto["histological_type"].notnull(),)]
coadhisto = coadhisto.loc[(coadhisto["histological_type"] != '[Discrepancy]',)]

#%% merge these two into one file

coad = coadhisto.merge(coadrna_t, on='sampleID', how='left')
# coad.shape

coad = coad.dropna()
coad = coad.reset_index()    # Get the sample ID mapping to the index here
coad_train = coad.drop(["index","sampleID"],axis=1)

#%% model training

# -------------------------###
# Get familiar with python DS
# ref: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

# Label the class
# le = LabelEncoder()
# le_count = 0
#
# # iterate through columns
# for col in coad_train:
#     if coad_train.loc[:, col].dtype == 'object':
#         # if less than 2 classes(which is better to use one-hot coding if not)
#         if len(list(coad_train.loc[:, col].unique())) <= 2:
#             # 'train' the label encoder with the training data
#             le.fit(coad_train.loc[:, col])
#             # Transform both training and testing
#             coad_train.loc[:, col] = le.transform(coad_train.loc[:, col])
#             # pdC.loc[:, col] = le.transform(pdC.loc[:, col])
#
#             # Keep track of how many columns were labeled
#             le_count += 1
#
# print('%d columns were label encoded.' % le_count)


#%%  as many columns also have less than 3 unique values we have to assign hot code by hand
le = LabelEncoder()
le.fit(coad_train.loc[:, 'histological_type'])
coad_train.loc[:, 'histological_type'] = le.transform(coad_train.loc[:, 'histological_type'])

# %%
# Exploratory Data Analysis(EDA)

# Distribution of the target classes(columns)
coad_train['histological_type'].value_counts()
coad_train['histological_type'].head(4)

coad_train['histological_type'].plot.hist()
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

Missing_values = missing_value_table(coad_train)
Missing_values.head(10)
# %%
# Column Types
# Number of each type of column
coad_train.dtypes.value_counts()

# Check the number of the unique classes in each object column
coad_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)

# %%
# Correlations
# correlations = coad_train.corr()['histological_type']
#
# # Display correlations
# print('Most Positive Correlations:\n', correlations.tail(15))
# print('\nMost Negative Correlations:\n', correlations.head(15))

# Create Cross-validation and training/testing


# %%
# Random forest 1st

# Define the RF
random_forest = RandomForestClassifier(n_estimators=200, random_state=123, max_features="sqrt",
                                       criterion="gini", oob_score=True, n_jobs=10, max_depth=9,
                                       verbose=0)
# %%
# Drop SENRES

train_labels = coad_train.loc[:, "histological_type"]
#
#
if 'histological_type' in coad_train.columns:
    train = coad_train.drop(['histological_type'], axis=1)
else:
    train = coad_train.copy()
# # train.iloc[0:3,0:3]
features = list(train.columns)
# train["SENRES"] = train_labels
random_forest.fit(train, train_labels)
#%% Undersampling
# Class count
count_class_0, count_class_1 = coad_train.histological_type.value_counts()

# Divide by class
df_class_0 = coad_train[coad_train['histological_type'] == 0]
df_class_1 = coad_train[coad_train['histological_type'] == 1]

df_class_0_under = df_class_0.sample(count_class_1*2)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
# s_train_labels = df_test_under.loc[:, "histological_type"]



#%% Test a leave one out frame

# test with 4 samples
#train = train.iloc[[0,1,2,3],:]
#train_labels = train_labels[:4]
top100 = pd.DataFrame()
m = 0
for i in df_test_under.index:
    train_labels = df_test_under.loc[:, "histological_type"].copy()
    if 'histological_type' in df_test_under.columns:
        train = df_test_under.drop(['histological_type'], axis=1)
    else:
        train = df_test_under.copy()
    features = list(train.columns)

    train_loo = train.drop(index=i, axis=0)
    test_loo = train.iloc[[m],:]
    train_labels_loo = train_labels.drop(index=i)
    test_labels_loo = train_labels.pop(i)
    test_labels_loo = pd.Series(test_labels_loo, index=[i])

    random_forest.fit(train_loo, train_labels_loo)
    print(random_forest.oob_score_)
    testP = random_forest.predict(test_loo)
    featurelist = train_loo.columns.values.tolist()
    index = list(range(len(featurelist)))
    if testP == test_labels_loo[i]:
        # RUN NERF
        pd_ff = flatforest(random_forest, test_loo)
        pd_f = extarget(random_forest, test_loo, pd_ff)
        pd_nt = nerftab(pd_f)

        g_localnerf = localnerf(pd_nt, 0)
        g_twonets = twonets(g_localnerf, str('coad_index_') + str(i), index, featurelist, index1=6, index2=6)

        # pagerank
        g_pagerank = g_localnerf.replace(index, featurelist)
        xg_pagerank = nx.from_pandas_edgelist(g_pagerank, "feature_i", "feature_j", "EI")
        DG = xg_pagerank.to_directed()
        rktest = nx.pagerank(DG, weight='EI')
        rkdata = pd.Series(rktest, name='position')
        rkdata.index.name = 'PR'
        rkrank = sorted(rktest, key=rktest.get, reverse=True)
        fea_corr = rkrank[0:100]
        top100[i] = fea_corr
        # rkrank = [featurelist[i] for i in rkrank]
        fn = "pagerank_sample_" + str(i) + ".txt"
        with open(os.getcwd() + '/output/pagerank/' + fn, 'w') as f:
            for item in rkrank:
                f.write("%s\n" % item)
    m = m + 1
#%%

# RF 1st train 5 trees

random_forest.fit(train, train_labels)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
feature_importances.to_csv("feature_importances.csv")
#
# feature_importances
# train.shape
# # Make predictions on the test data
# test_labels = lapaG.loc[:, "SENRES"]
# cell_lines_lapaG = lapaG.loc[:, "gdsc.name"]
#
# if 'SENRES' in lapaG.columns:
#     test = lapaG.drop(['SENRES'], axis=1)
# else:
#     test = lapaG.copy()
#
# test = test.drop(['gdsc.name'], axis=1)
# predictions = random_forest.predict(test)
# predictions
#
# confusion_matrix(test_labels, predictions)


#%%
# Pairwise RBO
# remove duplicates

# top50 = top50.iloc[:, np.r_[0:16,17,18,21]]

 rbo(top100.iloc[:, 0], top100.iloc[:, 1], p=0.9)['ext']
 corr_rbo_matrix = pd.DataFrame(np.zeros((top100.shape[1], top100.shape[1])))
 for i in range(top100.shape[1]):
     for j in range(top100.shape[1]):
         corr_rbo = rbo(top100.iloc[:, i], top100.iloc[:, j], p=0.9)
         corr_rbo_matrix[i][j] = corr_rbo['ext']
         corr_rbo_matrix[j][i] = corr_rbo['ext']
# # rrrrr = rbo(xaa,yaa,p=0.9)
print(corr_rbo_matrix)


#%% downstream
#
colon_bg = pd.DataFrame({
    'index_in_analysis': list(range(len(colon_cl))),
    'sample_name': colon_cl

})
colon_bg.to_csv("colon_bg.txt")
dic = dict(zip(colon_bg.index_in_analysis, colon_bg.sample_name))
corr_rbo_matrix.rename(index = dic, columns = dic)
# corr_withname = corr_rbo_matrix.index.replace(colon_bg.iloc[:,0],colon_bg.iloc[:,1])
# #%% plot heatmap

plt.figure(figsize=(24,20))
sns.clustermap(
    corr_rbo_matrix,
    cmap='YlGnBu',
    # annot=True,
    linewidths=2
)
plt.show()
# %% Separate corr matrix into two classes

df_test_under_class0_index = df_class_0_under.index
df_test_under_class1_index = df_class_1.index

top100_class00 = top100[[iii for iii in df_test_under_class0_index if top100.columns.to_list().count(iii)>0]]
top100_class11 = top100[[iii for iii in df_test_under_class1_index if top100.columns.to_list().count(iii)>0]]

# top 100 all has 86 samples only 12 are class 1 so we focus on class 0
#mapping the sample ID back to the matrix
colon_bg = pd.DataFrame({
    'index_in_analysis': coad.index,
    'sample_name': coad['sampleID']

})
colon_bg.to_csv("colon_bg.txt")
dic = dict(zip(colon_bg.index_in_analysis, colon_bg.sample_name))
top100_class00 = top100_class00.rename(columns = dic)

top100_class00.to_csv("top100_class00.csv")
top100_class00 = pd.read_csv("top100_class00.csv", index_col=[0])
# pairwaise RBO for class 0 74 samples
#%%
# Pairwise RBO
# remove duplicates

# top50 = top50.iloc[:, np.r_[0:16,17,18,21]]

 rbo(top100_class00.iloc[:, 0], top100_class00.iloc[:, 1], p=0.9)['ext']
 corr_rbo_matrix = pd.DataFrame(np.zeros((top100_class00.shape[1], top100_class00.shape[1])))
 for i in range(top100_class00.shape[1]):
     for j in range(top100_class00.shape[1]):
         corr_rbo = rbo(top100_class00.iloc[:, i], top100_class00.iloc[:, j], p=0.9)
         corr_rbo_matrix[i][j] = corr_rbo['ext']
         corr_rbo_matrix[j][i] = corr_rbo['ext']
# # rrrrr = rbo(xaa,yaa,p=0.9)
print(corr_rbo_matrix)
colon_bg_corr = pd.DataFrame({
    'index_in_analysis': list(range(len(top100_class00.columns))),
    'sample_name': top100_class00.columns

})
dic = dict(zip(colon_bg_corr.index_in_analysis, colon_bg_corr.sample_name))
corr_rbo_matrix = corr_rbo_matrix.rename(columns = dic)
corr_rbo_matrix.to_csv("top100_class0_corr.csv")

#corr_rbo_matrix = pd.read_csv("top100_class0_corr.csv")
#%% return clusters
import scipy.cluster.hierarchy as sch

d = sch.distance.pdist(corr_rbo_matrix)
L = sch.linkage(d, method='complete')
# 0.2 can be modified to retrieve more stringent or relaxed clusters
clusters = sch.fcluster(L, 0.7*d.max(), 'distance')

# clusters indicices correspond to incides of original df
for i,cluster in enumerate(clusters):
    print(corr_rbo_matrix.index[i], cluster)

colon_clusters = pd.DataFrame({
    'index_in_analysis': list(range(len(top100_class00.columns))),
    'sample_name': top100_class00.columns,
    'clusters': clusters

})

colon_clusters.to_csv("colon_clusters_samplename_20200915.csv", index = False)
#%% plot heatmap

lut = dict(zip(set(clusters),sns.hls_palette(len(set(clusters)))))
row_colors = pd.DataFrame(clusters)[0].map(lut)

plt.figure(figsize=(48,39))
sns.clustermap(
    corr_rbo_matrix,
    cmap='YlGnBu',
    row_colors=row_colors,
    # annot=True,
    linewidths=2

)
plt.savefig("corr_74class0_20200915.png")

plt.show()

# pd.DataFrame(clusters)[0].map(lut)

#%% average rank of genes in samples in cluster 2 and 5

colon_cluster_25 = colon_clusters.loc[(colon_clusters['clusters'] == 2) | (colon_clusters['clusters'] == 5) , "sample_name":"clusters"].copy()
top100_class00_rank_g2_5 = top100_class00[colon_cluster_25['sample_name']].copy()

# TODO to calculate the average rank of each gene in one group >100 = rank100, this can be wrap up into a new function

#%% 2020Oct test on TCGA-QG-A5YX-01 c5orf46
test_loo = coad.loc[coad['sampleID'] == "TCGA-A6-5662-01",:]
coad_train = coad.drop(["index","sampleID"],axis=1)


pd_ff = flatforest(random_forest, test_loo)
pd_f = extarget(random_forest, test_loo, pd_ff)
pd_nt = nerftab(pd_f)

g_localnerf = localnerf(pd_nt, 0)
g_twonets = twonets(g_localnerf, str('coad_index_') + str(i), index, featurelist, index1=6, index2=6)

# pagerank
g_pagerank = g_localnerf.replace(index, featurelist)
xg_pagerank = nx.from_pandas_edgelist(g_pagerank, "feature_i", "feature_j", "EI")
DG = xg_pagerank.to_directed()
rktest = nx.pagerank(DG, weight='EI')
rkdata = pd.Series(rktest, name='position')
rkdata.index.name = 'PR'
rkrank = sorted(rktest, key=rktest.get, reverse=True)
fea_corr = rkrank[0:100]
top100[i] = fea_corr
        # rkrank = [featurelist[i] for i in rkrank]
fn = "pagerank_sample_" + str(i) + ".txt"
with open(os.getcwd() + '/output/pagerank/' + fn, 'w') as f:
    for item in rkrank:
        f.write("%s\n" % item)