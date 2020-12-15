# Data preprocessing in R
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
#%%


# Code for RF (TBE)

#%%

gdsc = pd.read_csv('P:/VM/GDSC_GBM/readyforpython_sub.csv')
gdsc = gdsc.rename(columns = {'Cancer.Type..matching.TCGA.label.':'cancertype'})
gbmindex = gdsc.loc[gdsc.cancertype == 'GBM', gdsc.columns[0:3]].index.tolist()
gbmindex
gbmname = gdsc.loc[gbmindex, 'Sample.Name']
# Create list for subset
flist = list(range(2, len(gdsc.columns), 1))
# ciLapa.insert(0, 1)

# subset two sets
lapaC = gdsc.iloc[:, flist]
lapaC.head(1)

#%%
# -------------------------###
# Get familiar with python DS
# ref: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

# Label the class
le = LabelEncoder()
le_count = 0

# iterate through columns
for col in lapaC:
    if lapaC.loc[:, col].dtype == 'object':
        # if less than 2 classes(which is better to use one-hot coding if not)
        if len(list(lapaC.loc[:, col].unique())) <= 2:
            # 'train' the label encoder with the training data
            le.fit(lapaC.loc[:, col])
            # Transform both training and testing
            lapaC.loc[:, col] = le.transform(lapaC.loc[:, col])

            # Keep track of how many columns were labeled
            le_count += 1

print('%d columns were label encoded.' % le_count)

#%%
# Exploratory Data Analysis(EDA)

# Distribution of the target classes(columns)
lapaC['Temozolomide'].value_counts()
lapaC['Temozolomide'].head(4)
lapaC['Temozolomide'].plot.hist()
plt.show()

#%%
# Examine Missing values
def missing_value_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum()/len(df)

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

Missing_values = missing_value_table(lapaC)
Missing_values.head(10)

#%% branch: check the correlation status of ERBB2


# erbb2c = lapaC.iloc[:, 2:(lapaC.shape[1]-1)].corr()
# print('Most Positive Correlations:\n', erbb2c["ENSG00000141736"].tail(15))
# print('\nMost Negative Correlations:\n', erbb2c["ENSG00000141736"].head(15))




#%%
# Correlations
correlations = lapaC.iloc[:, 0:200].corr()['Temozolomide'].sort_values(na_position='first')

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))
# Create Cross-validation and training/testing


#%%
# Random forest 1st

# Define the RF
random_forest_gbm = RandomForestClassifier(n_estimators=500, random_state=123, max_features="sqrt",
                                       criterion="gini", oob_score=True, n_jobs=12, max_depth=8,
                                       verbose=0)
#%%
# Drop SENRES

train_labels = lapaC.loc[:, "Temozolomide"]
# cell_lines_lapaC = gdsc.loc[:, "Sample.Name"]
# lapaC = lapaC.drop(['Sample.Name'], axis=1)

if 'Temozolomide' in lapaC.columns:
    train = lapaC.drop(['Temozolomide'], axis=1)
else:
    train = lapaC.copy()
train.iloc[0:3,0:3]
features = list(train.columns)
# train["SENRES"] = train_labels



#%%

# RF 1st train 5 trees

random_forest_gbm.fit(train, train_labels)

# Extract feature importances
feature_importance_values = random_forest_gbm.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
feature_importances
train.shape
# Make predictions on the test data

"""
test_labels = lapaG.loc[:, "SENRES"]
cell_lines_lapaG = lapaG.loc[:, "gdsc.name"]

if 'SENRES' in lapaG.columns:
    test = lapaG.drop(['SENRES'], axis=1)
else:
    test = lapaG.copy()

test = test.drop(['gdsc.name'], axis=1)
predictions = random_forest.predict(test)
predictions

confusion_matrix(test_labels, predictions)

"""
#%%
random_forest_gbm.oob_score_



#%%
testygbm = train.iloc[gbmindex,:]
resultgbm = random_forest_gbm.predict(testygbm)

gbm_ref_label = train_labels[gbmindex].values

# cfm = confusion_matrix(train_labels, ret)
cfm = confusion_matrix(gbm_ref_label, resultgbm)

cfm



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
featurelist_g.loc[featurelist_g['query'] == 'ENSG00000229425'].index[0]


feag.pop(47080)

# POP out those duplicates
feag = list(feag)

index = list(range(len(featurelist)))
#
# sl = random_forest_gbm.feature_importances_
# fl = pd.DataFrame({
#     'feature_name': feag,
#     'score': sl,
#     'index': index
# })
#
#
# fls = fl.sort_values('score', ascending=False)
# fls
#%%
# testy2 = testygbm.iloc[[2, 3, 4, 5]]
testy2 = testygbm
TIE2_f = flatforest(random_forest_gbm, testy2)
TIE2 = extarget(random_forest_gbm, testy2, TIE2_f)
# nt_gbmf4 = nerftab(TIE2)
nt_gbmall = nerftab(TIE2)


# extract all 39 gbm cell lines
#%%
top50 = pd.DataFrame()

for sampleindex in range(testy2.shape[0]):
#for sampleindex in range(5):
    g_localnerf = localnerf(nt_gbmall, sampleindex)
    g_twonets = twonets(g_localnerf, str(sampleindex), index, feag, index1=5, index2=13)
    # pagerank
    g_pagerank = g_localnerf.replace(index, feag)
    xg_pagerank = nx.from_pandas_edgelist(g_pagerank, "feature_i", "feature_j", "EI")
    DG = xg_pagerank.to_directed()
    rktest = nx.pagerank(DG, weight='EI')
    rkdata = pd.Series(rktest, name='position')
    rkdata.index.name = 'PR'
    rkdata
    rkrank = sorted(rktest, key=rktest.get, reverse=True)
    fea_corr = rkrank[0:50]
    top50[sampleindex] = fea_corr
    # rkrank = [featurelist[i] for i in rkrank]
    fn = "pagerank_sample_" + str(sampleindex) + ".txt"
    with open('C:\\Yue\\pycprojects\\NERF-RF_interpreter\\result2019aug\\' + fn, 'w') as f:
        for item in rkrank:
            f.write("%s\n" % item)



#%%
# Pairwise RBO
# remove duplicates

#top50 = top50.iloc[:, 0:34]

rbo(top50.iloc[:, 0], top50.iloc[:, 1], p=0.9)['ext']
corr_rbo_matrix = pd.DataFrame(np.zeros((top50.shape[1], top50.shape[1])))
for i in range(top50.shape[1]):
    for j in range(top50.shape[1]):
        corr_rbo = rbo(top50.iloc[:, i], top50.iloc[:, j], p=0.9)
        corr_rbo_matrix[i][j] = corr_rbo['ext']
        corr_rbo_matrix[j][i] = corr_rbo['ext']
# rrrrr = rbo(xaa,yaa,p=0.9)
print(corr_rbo_matrix)


#%% downstream

gbmbg = pd.DataFrame({
    'index_in_analysis': list(range(len(gbmindex))),
    'sample_name': gbmname

})
gbmbg.to_csv("gbmbg.txt")
dic = dict(zip(gbmbg.index_in_analysis, gbmbg.sample_name))
corr_rbo_matrix.rename(index = dic, columns = dic)
corr_withname = corr_rbo_matrix.index.replace(gbmbg.iloc[:,0],gbmbg.iloc[:,1])
#%% plot heatmap

plt.figure(figsize=(24,20))
sns.clustermap(
    corr_rbo_matrix,
    cmap='YlGnBu',
    # annot=True,
    linewidths=2
)
plt.show()
#%%
pickle.dump(nt_gbmall, open('C:\\Yue\\pycprojects\\NERF-RF_interpreter\\result2019aug\\nt_gbmall_dump', 'wb'))

# By loading this, there is no need to run flatforest and extarget again
# Save current session gbmall file => after

#%% SEN and RES
g2 = localnerf(nt_gbmf4, 0)
g3 = localnerf(nt_gbmf4, 1)
g4 = localnerf(nt_gbmf4, 2)
g5 = localnerf(nt_gbmf4, 3)

LN405 = twonets(g2, "LN405_R_500", index, feag, index1=5, index2=12)
LN229 = twonets(g3, "LN229_S_500", index, feag, index1=5, index2=12)
LN18 = twonets(g4, "LN18_R_500", index, feag, index1=5, index2=12)
KS1 = twonets(g5, "KS1_S_500", index, feag, index1=5, index2=12)


feature_importance_values = random_forest_gbm.feature_importances_
feature_importances = pd.DataFrame({'feature': feag, 'importance': feature_importance_values})
feature_importances.to_csv(os.getcwd() + '/output/featureimp_GBM_500.txt', sep='\t')

#
#%%
# Corr matrix of All gbm cell lines
gbmexp_reindex = testy2.reset_index().iloc[:,1::]
gbm_syk = gbmexp_reindex.loc[:,['ENSG00000085563','ENSG00000182162','ENSG00000143119','ENSG00000102879',
                                'ENSG00000183010','ENSG00000126353','ENSG00000117091','ENSG00000075884',
                                'ENSG00000156738']]
plt.figure(figsize=(24,20))
sns.heatmap(
    gbm_syk,
    cmap='YlGnBu',
    # annot=True,
    linewidths=2
)
plt.show()


gbm_sykall = testy2.loc[:,['ENSG00000085563','ENSG00000182162','ENSG00000143119','ENSG00000102879',
                                'ENSG00000183010','ENSG00000126353','ENSG00000117091','ENSG00000075884',
                                'ENSG00000156738']]

corr_exp_gbm = gbmexp_reindex.T.corr(method = 'spearman')


plt.figure(figsize=(24,20))
sns.clustermap(
    corr_exp_gbm,
    cmap='YlGnBu',
    # annot=True,
    linewidths=2
)
plt.show()
#%%
X = train
y = train_labels
X = X.values
y = y.values

# #############################################################################
# Classification and ROC analysis

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
classifier = random_forest_gbm

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic CV')
plt.legend(loc="lower right")
plt.savefig('Per.png')
plt.show()
