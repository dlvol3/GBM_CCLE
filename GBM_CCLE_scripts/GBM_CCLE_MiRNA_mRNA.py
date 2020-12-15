# Using CCLE gene expression data, microRNA data -> TMZ resistance in GBM cell lines
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


#%%

if platform.system() == 'Windows':
    # Windows PC at home
    # gdscic = pd.read_csv('P:/VM/Drug/data/output/GDSCIC50.csv')
    ccleic = pd.read_csv('C:/Users/Yue/PycharmProjects/data/CCLEIC50.csv')
    cclemiRNA = pd.read_csv('C:/Users/Yue/PycharmProjects/data/GBM/CCLE/CCLE_miRNA_20181103.csv')  # Original miRNA data
    ccle_cellline = pd.read_table('C:/Users/Yue/PycharmProjects/data/GBM/CCLE/Cell_lines_annotations_20181226.txt')
# transpose the table-> ready for joining with the GDSC

miRNA = cclemiRNA.T
miRNAnoH = miRNA.iloc[2:,]
miRNAnoH.columns = miRNA.iloc[1,]
miRNAnoH['CCLE_ID'] = miRNAnoH.index
miRNAnoH = miRNAnoH.reset_index(drop = True)
miRNA_SampleName = pd.merge(miRNAnoH, ccle_cellline.iloc[:,[0,2]],how = 'left', left_on='CCLE_ID', right_on='CCLE_ID')
miRNA_S = miRNA_SampleName.drop(["CCLE_ID"],axis = 1)  ## Ready for joining with TMZ_GDSC_EXP data
miRNA_S['Name'].replace("U-251 MG","U251")


#%% Simple search function for pandas

def searchp(df, term):
    for row in range(df.shape[0]):  # df is the DataFrame
        for col in range(df.shape[1]):
            if df.iloc[row, col] == term:
                print(row, col)
                break
#%% Check if the substitution success?
searchp(miRNA_S, term = "U-251 MG")
#%% GBM cell lines in GDSC


gdsc = pd.read_csv('C:/Users/Yue/PycharmProjects/data/GBM/readyforpython_sub.csv')
gdsc = gdsc.rename(columns = {'Cancer.Type..matching.TCGA.label.':'cancertype'})
gbmindex = gdsc.loc[gdsc.cancertype == 'GBM', gdsc.columns[0:3]].index.tolist()
gbmindex
gbmname = gdsc.loc[gbmindex, 'Sample.Name']

gdsc = gdsc.iloc[0:201,]

# join GDSC exp data with miRNA

allinone = pd.merge(gdsc, miRNA_S, how = "left", left_on="Sample.Name", right_on="Name")
# Create list for subset
flist = list(range(2, (len(allinone.columns)-1), 1))
# ciLapa.insert(0, 1)

# subset two sets
tmz= allinone.iloc[:, flist]   # Almost ready, need to remove the unmatched NAs

tmz_ready = tmz.dropna(axis = 0) # 110 samples left ready

# Check GBM cell lines left

GBM_test = tmz_ready.loc[gbmindex]
