# PageRank for scoring ranking of the nodes
# undirected Network to directed by NetworkX .todirect
# Weight of the edges: EI calculated based on Gini gain

#%%

import networkx as nx


#%%
# Get the Garph from the previous twonet()

tbreast
tbreastsub = tbreast.replace(index, feag)

Gbreast = nx.from_pandas_edgelist(tbreast, "feature_i", "feature_j", "EI")

nx.is_directed(DG)

DG = Gbreast.to_directed()

rktest = nx.pagerank(DG, weight='EI')

rkdata = pd.Series(rktest, name='position')
rkdata.index.name = 'PR'
rkdata
rkrank = sorted(rktest, key= rktest.get, reverse=True)

rkrank = [featurelist[i] for i in rkrank]
with open('your_file.txt', 'w') as f:
    for item in rkrank:
        f.write("%s\n" % item)



tlung
tau565 = tau565.replace(index, featurelist)

Gbla = nx.from_pandas_edgelist(tau565, "feature_i", "feature_j", "EI")

nx.is_directed(DG)

DG = Gbla.to_directed()

rktest = nx.pagerank(DG, weight='EI')

rkdata = pd.Series(rktest, name='position')
rkdata.index.name = 'PR'
rkdata
rkrank = sorted(rktest, key= rktest.get, reverse=True)

#rkrank = [featurelist[i] for i in rkrank]
with open('your_file_565.txt', 'w') as f:
    for item in rkrank:
        f.write("%s\n" % item)


# tlung

tlung1 = tlung.replace(index, featurelist)

Glung = nx.from_pandas_edgelist(tlung1, "feature_i", "feature_j", "EI")

DG = Glung.to_directed()

rktest = nx.pagerank(DG, weight='EI')

rkdata = pd.Series(rktest, name='position')
rkdata.index.name = 'PR'
rkdata
rkrank = sorted(rktest, key= rktest.get, reverse=True)

#rkrank = [featurelist[i] for i in rkrank]
with open('your_file_lung.txt', 'w') as f:
    for item in rkrank:
        f.write("%s\n" % item)

#
# if rf.type == "classification":
#     embed a regression function in the following

# tS3

ts3 = s3.replace(index, featurelist)

Gs3 = nx.from_pandas_edgelist(ts3, "feature_i", "feature_j", "EI")

DG = Gs3.to_directed()

rktest = nx.pagerank(DG, weight='EI')

rkdata = pd.Series(rktest, name='position')
rkdata.index.name = 'PR'
rkdata
rkrank = sorted(rktest, key= rktest.get, reverse=True)

#rkrank = [featurelist[i] for i in rkrank]
with open('your_file_s3pd.txt', 'w') as f:
    for item in rkrank:
        f.write("%s\n" % item)


# tH322

tH322 = H322.replace(index, featurelist)

GH322 = nx.from_pandas_edgelist(tH322, "feature_i", "feature_j", "EI")

DG = GH322.to_directed()

rktest = nx.pagerank(DG, weight='EI')

rkdata = pd.Series(rktest, name='position')
rkdata.index.name = 'PR'
rkdata
rkrank = sorted(rktest, key= rktest.get, reverse=True)

#rkrank = [featurelist[i] for i in rkrank]
with open('your_file_H322pd.txt', 'w') as f:
    for item in rkrank:
        f.write("%s\n" % item)


#%%

page rank them all!
