#%%
# Generating the allinone table
# Flat the random forest classifiers
# Yue Zhang <yue.zhang@lih.lu>
# Oct 2018

import numpy as np
import pandas as pd
import time


def timing(func):
    def wrap(*args, **kw):

        print('<function name: %s>' % func.__name__)
        time1 = time.time()
        ret = func(*args, **kw)
        time2 = time.time()
        print('[time elapsed: %d s]' % (time2-time1))
        return ret
    return wrap

@timing
def flatforest(rf, testdf):
    """
    Extract the forest information, O(n) ~ number of trees
    :param rf:
    :param testdf:
    :return:
    """
    try:
        tree_infotable = pd.DataFrame()
        # raw_hits = pd.DataFrame()
        # predictlist_for_all = pd.DataFrame()

        for t in range(rf.n_estimators):
            # Generate the info table for trees

            # Preparation

            # Node index # Count from leftleft first /list
            nodeIndex = list(range(0, rf.estimators_[t].tree_.node_count, 1))
            # Node index forest level
            nodeInForest = list(map(lambda x: x + rf.decision_path(testdf)[1].item(t), nodeIndex))
            # lc # left children of each node, by index ^ /ndarray 1D
            lc = rf.estimators_[t].tree_.children_left
            # rc # right children of each node, by index ^ /ndarray 1D
            rc = rf.estimators_[t].tree_.children_right
            # Proportion of sample in each nodes  /ndarray +2D add later
            # TODO if the pv info is needed for the weighted GS score, re-calculate. No need to add it into the table.
            pv = rf.estimators_[t].tree_.value
            # Feature index, by index /1d array
            featureIndex = rf.estimators_[t].tree_.feature
            # Feature threshold, <= %d %threshold
            featureThreshold = rf.estimators_[t].tree_.threshold
            # Gini impurity of the node, by index
            gini = rf.estimators_[t].tree_.impurity
            # Tree index
            treeIndex = t+1
            testlist = pd.DataFrame(
                {'node_index': nodeIndex,
                 'left_c': lc,
                 'right_c': rc,
                 'feature_index': featureIndex,
                 'feature_threshold': featureThreshold,
                 'gini': gini,
                 'tree_index': treeIndex,
                 'nodeInForest': nodeInForest
                 })
# Add some salt here
            # Calculation of the default gini gain
            gslist = list()
            nodetype = list()
            for ii in range(rf.estimators_[t].tree_.node_count):
                if testlist.loc[:, 'feature_index'][ii] == -2:
                    gslist.append(-1)
                    nodetype.append("leaf_node")
                    continue  # Next if node is leaf

                ri = testlist.loc[:, 'right_c'][ii]  # right child index of node i
                li = testlist.loc[:, 'left_c'][ii]  # left child index of node i

                gs_index = testlist.loc[:, 'gini'][ii] \
                    - np.sum(pv[li])/np.sum(pv[ii])*testlist.loc[:, 'gini'][li] \
                    - np.sum(pv[ri])/np.sum(pv[ii])*testlist.loc[:, 'gini'][ri]

                gslist.append(gs_index)
                nodetype.append("decision_node")

            testlist['GS'] = pd.Series(gslist).values
            testlist['node_type'] = pd.Series(nodetype).values

            tree_infotable = pd.concat([tree_infotable, testlist])
        print("Forest %s flatted, matrix generate with %d rows and %d columns" % (rf, tree_infotable.shape[0],
                                                                                  tree_infotable.shape[1]))
        return tree_infotable
    except TypeError as argument:
        print("Process disrupted, non-valid input type ", argument)

@timing
def extarget(rf, testdf, flattedf):
    try:
        raw_hits = pd.DataFrame()
        predictlist_for_all = pd.DataFrame()
        for s_index in range(rf.decision_path(testdf)[0].indptr.shape[0] - 1):  # Loop on samples for prediction
            sample_ceiling = rf.decision_path(testdf)[0].indptr[s_index + 1]  # The ceiling hit index of the current s
            sample_floor = rf.decision_path(testdf)[0].indptr[s_index]
            hitall = pd.DataFrame()
            predictlist = list()   # Store the predictions among the forest for a certain sample
            treelist = list()
            samplelist = s_index
            for ttt in range(rf.n_estimators):
                pred_s_t = rf.estimators_[ttt].predict(testdf)[s_index]
                predictlist.append(pred_s_t)
                treelist.append(ttt)
            predictlist_for_sample = pd.DataFrame(
                {'prediction': predictlist,
                 'tree index': treelist,
                 'sample': samplelist
                 })
            predictlist_for_sample['matching'] = np.where(predictlist_for_sample['prediction'] ==
                                                          rf.predict(testdf)[predictlist_for_sample['sample']],
                                                          'match', 'not_matching')
            predictlist_for_all = pd.concat([predictlist_for_all, predictlist_for_sample])

            for hit_index in range(sample_floor, sample_ceiling):  # Loop through the hits of the current sample
                hit = flattedf.loc[flattedf['nodeInForest'] == rf.decision_path(testdf)[0].indices[hit_index],
                            ['feature_index', 'GS', 'tree_index','feature_threshold']]
                hit['sample_index'] = pd.Series(s_index).values
                hitall = pd.concat([hitall, hit])
            raw_hits = pd.concat([raw_hits, hitall])

        df = list()
        df.extend((flattedf, raw_hits, predictlist_for_all))
        print("All node used for predicting samples extracted")
        return df
    except TypeError as argument:
        print("Process disrupted, non-valid input type ", argument)
