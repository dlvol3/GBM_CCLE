#%%
# Generating the allinone table
# Take the output of flatforest() to generate the network ready table for the NERF progress
# Yue Zhang <yue.zhang@lih.lu>
# Oct 2018


import pandas as pd
import time


def timing(func):
    def wrap(*args, **kw):
        print('<function name: {0}>'.format(func.__name__))
        time1 = time.time()
        ret = func(*args, **kw)
        time2 = time.time()
        print('[timecosts: {0} s]'.format(time2-time1))
        return ret
    return wrap


@timing
# nerftab function for generating pairs of the 'correct' decision features
def nerftab(df_ff):
    # All possible pairs generator
    # TODO give the tree a index?
    # TODO Same to previous one, give sample a index to loop with

    try:
        list_of_single_decisions = pd.DataFrame()
        list_of_all_decision_pairs = pd.DataFrame()
        for psample in range(max(df_ff[1].loc[:, 'sample_index']) + 1):  # Loop on predict samples
            t_in_psample = df_ff[2].loc[(df_ff[2]['matching'] == 'match') & (df_ff[2]['sample'] == psample),'tree index']
            # Trees with the 'correct' prediction during the prediction process of the sample psample

            t_p_psample = df_ff[1].loc[(df_ff[1]['tree_index'].isin(t_in_psample + 1)) &
                                     (df_ff[1]['sample_index'] == psample) & (df_ff[1]['feature_index'] != -2), ]
            # t_in_sample + 1 to make sure the alignment between df_ff[] tree index
            # For each sample, get the 'correct' trees with decision paths, remove the leaf nodes

            single_decisions_sample = pd.DataFrame()
            all_decision_pairs_sample = pd.DataFrame()
            for itree in df_ff[2].loc[(df_ff[2]['matching'] == 'match') & (df_ff[2]['sample'] == psample), 'tree index']:
                nodes_cand = t_p_psample.loc[t_p_psample['tree_index'] == (itree + 1), ]  # Of each tree!
                if nodes_cand.shape[0] == 1:
                    #  If-statement for the situation where only one decision node presents
                    single_decisions_sample = pd.concat([single_decisions_sample, nodes_cand])
                    continue
                else:
                    pairsofonetree = pd.DataFrame()
                    for i in range(nodes_cand.shape[0]):
                        pairs_inside = pd.DataFrame()
                        for j in range(i + 1, nodes_cand.shape[0]):
                            if nodes_cand.iloc[i, 0] <= nodes_cand.iloc[j, 0]:
                                f_i = nodes_cand.iloc[i, 0]  # feature index of feature i, the small one
                                f_j = nodes_cand.iloc[j, 0]
                            elif nodes_cand.iloc[i, 0] > nodes_cand.iloc[j, 0]:
                                f_j = nodes_cand.iloc[i, 0]  # feature index of feature i, assign the small one to i
                                f_i = nodes_cand.iloc[j, 0]
                            gs_i = nodes_cand.iloc[i, 1]  # gs of feature i
                            gs_j = nodes_cand.iloc[j, 1]
                            ft_i = nodes_cand.iloc[i, 3]  # feature threshold of i
                            ft_j = nodes_cand.iloc[j, 3]
                            tr_index = nodes_cand.iloc[i, 2]
                            sp_index = nodes_cand.iloc[i, 4]
                            listofunit = [[f_i, f_j, gs_i, gs_j, ft_i, ft_j, tr_index, sp_index]]
                            dfunit = pd.DataFrame(listofunit,
                                                  columns=['feature_i', 'feature_j', 'GS_i', 'GS_j', 'threshold_i',
                                                           'threshold_j',
                                                           'tree_index', 'sample_index'])
                            pairs_inside = pd.concat([pairs_inside, dfunit])
                        pairsofonetree = pd.concat([pairsofonetree, pairs_inside])
                all_decision_pairs_sample = pd.concat([all_decision_pairs_sample, pairsofonetree])
                # One sample, all the trees^^^
            list_of_single_decisions = pd.concat([list_of_single_decisions, single_decisions_sample])
            list_of_all_decision_pairs = pd.concat([list_of_all_decision_pairs, all_decision_pairs_sample])
        df = list()
        df.extend((list_of_single_decisions, list_of_all_decision_pairs))
        print("NERFtab finished")
        return df
    except TypeError as argument:
        print("Process disrupted, non-valid input type ", argument)




