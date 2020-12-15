import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
import platform
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
def nerf(rf, testdf, local_index):
    try:
        ff = flatforest(rf, testdf)
        nt = nerftab(ff)
        # insert some notifications
        output = localnerf(nt, local_index)
        return output
    except TypeError as argument:
        print("Process disrupted, non-valid input type ", argument)


# TODO pack NERF into a package, import to use nerf()
