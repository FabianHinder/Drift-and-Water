import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score as roc
from scipy.stats import ks_2samp
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold

from tqdm import tqdm

import random
import time
import pickle

import sys
import json
import time

from util import time_it, pipes, baseline_X, year_length, hour_len, day_len, week_len, sensor_cols, read_file

## CODE FROM https://github.com/FabianHinder/One-or-Two-Things-We-Know-about-Concept-Drift
def d3(X,clf = LogisticRegression(solver='liblinear')):
    y = np.ones(X.shape[0])
    y[:int(X.shape[0]/2)] = 0
    
    predictions = np.zeros(y.shape)
    
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = probs
    auc_score = roc(y, predictions)
    
    return 1 - auc_score

def gen_window_matrix(l1,l2, n_perm, chage=dict()):
    if (l1,l2, n_perm) not in chage.keys():
        w = np.array(l1*[1./l1]+(l2)*[-1./(l2)])
        W = np.array([w] + [np.random.permutation(w) for _ in range(n_perm)])
        chage[(l1,l2,n_perm)] = W
    return chage[(l1,l2,n_perm)]
def mmd(X, s=None, n_perm=2500):
    K = apply_kernel(X, metric="linear")
    if s is None:
        s = int(X.shape[0]/2)
    
    W = gen_window_matrix(s,K.shape[0]-s, n_perm)
    s = np.einsum('ij,ij->i', np.dot(W, K), W)
    p = (s[0] < s).sum() / n_perm
    
    return s[0], p

def ks(X, s=None):
    if s is None:
        s = int(X.shape[0]/2)
    scores = [ks_2samp(X[:,i][:s], X[:,i][s:],mode="exact")[1] for i in range(X.shape[1])]
    return min(scores), scores

def get_time_kernel(n_size, n_perm, kernel, chage=dict()):
    if (n_size,n_perm,kernel) not in chage.keys():
        T = np.linspace(-1,1,n_size).reshape(-1,1)
        H = np.eye(n_size) - (1/n_size) * np.ones((n_size, n_size))
        if kernel == "min":
            K = [H @ np.minimum(T[None,:],T[:,None])[:,:,0] @ H] 
            for _ in range(n_perm):
                T = np.random.permutation(T)
                K.append(H @ np.minimum(T[None,:],T[:,None])[:,:,0] @ H)
        else:
            K = [H @ apply_kernel(T, metric=kernel) @ H] 
            for _ in range(n_perm):
                T = np.random.permutation(T)
                K.append(H @ apply_kernel(T, metric=kernel) @ H)
        K = np.array(K)
        K = K.reshape(n_perm+1,n_size*n_size)
        chage[(n_size,n_perm,kernel)] = K
    return chage[(n_size,n_perm,kernel)]
def dawidd(X, T_kernel="rbf", n_perm=2500):
    n_size = X.shape[0]
    
    K_X = apply_kernel(X, metric="linear")
    s = get_time_kernel(n_size, n_perm, T_kernel) @ K_X.ravel()
    p = (s[0] < s).sum() / n_perm
    
    return (1/n_size)**2 * s[0], p
#####

def run_exp(pipe):
    leak_X = read_file(pipe) if pipe != "Baseline" else baseline_X
    results = []
    
    for position in ([0]+[3*hour_len, 6*hour_len, 12*hour_len, day_len, 2*day_len, 4*day_len, 6*day_len] if pipe != "Baseline" else [0]):
        for start,end in [(day_len*i,day_len*i+2*week_len) for i in range(0,365-14,3)]:
            X = np.vstack( (baseline_X[start:position+week_len+start],leak_X[position+week_len+start:end]) )
            assert X.shape[0] == 2*week_len
            X = X[::2]

            results += [{"pipe": pipe, "position": position, "detector": detector, "value": value, "comp_time": time, "start": start, "end": end} for detector,(time,value) in {
                "d3 (lin)": time_it(d3, X),
                "d3 (knn)": time_it(lambda x: d3(x, clf=KNeighborsClassifier()), X),
                "ks" : time_it(lambda x:ks(x)[0], X),
                "mmd" : time_it(lambda x:mmd(x)[1], X),
                "dawidd" :  time_it(lambda x:dawidd(x)[1], X)
            }.items()]
    return results

if __name__ == "__main__" and sys.argv[1] == "train_model":
    n = 25
    with open("pipes.json","w") as f:
        json.dump( {i: pipes[i::n] for i in range(n)} , f )
    print("Prepared for running %i instances in parallel"%n)

if __name__ == "__main__" and sys.argv[1] == "run_exp":
    pipe_set = int(sys.argv[2])

    with open("pipes.json", "r") as f:
        pipes = json.load(f)[str(pipe_set)]

    results = []
    t0 = time.time()
    for r in tqdm(map(run_exp,pipes), total=len(pipes)):
        results += r
        if time.time()-t0 > 5*60:
            pd.DataFrame(results).to_pickle(f"out__results_dd_{pipe_set}.pkl.xz")
            t0 = time.time()
    pd.DataFrame(results).to_pickle(f"out__results_dd_{pipe_set}.pkl.xz")


