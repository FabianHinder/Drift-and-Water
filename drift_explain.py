from util import time_it, pipes, baseline_X, year_length, month_len, day_len, sensor_cols, read_file
import os
import sys
import time
import random
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso as LassoModel
from sklearn.linear_model import LassoCV as LassoModelCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC as LinearSVMModel
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from boruta_py import BorutaPy


def Mean(X,y, **params):
    return DummyRegressor().fit(X,y), (X * y[:,None]).mean(axis=0)

def Lasso(X,y, **params):
    model = LassoModel().fit(X,y)
    return model, model.coef_

def LassoCV(X,y, **params):
    model = LassoModelCV(verbose=0, max_iter=1000).fit(X,y)
    return model, model.coef_

def LogisticRegression(X,y, **params):
    model = LogisticRegressionCV(verbose=0, penalty="l2", solver="saga", max_iter=400).fit(X,y)
    return model, model.coef_.ravel()

def LogisticRegressionSparse(X,y, **params):
    model = LogisticRegressionCV(verbose=0, penalty="l1", solver="saga", max_iter=400).fit(X,y)
    return model, model.coef_.ravel()

def LogisticRegressionEN(X,y, **params):
    model = LogisticRegressionCV(verbose=0, penalty="elasticnet", l1_ratios=list(np.linspace(0,1,20)), solver="saga", max_iter=50).fit(X,y)
    return model, model.coef_.ravel()

def LinearSVM(X,y, **params):
    model = LinearSVMModel(verbose=0, max_iter=1000).fit(X,y)
    return model, model.coef_.ravel()

def FeatureImportanceRF_wDD(X,y, **params):
    return FeatureImportanceRF(X, y, **params)
def FeatureImportanceRF(X,y, **params):
    return FeatureImportance0(X,y,  RandomForestRegressor(), **params)
def FeatureImportanceET_wDD(X,y, **params):
    return FeatureImportanceET(X, y, **params)
def FeatureImportanceET(X,y, **params):
    return FeatureImportance0(X,y,ExtraTreesRegressor(),**params)
def FeatureImportance0(X,y, model, **params):
    model = model.fit(X,y)
    return model, model.feature_importances_

def PermutationFeatureImportanceRF_wDD(X,y, **params):
    return PermutationFeatureImportanceRF(X, y, **params)
def PermutationFeatureImportanceRF(X,y, **params):
    return PermutationFeatureImportance0(X,y,RandomForestRegressor())
def PermutationFeatureImportanceET_wDD(X,y, **params):
    return PermutationFeatureImportanceET(X, y, **params)
def PermutationFeatureImportanceET(X,y, **params):
    return PermutationFeatureImportance0(X,y,ExtraTreesRegressor(), **params)
def PermutationFeatureImportance0(X,y, model, **params):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = model.fit(X_train,y_train)
    result = permutation_importance(model, X_test, y_test, n_repeats=10, n_jobs=1)
    return model, result.importances_mean

def Boruta(X,y, **params):
    model = RandomForestRegressor()
    boruta = BorutaPy(model, n_estimators='auto', early_stopping=True).fit(X,y)
    X_sel = X.copy()

    idx = np.logical_not(boruta.support_ + boruta.support_weak_)
    X_sel[:, idx] = np.random.random(size=(X.shape[0], idx.sum()))
    model = RandomForestRegressor().fit(X_sel,y)
    return model, boruta.stat

methods = {fun.__name__:fun for fun in 
    [Mean,LogisticRegression,LogisticRegressionSparse,LinearSVM,
     FeatureImportanceRF,FeatureImportanceET,PermutationFeatureImportanceRF,PermutationFeatureImportanceET,
     FeatureImportanceRF_wDD,FeatureImportanceET_wDD,PermutationFeatureImportanceRF_wDD,PermutationFeatureImportanceET_wDD]
}
def generate_setups(pipe):
    setups = []
    for start in range(0,year_length-2*month_len,4*day_len):
        for method in methods.keys():
            for leak_pos in [0.5,0.625,0.75]:
            	setups.append({"pipe":pipe,"start":start,"leak_at":start+int(leak_pos * 2*month_len),"end":start+2*month_len,"method":method})
    return setups

def create_scenario(with_leak, start,leak_at,end, drift_detection, degree=5):
    before_leak = baseline_X[start:leak_at]
    after_leak = with_leak[leak_at+1:end]

    if drift_detection:
        y = np.array(before_leak.shape[0]*[-1] + after_leak.shape[0]*[1]).astype(int)
    else:
        y = np.linspace(0,2*np.pi, before_leak.shape[0] + after_leak.shape[0] )
        y = np.array([np.sin(n*y+offset) for offset in [0,np.pi/2] for n in range(1,degree+1)]).T

    X = np.vstack( (before_leak,after_leak) )
    return X, y

def run_experiment(pipe, start,leak_at,end, method, split, **params):
    with_leak = read_file(pipe)
    
    drift_detection = not method.endswith("_wDD")  # TODO very dirty work-around FIXME
    X,y = create_scenario(with_leak, start,leak_at,end, drift_detection)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42*split)
    
    t0 = time.time()
    model,feature_scores = methods[method](X_train,y_train, **params)
    t1 = time.time()

    if len(feature_scores.shape) != 1:
        print("Deformed feature scores: '%s' %s"%(method,str(feature_scores.shape)))
        feature_scores = feature_scores.ravel()
    
    return {"feature_scores":json.dumps(dict(zip(sensor_cols,map(lambda x: "%.5f"%x, list(feature_scores))))), "model_score": model.score(X_test,y_test), "time":t1-t0  ,  
           "drift_detection": drift_detection} ## TODO here too cf l. 162

def run_exp(pipe):
     def add_split(d,split=1):
         d["split"] = split
         return d
     def merge(d1,d2):
         return dict(list(d1.items())+list(d2.items()))
     return list(map(lambda setup:merge(run_experiment(**setup),setup), map(add_split, generate_setups(pipe))))


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
            pd.DataFrame(results).to_pickle(f"out__drift_explain_{pipe_set}.pkl.xz")
            t0 = time.time()
    pd.DataFrame(results).to_pickle(f"out__drift_explain_{pipe_set}.pkl.xz")


