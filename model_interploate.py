from util import time_it, pipes, baseline_X, year_length, hour_len, day_len, week_len, sensor_cols, read_file
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from joblib import dump, load

from tqdm import tqdm
import sys
import json
import time

def interpolation_data_label_split(X):
    n = X.shape[1]
    return {i: (X[:,[j for j in range(n) if j != i]], X[:,i]) for i in range(n)}

def score_model(model, data):
    data = data[np.random.choice(range(data.shape[0]), size=1000, replace=False)]
    MSE = []
    for feature,(X,y) in interpolation_data_label_split(data).items():
        MSE.append( ((model[feature].predict(X)-y)**2) )
    MSE = np.array(MSE).sum(axis=0)
    base = model["roc-base"]
    
    return np.mean(MSE), roc_auc_score( np.array(base.shape[0]*[False]+MSE.shape[0]*[True]), np.hstack((base,MSE)) )

if __name__ == "__main__" and sys.argv[1] == "train_model":
    models = [Ridge(), RandomForestRegressor(), KNeighborsRegressor(), KernelRidge(kernel="poly")]

    assert int(year_length / (2*week_len)) == year_length / (2*week_len)
    folds = list(map(lambda x: (x,[i for i in range(year_length) if i not in x]), [range(i*week_len, (i+2)*week_len) for i in range( int(year_length / week_len) -2)]))

    results = []
    trained_models = []
    for model0 in models:
        print(model0)
        for fold,(train,test) in enumerate(tqdm(folds)):
            test = np.random.choice(test, size=1000, replace=False)

            model = {"name": str(model0), "fold": fold}
            MSE = []
            for feature,(X,y) in interpolation_data_label_split(baseline_X).items():
                model[feature] = clone(model0).fit(X[train],y[train])
                MSE.append( ((model[feature].predict(X[test])-y[test])**2) )
            MSE = np.array(MSE).sum(axis=0)
            model["roc-base"] = MSE

            trained_models.append( model )
            mse_0, roc_0 = score_model(model, baseline_X)
            results.append( {"model": model["name"], "fold": model["fold"], "node": "Baseline_CV", "MSE": np.mean(MSE), "ROC": -1} )
            results.append( {"model": model["name"], "fold": model["fold"], "node": "Baseline_0",  "MSE": mse_0, "ROC": roc_0} ) 
    dump(trained_models, "models_interpolate_simple.mdl.xz")
    pd.DataFrame(results).to_pickle("out__interpolation_simple_baseline.pkl.xz")
    
    n = 25
    with open("pipes.json","w") as f:
        json.dump( {i: pipes[i::n] for i in range(n)} , f )
    print("Prepared for running %i instances in parallel"%n)

if __name__ == "__main__" and sys.argv[1] == "run_exp":
    pipe_set = int(sys.argv[2])

    with open("pipes.json", "r") as f:
        pipes = json.load(f)[str(pipe_set)]
    models = load("models_interpolate_simple.mdl.xz")

    t0 = time.time()
    results = []
    for pipe in tqdm(pipes):
        for model in models:
            mse_0, roc_0 = score_model(model, read_file(pipe))
            results.append( {"model": model["name"], "fold": model["fold"], "node": pipe, "MSE": mse_0, "ROC": roc_0} ) 
            if time.time()-t0 > 5*60:
                pd.DataFrame(results).to_pickle(f"out__interpolate_simple_{pipe_set}.pkl.xz")
                t0 = time.time()
    pd.DataFrame(results).to_pickle(f"out__interpolate_simple_{pipe_set}.pkl.xz")

