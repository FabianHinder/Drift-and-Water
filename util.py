import numpy as np
import pandas as pd

import os
import glob
import time


####################################
base_path = "./pressures"
####################################



hour_len = int(60/15)
day_len = 24*hour_len
week_len = 7*day_len
month_len = 4*week_len
sensor_cols = ["n54","n516","n769","n722","n506","n679","n342","n415","n726","n188","n740","n549","n495","n636","n229",
               "n163","n410","n519","n752","n296","n458","n429","n288","n644","n469","n105","n114","n613","n332"]

def read_file(file):
    path = f"{base_path}/{file}/pressure.pkl"
    X = pd.read_pickle(path)[sensor_cols].values
    n = X.shape[0]
    n = n-n%week_len
    return X[:n].astype(float)

pipes = list(map(lambda x:x[len(base_path):].split("/")[2], glob.glob(base_path+"/*/pressure.pkl")))
pipes.sort()
#pipes = pipes[1:]

baseline_X = read_file("Baseline")
year_length = baseline_X.shape[0]

def time_it(f,x):
    t0 = time.time()
    y = f(x)
    t1 = time.time()
    return float("%.8f"%(t1-t0)), float("%.2f"%y)
