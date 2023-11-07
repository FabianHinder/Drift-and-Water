from util import time_it, pipes, baseline_X, year_length, hour_len, day_len, week_len, sensor_cols, read_file
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from usup_dd import ks as ks

def run_exp(pipe):
    leak_X = read_file(pipe) if pipe != "Baseline" else baseline_X
    results = []
    
    for position in [0]:
        for start,end in [(day_len*i,day_len*i+2*week_len) for i in range(0,365-14,3)]:
            X = np.vstack( (baseline_X[start:position+week_len+start],leak_X[position+week_len+start:end]) )
            assert X.shape[0] == 2*week_len
            X = X[::2]

            results += [{"pipe": pipe, "position": position, "detector": detector, "value": value, "coefs": coefs, "start": start, "end": end} for detector,(value,coefs) in {
                "ks": ks(X)
            }.items()]
    return results

if __name__ == "__main__" and sys.argv[1] == "run_exp":
    results = []
    t0 = time.time()
    for r in tqdm(map(run_exp, pipes), total=len(pipes)):
        results += r
        if time.time() - t0 > 5*60:
            pd.DataFrame(results).to_pickle("localization_ks.pkl.xz")
            t0 = time.time()
    adpd.DataFrame(results).to_pickle("localization_ks.pkl.xz")
