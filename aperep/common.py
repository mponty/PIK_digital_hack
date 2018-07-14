import pandas as pd
import numpy as np

def date2int(data):
  data = pd.to_datetime(data)
  data = pd.DatetimeIndex(data).astype(np.int64)
  return data

def unique():
    memo = []
    def helper(x):
        if x not in memo:            
            memo.append(x)
            print(x)

    return helper

unique_print = unique()
    
inputdir="input/"
