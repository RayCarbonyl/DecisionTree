from itertools import tee
import numpy as np
def train_dev_spiltting(whole_size,ratio = 0.7,shuffle = True):
    #return the index of train set and dev set
    #ratio = train set size / whole dataset size
    whole_idx = np.arange(whole_size)     
    if(shuffle==True):
        np.random.shuffle(whole_index)
    train_size = int(whole_size*ratio)
    train_idx = whole_idx[:train_size]
    dev_idx = whole_idx[train_size:]
    #print(train_idx,dev_idx)
    return train_idx,dev_idx

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)