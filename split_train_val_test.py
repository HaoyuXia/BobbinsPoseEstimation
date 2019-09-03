import numpy as np
import pandas as pd
import random

def split_train_val_test(length, seed):
    train = []
    val = []
    test = []
    if (length % 10 != 0):
        print('invalid length!!!')
        return train, val, test
    if (seed != 0):
        random.seed(seed)
    for i in range(0, int(length/10)):
        val_test = random.sample(range(10*i + 1, 10*i + 10 + 1), 3)
        val.append(val_test[0])
        val.append(val_test[1])
        test.append(val_test[2])
        for j in range(10*i + 1, 10*i + 10 + 1, 1):
            if j not in val_test:    
                    train.append(j)
    
    return train, val, test

if __name__ == '__main__':
    train, val, test = split_train_val_test(100, 20)