
import pandas as pd
import tensorflow as tf
import datetime
import numpy as np
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

def prep_transform(data,n_input, n_out):
    """
    Transforms data into a suppervised learning representation
    """
    x, y = [],[]
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = n_out + in_end
        if (out_end < len(data)):
            x_inp = data[in_start: in_end].reshape((n_input, 1))
            x.append(x_inp)
            y_inp = data[in_end: out_end, 0]
            y.append(y_inp)
        in_start = in_start + 1 
    return np.array(x), np.array(y)

def scale(data):
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    scaler = StandardScaler()
    scaled_price = scaler.fit_transform(data)
    return scaled_price, scaler