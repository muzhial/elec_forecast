import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
import xgboost as xgb


def parse_data(data_file):
    pd_data = pd.read_csv(data_file)
    print('=' * 5, 'info', '=' * 5)
    print(pd_data.info())
    print('=' * 5, 'describe', '=' * 5)
    print(pd_data.describe())

    is_null = pd.isnull(pd_data).values.any()
    print(f'exist null value: {is_null}')

    print('=' * 5, 'data', '=' * 5)
    print(pd_data.head())


if __name__ == '__main__':
    train_file = '/mnt/sda5/dataset/GuoWangData/data_split/JSFD001_train.csv'
    test_file = '/mnt/sda5/dataset/GuoWangData/data_split/JSFD001_val.csv'
    parse_data(train_file)
