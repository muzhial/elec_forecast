import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
import xgboost as xgb


def series_data(data, day_interval=96):
    day_num = len(data) // day_interval
    data_merge = []
    for d in range(day_interval):
        day_split_data = data[d::day_interval, :]
        split_data_y = pd.DataFrame(day_split_data[:, :1])
        split_data_y = split_data_y.shift(-2)
        split_data_y.dropna(inplace=True)
        # split_data_y = pd.concat(
        #     [split_data_y, split_data_y.shift(-2)], axis=1)
        # print(split_data_y)

        split_data_x = pd.DataFrame(day_split_data[:, 1:])
        split_data_x = pd.concat(
            [split_data_x, split_data_x.shift(-1)], axis=1)
        split_data_x.dropna(inplace=True)

        min_len = min(len(split_data_x), len(split_data_y))

        split_data_x = split_data_x.values
        split_data_y = split_data_y.values
        split_data = np.concatenate(
            (split_data_x[:min_len, :], split_data_y[:min_len, :]),
            axis=1)

        data_merge.append(split_data)

    data_merge = np.concatenate(data_merge, axis=0)
    return data_merge

def parse_data(data_file):
    pd_data = pd.read_csv(data_file)
    # print('=' * 5, 'info', '=' * 5)
    # print(pd_data.info())
    # print('=' * 5, 'describe', '=' * 5)
    # print(pd_data.describe())

    # is_null = pd.isnull(pd_data).values.any()
    # print(f'exist null value: {is_null}')

    # print('=' * 5, 'data', '=' * 5)
    # print(pd_data.head())

    pd_data['date_time'] = pd.DatetimeIndex(pd_data['date_time'])
    # print(pd_data['date_time'].dt.[:5])

    data_mat = pd_data.values
    return data_mat[:, 1:]

def xgboost_forecast(train, test_x):
    train_x, train_y = train[:, :-1], train[:, -1]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(train_x, train_y)
    # yhat = model.predict(np.asarray([test_x]))
    # return yhat[0]

def main():
    train_file = '/mnt/sda5/dataset/GuoWangData/data_split/JSFD001_train.csv'
    test_file = '/mnt/sda5/dataset/GuoWangData/data_split/JSFD001_val.csv'
    train_mat = parse_data(train_file)
    train_data = series_data(train_mat)
    # print(train_data.shape)
    xgboost_forecast(train_data, None)

if __name__ == '__main__':
    main()
