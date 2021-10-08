import copy
import glob
import os

import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import xgboost as xgb


def time_features(df):
    df['year'] = df['date_time'].dt.year
    df['quarter'] = df['date_time'].dt.quarter
    df['month'] = df['date_time'].dt.month
    df['dayofyear'] = df['date_time'].dt.dayofyear
    df['dayofmonth'] = df['date_time'].dt.dayofmonth
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['weekofyear'] = df['weekofyear'].dt.weekofyear
    df['hour'] = df['date_time'].dt.hour
    return df

def fft_features(data):
    features = []
    for i in range(data.shape[1]):
        fft_complex = fft(data[:, i])
        fft_mag = [np.sqrt(
            np.real(x) * np.real(x) + np.imag(x) * np.imag(x)
            ) for x in fft_complex]
        # fft_mag = np.log(fft_mag)
        features.append(fft_mag)
    return np.asarray(features).T

def series_data(data, day_interval=96):
    data_merge = []
    for d in range(day_interval):
        day_split_data = data[d::day_interval, :]
        split_data_y = pd.DataFrame(day_split_data[:, :1])
        split_data_y = split_data_y.shift(-3)
        split_data_y.dropna(inplace=True)

        split_data_x = pd.DataFrame(day_split_data[:, 1:])

        split_data_x = split_data_x.values
        split_data_y = split_data_y.values
        split_data_x_fft = fft_features(split_data_x)
        split_data_x = np.concatenate(
            (split_data_x, split_data_x_fft),
            axis=1)

        split_data_x = pd.DataFrame(split_data_x)
        split_data_x = pd.concat(
            [split_data_x, split_data_x.shift(-1), split_data_x.shift(-2)],
            axis=1)
        split_data_x.dropna(inplace=True)

        min_len = min(len(split_data_x), len(split_data_y))
        split_data_x = split_data_x.values

        split_data = np.concatenate(
            (split_data_x[:min_len, :], split_data_y[:min_len, :]),
            axis=1)

        data_merge.append(split_data)

    data_merge = np.concatenate(data_merge, axis=0)
    return data_merge

def parse_data(data_file):
    pd_data = pd.read_csv(data_file)

    pd_data['date_time'] = pd.DatetimeIndex(pd_data['date_time'])
    # print(pd_data['date_time'].dt.[:5])

    data_mat = pd_data.values
    return data_mat[:, 1:]

def xgboost_forecast(train, test):
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(train_x, train_y)

    yhat = model.predict(test_x)

    mse = mean_squared_error(test_y, yhat)
    return mse, yhat

def prepare_data(base_dir, out_dir):
    file_list = []
    pattn = os.path.join(base_dir, '*.csv')
    for f in glob.glob(pattn):
        if f.endswith('_train.csv'):
            f_name = os.path.basename(f)
            f_name_val = f_name.replace('_train', '_val')
            f_name_result = f_name.replace('_train', '')
            f_val = os.path.join(base_dir, f_name_val)
            f_result = os.path.join(out_dir, f_name_result)
            file_list.append([f, f_val, f_result])
    return file_list

def main():
    base_dir = '/mnt/sda5/dataset/GuoWangData/data_split'
    out_dir = '/mnt/sda5/dataset/GuoWangData/results'
    file_list = prepare_data(base_dir, out_dir)

    mses = []
    for file_item in file_list:
        print(f'loading {file_item[0]}')

        train_mat = parse_data(file_item[0])
        test_mat = parse_data(file_item[1])
        print('parse data', train_mat.shape, test_mat.shape)

        train_data = series_data(train_mat)
        test_data = series_data(test_mat)
        print('merge data', train_data.shape, test_data.shape)
        mse, predictions = xgboost_forecast(train_data, test_data)
        # print(f'MSE: {mse}')
        res_df = pd.DataFrame(
            np.stack((test_data[:, -1], predictions), axis=-1))
        res_df.to_csv(file_item[2])
        mses.append([
            os.path.basename(file_item[0]),
            mse])
        print(f'MSE: {mse}')

    mse_df = pd.DataFrame(mses)
    mse_df.to_csv('mse.csv')

if __name__ == '__main__':
    main()
