import copy
import glob
import os

import numpy as np
import pandas as pd
import xgboost as xgb


def prepare_data(base_dir, pred_dir, out_dir):
    file_list = []
    pattn = os.path.join(base_dir, '*.csv')
    for f in glob.glob(pattn):
        if f.endswith('_val.csv'):
            f_name = os.path.basename(f)
            f_name_pred = f_name.replace('_val', '')
            # f_val = os.path.join(base_dir, f_name_val)
            f_pred = os.path.join(pred_dir, f_name_pred)
            f_out = os.path.join(out_dir, f_name_pred)
            file_list.append([f, f_pred, f_out])
    return file_list

def add_time(file_list, day_interval=96):
    for f in file_list:
        print(f'loading {f[0]}')
        val_df = pd.read_csv(f[0])
        pred_df = pd.read_csv(f[1])
        val_df = val_df.reset_index(drop=True)
        pred_df = pred_df.reset_index(drop=True)
        pred_df = pred_df.iloc[:, 1:]
        print(pred_df.shape)

        # print(val_df.shape)
        # print(pred_df.shape)
        val_data = []
        for d in range(day_interval):
            val_time_group = val_df.iloc[d::day_interval, :1]
            limit = val_time_group.shape[0] - 3
            val_data.append(val_time_group.iloc[:limit, :])

        val_data = pd.concat(val_data, axis=0)

        val_data = val_data.values
        pred_data = pred_df.values

        # print(type(val_data))
        # print(pred_data[:3, :])
        val_data = np.concatenate(
            (val_data, pred_data),
            axis=1)
        # val_data = val_data[val_data[1, :].argsort()]
        val_data = pd.DataFrame(val_data)
        # print(val_data.info())
        val_data = val_data.sort_values(0, axis=0)
        # print(val_data.shape)
        # print(val_data.iloc[:5, :])
        # break
        val_data.to_csv(f[2])

def main():
    base_dir = '/mnt/sda5/dataset/GuoWangData/data_split'
    pred_dir = '/mnt/sda5/dataset/GuoWangData/add_time/results_fft_3for1/results'
    out_dir = '/mnt/sda5/dataset/GuoWangData/add_time/results_fft_3for1/val_pred'
    file_list = prepare_data(base_dir, pred_dir, out_dir)
    add_time(file_list)


if __name__ == '__main__':
    main()
