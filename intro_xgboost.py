import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
import xgboost as xgb


def pre_process_data(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    print('=' * 7, 'info', '=' * 7)
    print(train.info())
    print(test.info())

    train['log_loss'] = np.log(train['loss'])

    features = [x for x in train.columns
        if x not in ['id', 'loss', 'log_loss']]
    cat_features = [x for x in train.select_dtypes(include=['object'])
        if x not in ['id', 'loss', 'log_loss']]
    num_features = [x for x in train.select_dtypes(exclude=['object'])
        if x not in ['id', 'loss', 'log_loss']]

    train_x = train[features]
    train_y = train['log_loss']

    # cat feature to category type
    for c in range(len(cat_features)):
        train_x.loc[:, cat_features[c]] = train_x.loc[:, cat_features[c]].astype(
            'category').cat.codes

    print('train_x shape', train_x.shape)
    print('train_y shape', train_y.shape)
    # print(train_x.head())

    return train_x, train_y

def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y),np.exp(yhat))

def xg_model(train_x, train_y, xgb_params):
    dtrain = xgb.DMatrix(train_x, train_y)
    bst_cv1 = xgb.cv(
        xgb_params,
        dtrain,
        num_boost_round=50,
        nfold=3,
        seed=0,
        feval=xg_eval_mae,
        maximize=False,
        early_stopping_rounds=10)

    print (f'CV score: {type(bst_cv1)}', bst_cv1.iloc[-1,:]['test-mae-mean'])

    plt.figure()
    bst_cv1[['train-mae-mean', 'test-mae-mean']].plot()
    plt.savefig('result1.png')

def main():
    train_file = '/mnt/sda5/dataset/allstate-claims-severity/train.csv'
    test_file = '/mnt/sda5/dataset/allstate-claims-severity/test.csv'
    pre_process_data(train_file, test_file)

    xgb_params = {
        'seed': 0,
        'eta': 0.1,
        'colsample_bytree': 0.5,
        'silent': 1,
        'subsample': 0.5,
        'objective': 'reg:linear',
        'max_depth': 5,
        'min_child_weight': 3
    }
    train_x, train_y = pre_process_data(train_file, test_file)
    xg_model(train_x, train_y, xgb_params)


if __name__ == '__main__':
    main()
