import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
import xgboost as xgb


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    df = pd.DataFrame(data)
    cols = []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

    for i in range(n_out):
        cols.append(df.shift(i))

    agg = pd.concat(cols, axis=1)
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

def walk_forward_validation(data, n_test):
    predictions = []
    train, test = data[:-n_test, :], data[-n_test:, :]
    history = [x for x in train]
    for i in range(len(test)):
        test_x, test_y = test[i, :-1], test[i, -1]
        yhat = xgboost_forecast(history, test_x)
        predictions.append(yhat)
        history.append(test[i])
        print(f'=> expected: {test_y:.2f}, predicted: {yhat:.2f}')

    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions

def xgboost_forecast(train, test_x):
    train = np.asarray(train)
    train_x, train_y = train[:, :-1], train[:, -1]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(train_x, train_y)
    yhat = model.predict(np.asarray([test_x]))
    return yhat[0]

def main():
    series = pd.read_csv('./daily-total-female-births.csv',
                         header=0, index_col=0)
    # print(series)
    values = series.values
    # print(values)
    data = series_to_supervised(values, n_in=6)
    # print(data[:10])
    mae, y, yhat = walk_forward_validation(data, 12)
    print(f'MAE: {mae:.3f}')

if __name__ == '__main__':
    main()
