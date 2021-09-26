import pandas as pd


def common_method():
    df = pd.read_excel('https://www.gairuo.com/file/data/dataset/team.xlsx')

    print(df)
    print(df.head())  # head 5 default
    print(df.tail())  # tail 5 default
    print(df.sample(5))

    print('=' * 5, 'shape', '=' * 5)
    print(df.shape)

    print('=' * 5, 'info', '=' * 5)
    print(df.info())

    print('=' * 5, 'describe', '=' * 5)
    print(df.describe())

def main():
    common_method()


if __name__ == '__main__':
    main()
