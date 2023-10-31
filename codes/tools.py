import pandas as pd


def zscore_train(df, output_path=None):
    # 给imbalance_size加符号
    df['imbalance_size'] = df['imbalance_size'] * df['imbalance_buy_sell_flag']
    df.drop('imbalance_buy_sell_flag', axis=1, inplace=True)

    # 计算与wap的横向偏移（减1）（待商榷）
    price_columns = [each for each in df.columns.tolist() if str(each).__contains__('price')]
    for each in price_columns:
        df[each] = df[each] - 1

    # 删id列
    df.drop(['date_id', 'time_id', 'row_id', 'wap'], axis=1, inplace=True)

    # 标准化
    data_info = {}
    for each in df.columns.tolist():
        if each == 'target':
            continue
        data_info[each] = [df[each].mean(), df[each].std()]
        df[each] = (df[each] - df[each].mean()) / df[each].std()
        df[each].fillna(0, inplace=True)
    data_info = pd.DataFrame(data_info)

    if output_path:
        df.to_csv(output_path + 'train_zscore.csv', index=False)
        data_info.to_csv(output_path + 'train_info.csv', index=False)
    return df


def zscore_test(df, info_path):
    # 给imbalance_size加符号
    df['imbalance_size'] = df['imbalance_size'] * df['imbalance_buy_sell_flag']

    # 计算与wap的横向偏移（减1）（待商榷）
    price_columns = [each for each in df.columns.tolist() if str(each).__contains__('price')]
    for each in price_columns:
        df[each] = df[each] - 1

    # 标准化
    info = pd.read_csv(info_path)
    info = info.to_dict()
    output = pd.DataFrame()
    for each in info.keys():
        mean, std = info[each].values()
        output[each] = (df[each] - mean) / std
    output.fillna(0, inplace=True)
    return output


if __name__ == '__main__':
    train = pd.read_csv('../data/origin/train.csv')
    train = zscore_train(train, '../data/processed/')

    test = pd.read_csv('../data/origin/test.csv')
    test = zscore_test(test, '../data/processed/train_info.csv')
    test.to_csv('../data/processed/test_zscore.csv', index=False)
