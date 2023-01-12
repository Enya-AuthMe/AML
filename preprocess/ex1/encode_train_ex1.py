import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as sk_prepro

ex1 = './Class_Dataset/trainset/class_train_ex1.csv'
df = pd.read_csv(ex1)

# Control section
standard_way = 'min-max' # 'z-score' of 'min-max'


def dummy_encoding(tag, df):
    dictn = {}
    df_dum = pd.get_dummies(df[tag])
    lenth = len(df[tag].value_counts())
    for i in range(lenth):
        dictn[i] = tag + '_' + str(i)
    df_dum = df_dum.rename(columns=dictn)
    df = pd.concat([df, df_dum], axis=1)
    return df


def class_anly(tag):
    dictn = {tag: [], 'Proportion': []}
    for i in range(len(df[tag].value_counts())):
        denominator = df[tag].value_counts()[i]
        dft = df[df[tag] == i]
        numerator = len(dft[dft['sar'] == 1])
        pro = numerator / denominator
        dictn[tag].append(i)
        dictn['Proportion'].append(pro)
        dfn = pd.DataFrame(dictn)
    if standard_way == 'z-score':
        dfn['Proportion'] = (dfn['Proportion']-dfn['Proportion'].mean()) / dfn['Proportion'].std() # z-score
    else:
        dfn['Proportion'] = (dfn['Proportion']-dfn['Proportion'].min())/(dfn['Proportion'].max()-dfn['Proportion'].min()) # min-max
    return dfn


def target_encoding(table):
    col_0 = table.columns[0]
    col_1 = table.columns[1]
    # apply Proportion as score to all class
    for i in range(len(table[col_0])):
        df.loc[df[col_0] == i, col_0] = table.loc[i, col_1]


def delete_empty(main_df):
    df_null = main_df.loc[main_df['total_asset'] == 0]
    df_null = df_null.loc[df_null['ccba_num'] == 0]
    df_null = df_null.loc[df_null['cdtx_num'] == 0]
    df_null = df_null.loc[df_null['dp_num'] == 0]
    df_null = df_null.loc[df_null['remit_num'] == 0]
    arr_null = df_null.index.values

    for i, idx in enumerate(arr_null):
        main_df.drop(idx, axis=0, inplace=True)
    main_df.reset_index(drop=True, inplace=True)
    return 


def get_daily_trans_num(df_col_name):
    if standard_way == 'z-score':
        df_col_name = sk_prepro.StandardScaler().fit_transform(df_col_name.values.reshape(-1, 1))
    else:
        df_col_name = sk_prepro.MinMaxScaler().fit_transform(df_col_name.values.reshape(-1, 1))
    return df_col_name


# Delete all null rows (included 'total_asset' and transaction info)
delete_empty(df)

# 'risk rank' one hot encode
df = dummy_encoding(tag='risk_rank', df=df)

# 'AGE'
age_class = class_anly(tag='AGE')
target_encoding(table=age_class)

# 'occupation_code'
df.loc[df['occupation_code'].isnull(), 'occupation_code'] = 21.0
occ_class = class_anly(tag='occupation_code')
target_encoding(occ_class)

# 'total_asset'
asset = np.array(df['total_asset'])
asset[asset == 0] = 1
asset = np.log10(asset)
qct = pd.qcut(x=asset, q=7, labels=[0, 1, 2, 3, 4, 5, 6], duplicates='drop')
df_asset = pd.DataFrame(qct, columns=['total_asset_class'])
df = pd.concat([df, df_asset], axis=1)

dft = df.loc[df['sar'] == 1, 'total_asset_class']
asset_1 = np.array(dft)

df = dummy_encoding(tag='total_asset_class', df=df)

# 'ccba_num'
df = dummy_encoding(tag='ccba_num', df=df)

# 'cdtx_num'
df['cdtx_num'] = get_daily_trans_num(df['cdtx_num'])

# 'dp_num'
df['dp_num'] = get_daily_trans_num(df['dp_num'])

# 'remit_num'
df['remit_num'] = get_daily_trans_num(df['remit_num'])

# drop useless columns
df.drop(['alert_key', 'cust_id', 'risk_rank',
        'total_asset', 'date', 'total_asset_class', 'ccba_num'], axis=1, inplace=True)

breakpoint()
