# for contest testset
import pandas as pd
import numpy as np
import bisect
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


ex1 = './Class_Dataset/testset/class_test_ex1.csv'
class_train_path = './Class_Dataset/trainset/class_train_ex1.csv'
df = pd.read_csv(ex1)
df_train = pd.read_csv(class_train_path)


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
    for i in range(len(df_train[tag].value_counts())):
        denominator = df_train[tag].value_counts()[i]
        dft = df_train[df_train[tag] == i]
        numerator = len(dft[dft['sar'] == 1])
        pro = numerator / denominator
        dictn[tag].append(i)
        dictn['Proportion'].append(pro)
        dfn = pd.DataFrame(dictn)
        dfn['Proportion'] = (
            dfn['Proportion']-dfn['Proportion'].mean()) / dfn['Proportion'].std()
    return dfn


def frq_encoding(table):
    col_0 = table.columns[0]
    col_1 = table.columns[1]
    # apply Proportion as score to all class
    for i in range(len(table[col_0])):
        df.loc[df[col_0] == i, col_0] = table.loc[i, col_1]


def log10(dataframe):
    asset = np.array(dataframe['total_asset'])
    asset[asset == 0] = 1
    asset = np.log10(asset)
    return asset


def train_asset_class():
    asset = log10(dataframe=df_train)
    qct = pd.qcut(x=asset, q=7, duplicates='drop')
    asset_interval = []
    for i in range(len(qct.categories)-1):
        asset_interval.append(qct.categories[i].right)
    return asset_interval


# 'risk rank' one hot encode
df = dummy_encoding(tag='risk_rank', df=df)

# 'AGE'
age_class = class_anly(tag='AGE')
frq_encoding(table=age_class)

# 'occupation_code'
df.loc[df['occupation_code'].isnull(), 'occupation_code'] = 21.0
occ_class = class_anly(tag='occupation_code')
frq_encoding(occ_class)

# 'total_asset'
train_asset_interval = train_asset_class()
df['total_asset'] = log10(dataframe=df)
for i, tt_train_asset in enumerate(df['total_asset']):
    df.loc[i, 'total_asset'] = bisect.bisect(
        train_asset_interval, tt_train_asset)
df = dummy_encoding(tag='total_asset', df=df)

# 'ccba_num'
df = dummy_encoding(tag='ccba_num', df=df)

# 'cdtx_num'
df['cdtx_num'] = StandardScaler().fit_transform(
    df['cdtx_num'].values.reshape(-1, 1))

# 'dp_num'
df['dp_num'] = StandardScaler().fit_transform(
    df['dp_num'].values.reshape(-1, 1))

# 'remit_num'
df['remit_num'] = StandardScaler().fit_transform(
    df['remit_num'].values.reshape(-1, 1))

# drop useless columns
df.drop(['cust_id', 'risk_rank',
        'total_asset', 'date', 'ccba_num', 'sar'], axis=1, inplace=True)

breakpoint()
