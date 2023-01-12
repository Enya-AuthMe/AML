import authmecv as acv
import pandas as pd
from tqdm import tqdm
import numpy as np
import bisect

# Control section
data_mode = 'train'  # 'train' or 'test'

DIR = acv.get_curdir(__file__)
DATA_ROOT = DIR.parent.parent / '訓練資料集_first'
DATA = {
    'ccba': 'public_train_x_ccba_full_hashed.csv',          # 信用卡
    'cdtx': 'public_train_x_cdtx0001_full_hashed.csv',      # 消費紀錄
    'custinfo': 'public_train_x_custinfo_full_hashed.csv',  #
    'dp': 'public_train_x_dp_full_hashed.csv',              # 貸款
    'remit': 'public_train_x_remit1_full_hashed.csv',       # 外匯
    'test_date': 'public_x_alert_date.csv',
    'train_date': 'train_x_alert_date.csv',
    'y_path': 'train_y_answer.csv',
    'event_table': 'ex2/event_table.csv',
    'bad_id': 'bad_id.csv',
    'good_id': 'good_id.csv'
}

# Load into DataFrame
df_list = acv.PowerDict({k: pd.read_csv(str(DATA_ROOT / v))
                        for k, v in DATA.items()})


def find_month(date):
    month = [0, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 365]
    mon = bisect.bisect(month, date) - 1
    return mon


def scaler(series):
    scaled_series = (series-series.min()) / (series.max() - series.min())
    return scaled_series


def preset(x):
    alert_date = x.date.iloc[0]
    sorted_x = x.sort_values(by='date_cd').reset_index(drop=True)
    recent_date = sorted_x.date_cd.iloc[0]
    y_dict = {'alert_key': [np.nan], 'col': [np.nan]}
    y = pd.DataFrame(y_dict)
    y.alert_key = x.alert_key.iloc[0]
    history_df = x[x.date_cd<alert_date]
    return alert_date, sorted_x, recent_date, y, history_df


def most(x, type_name):
    alert_date, sorted_x, recent_date, y, history_df = preset(x)
    if (recent_date >= alert_date-30) & (alert_date-30 >0):
        today_date = sorted_x.date_cd.iloc[0]
        today_df = x[x.date_cd == today_date]
        if type_name == 'country':
            if ~today_df.country.isna().all():
                y.col = today_df.country.mode()[0]
        elif type_name == 'cur_type':
            if ~today_df.cur_type.isna().all():
                y.col = today_df.cur_type.mode()[0]
    return y


def kinds(x, type_name):
    alert_date, sorted_x, recent_date, y, history_df = preset(x)
    if (recent_date >= alert_date-30) & (alert_date-30 >0):
        today_date = sorted_x.date_cd.iloc[0]
        today_df = x[x.date_cd == today_date]
        if type_name == 'country':
            if ~today_df.country.isna().all():
                y.col = today_df.country.nunique()
        elif type_name == 'cur_type':
            if ~today_df.cur_type.isna().all():
                y.col = today_df.cur_type.nunique()
    return y


def most_history(x, type_name):
    alert_date, sorted_x, recent_date, y, history_df = preset(x)
    if type_name == 'country':
        if ~history_df.country.isna().all():
            y.col = history_df.country.mode()[0]
    elif type_name == 'cur_type':
        if ~history_df.cur_type.isna().all():
            y.col = history_df.cur_type.mode()[0]
    return y


def kinds_history(x, type_name):
    alert_date, sorted_x, recent_date, y, history_df = preset(x)
    if type_name == 'country':
        if ~history_df.country.isna().all():
            y.col = history_df.country.nunique()
    elif type_name == 'cur_type':
        if ~history_df.cur_type.isna().all():
            y.col = history_df.cur_type.nunique()
    return y


def calculate(x):
    alert_date, sorted_x, recent_date, y, history_df = preset(x)
    y = y.assign(amt_mean=np.nan, amt_min=np.nan, amt_max=np.nan)
    y = y.rename(columns={'col': 'amt_sum'})
    if (recent_date >= alert_date-30) & (alert_date-30 >0):
        today_date = sorted_x.date_cd.iloc[0]
        today_df = x[x.date_cd == today_date]
        if ~today_df.amt.isna().all():
            y.amt_sum = today_df.amt.sum()
            y.amt_mean = today_df.amt.mean()
            y.amt_min = today_df.amt.min()
            y.amt_max = today_df.amt.max()
    return y


def day_avg(x):
    y_dict = {'alert_key': [np.nan], 'date_cd': [np.nan], 'amt_day_avg': [np.nan]}
    y = pd.DataFrame(y_dict)
    y.alert_key = x.alert_key.iloc[0]
    y.date_cd = x.date_cd.iloc[0]
    if ~x.amt.isna().all():
        y.amt_day_avg = x.amt.mean()
    return y


def calculate_history(x): # return df with alert_key date_cd amt(day_avg)
    alert_date, sorted_x, recent_date, y, history_df = preset(x)
    y.drop(columns='col', inplace=True)
    if ~history_df.amt.isna().all():
        history_df_day_avg = history_df.groupby('date_cd').apply(day_avg)
        history_df_day_avg.reset_index(drop=True, inplace=True)
        y = pd.merge(y, history_df_day_avg, on='alert_key')
    else:
        y = y.assign(date_cd=np.nan, amt_day_avg=np.nan)
    return y


def today(x):
    today_date = x.date_cd.max()
    if (x.date_cd.max()>x.date-30).any() & (x.date-30>0).all():
        y = x[x.date_cd==today_date]
        return y


# event_table
event_df = df_list.event_table.copy()

# month
month = df_list.cdtx.date.copy()
for i in range(len(month)):
    month[i] = find_month(month[i])
month_df = month.to_frame(name='month')
cdtx_df_tmp = pd.concat([df_list.cdtx.copy(), month_df], axis=1)

# cdtx
cdtx_df = pd.merge(event_df, cdtx_df_tmp, on=['cust_id'], how='left')
cdtx_df.rename(columns={'date_x': 'date', 'date_y': 'date_cd',
               'month_x': 'month', 'month_y': 'month_cd'}, inplace=True)

# country
country_most_df_tmp = cdtx_df.groupby('alert_key').apply(most, 'country')
country_kinds_df_tmp = cdtx_df.groupby('alert_key').apply(kinds, 'country')
country_most_history_df_tmp = cdtx_df.groupby('alert_key').apply(most_history, 'country')
country_kinds_history_df_tmp = cdtx_df.groupby('alert_key').apply(kinds_history, 'country')

country_most_df = country_most_df_tmp.rename(columns={'col': 'country_most'}).reset_index(drop=True)
country_kinds_df = country_kinds_df_tmp.rename(columns={'col': 'country_kinds'}).reset_index(drop=True)
country_most_history_df = country_most_history_df_tmp.rename(columns={'col': 'country_most_history'}).reset_index(drop=True)
country_kinds_history_df = country_kinds_history_df_tmp.rename(columns={'col': 'country_kinds_history'}).reset_index(drop=True)

main_df = pd.merge(event_df, country_most_df, on='alert_key')
main_df = pd.merge(main_df, country_kinds_df, on='alert_key')
main_df = pd.merge(main_df, country_most_history_df, on='alert_key')
main_df = pd.merge(main_df, country_kinds_history_df, on='alert_key')

# cur
cur_type_most_tmp = cdtx_df.groupby('alert_key').apply(most, 'cur_type')
cur_type_kinds_tmp = cdtx_df.groupby('alert_key').apply(kinds, 'cur_type')
cur_type_most_history_tmp = cdtx_df.groupby('alert_key').apply(most_history, 'cur_type')
cur_type_kinds_history_tmp = cdtx_df.groupby('alert_key').apply(kinds_history, 'cur_type')

cur_type_most_df = cur_type_most_tmp.rename(columns={'col': 'cur_type_most'}).reset_index(drop=True)
cur_type_kinds_df = cur_type_kinds_tmp.rename(columns={'col': 'cur_type_kinds'}).reset_index(drop=True)
cur_type_most_history_df = cur_type_most_history_tmp.rename(columns={'col': 'cur_type_most_history'}).reset_index(drop=True)
cur_type_kinds_history_df = cur_type_kinds_history_tmp.rename(columns={'col': 'cur_type_kinds_history'}).reset_index(drop=True)

main_df = pd.merge(main_df, cur_type_most_df, on='alert_key')
main_df = pd.merge(main_df, cur_type_kinds_df, on='alert_key')
main_df = pd.merge(main_df, cur_type_most_history_df, on='alert_key')
main_df = pd.merge(main_df, cur_type_kinds_history_df, on='alert_key')

# amt
amt_cal_df = cdtx_df.groupby('alert_key').apply(calculate)
amt_cal_df.reset_index(drop=True, inplace=True)
# amt_cal_history_df = cdtx_df.groupby(['alert_key', 'date_cd']).apply(calculate_history) # 太久 等到發瘋
main_df = pd.merge(main_df, amt_cal_df, on='alert_key')

# freq
cdtx_included_df = cdtx_df.loc[cdtx_df.date_cd<=cdtx_df.date, :]  # 含當天
cdtx_history_df = cdtx_df.loc[cdtx_df.date_cd<cdtx_df.date, :]  # 不含當天
# trans_freq
trans_freq_grp = cdtx_included_df.groupby(['alert_key', 'date', 'date_cd'])['amt'].count().reset_index(name='trans_freq')
trans_freq_df = trans_freq_grp.groupby(['alert_key']).apply(today)
trans_freq_df.reset_index(drop=True, inplace=True)
trans_freq_df.drop(columns=['date', 'date_cd'], inplace=True)
main_df = pd.merge(main_df, trans_freq_df, on='alert_key', how='left')
# trans_freq_history
trans_freq_history_grp = cdtx_history_df.groupby(['alert_key', 'date_cd'])['amt'].count().reset_index(name='trans_freq_daliy') # 天筆數
trans_freq_history_df = trans_freq_history_grp.groupby(['alert_key'])['trans_freq_daliy'].sum().reset_index(name='trans_freq_history') # 過去每天筆數加總
main_df = pd.merge(main_df, trans_freq_history_df, on='alert_key', how='left')


breakpoint()
# encoded_df_train.to_csv('/Users/yuj/repo/SAR/訓練資料集_first/ex2/trainset/train_ex2_ccba.csv', index=False)
# encoded_df_test.to_csv('/Users/yuj/repo/SAR/訓練資料集_first/ex2/testset/test_ex2_ccba.csv', index=False)
