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
    'event_table': 'ex2/event_table.csv'
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


# event_table
event_df = df_list.event_table.copy()

# month
month = df_list.ccba.byymm.copy()
for i in range(len(month)):
    month[i] = find_month(month[i])
month_df = month.to_frame(name='month')
ccba_df = pd.concat([df_list.ccba.copy(), month_df], axis=1)

# main
main_df = pd.merge(event_df, ccba_df, on=['cust_id', 'month'], how='left')
main_df = main_df.drop(columns='byymm')

# add feature -> cuse_rate
main_df_tmp = main_df.assign(cuse_rate=0)
main_df_tmp.cuse_rate = main_df_tmp.usgam / main_df_tmp.cycam

# alternate nan with 0
for col_num in range(4, len(main_df_tmp.columns)):
    col_name = main_df_tmp.columns[col_num]
    main_df_tmp.loc[main_df_tmp.iloc[:,col_num].isna(), col_name] = 0

# normalize
for col_num in range(4, len(main_df_tmp.columns)):
    col_name = main_df_tmp.columns[col_num]
    main_df_tmp.iloc[:, col_num] = scaler(main_df_tmp.iloc[:, col_num])

# output
# encoded_train
encoded_df_train = pd.merge(df_list.y_path, main_df_tmp, on='alert_key')
encoded_df_train.drop(columns=['date', 'month', 'cust_id'], inplace=True)
# encoded_test
encoded_df_test = pd.merge(df_list.test_date, main_df_tmp, on=['alert_key', 'date'])
encoded_df_test.drop(columns=['date', 'month', 'cust_id'], inplace=True)

breakpoint()
# encoded_df_train.to_csv('/Users/yuj/repo/SAR/訓練資料集_first/ex2/trainset/train_ex2_ccba.csv', index=False)
# encoded_df_test.to_csv('/Users/yuj/repo/SAR/訓練資料集_first/ex2/testset/test_ex2_ccba.csv', index=False)