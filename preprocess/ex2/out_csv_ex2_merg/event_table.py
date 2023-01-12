import authmecv as acv
import pandas as pd
from tqdm import tqdm
import numpy as np
import bisect

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
    'y_path': 'train_y_answer.csv'
}

# Load into DataFrame
df_list = acv.PowerDict({k: pd.read_csv(str(DATA_ROOT / v))
                        for k, v in DATA.items()})


def find_month(date):
    month = [0, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 365]
    mon = bisect.bisect(month, date) - 1
    return mon


def scaler(df):
    scaled_df = (df-df.min()) / (df.max() - df.min())
    return scaled_df


# concat train & test
main_df = pd.concat([df_list.train_date, df_list.test_date], ignore_index=True)
month = main_df.date.copy()
for i in range(len(month)):
    month[i] = find_month(month[i])
month_df = month.to_frame(name='month')
main_df = pd.concat([main_df, month_df], axis=1)
main_df = pd.merge(main_df, df_list.custinfo, on='alert_key')
main_df = main_df.drop(columns=['risk_rank', 'occupation_code', 'total_asset', 'AGE'])
breakpoint()
# main_df.to_csv('/Users/yuj/repo/SAR/訓練資料集_first/allset/event_table.csv', index=False)