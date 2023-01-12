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

# add new features
main_df_tmp = main_df.copy().assign(total_asset_avg=0, total_asset_std=0, total_asset_min=0, total_asset_max=0, 
                total_asset_diff_avg=0, total_asset_diff_std=0)
for idx in acv.Tqdm(main_df.index):
    cust_id = main_df.cust_id[idx]
    event_date = main_df.date[idx]
    cust_id_df = main_df.loc[main_df.cust_id == cust_id, :]
    cust_past_act = cust_id_df.loc[cust_id_df.date < event_date, :]
    main_df_tmp.total_asset_avg[idx] = cust_past_act.total_asset.mean()
    main_df_tmp.total_asset_std[idx] = cust_past_act.total_asset.std()
    main_df_tmp.total_asset_min[idx] = cust_past_act.total_asset.min()
    main_df_tmp.total_asset_max[idx] = cust_past_act.total_asset.max()
    cust_past_act_df_diff = cust_past_act.total_asset.diff()
    main_df_tmp.total_asset_diff_avg[idx] = cust_past_act_df_diff.mean()
    main_df_tmp.total_asset_diff_std[idx] = cust_past_act_df_diff.std()

# alternate nan with 0
main_df_tmp.loc[main_df_tmp.total_asset_avg.isna(), 'total_asset_avg'] = 0
main_df_tmp.loc[main_df_tmp.total_asset_std.isna(), 'total_asset_std'] = 0
main_df_tmp.loc[main_df_tmp.total_asset_min.isna(), 'total_asset_min'] = 0
main_df_tmp.loc[main_df_tmp.total_asset_max.isna(), 'total_asset_max'] = 0
main_df_tmp.loc[main_df_tmp.total_asset_diff_avg.isna(), 'total_asset_diff_avg'] = 0
main_df_tmp.loc[main_df_tmp.total_asset_diff_std.isna(), 'total_asset_diff_std'] = 0

# one-hot encoding
risk_dum_df = pd.get_dummies(main_df_tmp.risk_rank, prefix='risk_rank')
occ_dum_df = pd.get_dummies(main_df_tmp.occupation_code, prefix='occ_code', dummy_na=True)
age_dum_df = pd.get_dummies(main_df_tmp.AGE, prefix='age')

# normalize
asset_norm_df = scaler(main_df_tmp.total_asset)
asset_norm_avg_df = scaler(main_df_tmp.total_asset_avg)
asset_norm_std_df = scaler(main_df_tmp.total_asset_std)
asset_norm_min_df = scaler(main_df_tmp.total_asset_min)
asset_norm_max_df = scaler(main_df_tmp.total_asset_max)
asset_norm_diff_avg_df = scaler(main_df_tmp.total_asset_diff_avg)
asset_norm_diff_std_df = scaler(main_df_tmp.total_asset_diff_std)

# output
# encoded_train
encoded_df_train_tmp = pd.concat([df_list.train_date.alert_key, risk_dum_df, occ_dum_df, age_dum_df, 
                    asset_norm_df, asset_norm_avg_df, asset_norm_min_df, asset_norm_max_df,
                    asset_norm_diff_avg_df, asset_norm_diff_std_df], axis=1)
encoded_df_train = pd.merge(df_list.y_path, encoded_df_train_tmp, on='alert_key', how='left')
# encoded_test
encoded_df_test_tmp = pd.concat([df_list.test_date.alert_key, risk_dum_df, occ_dum_df, age_dum_df, 
                    asset_norm_df, asset_norm_avg_df, asset_norm_min_df, asset_norm_max_df, 
                    asset_norm_diff_avg_df, asset_norm_diff_std_df], axis=1)
encoded_df_test = pd.merge(df_list.test_date, encoded_df_test_tmp, on='alert_key', how='left')
encoded_df_test = encoded_df_test.drop(columns=['date'])

breakpoint()

# encoded_df_train.to_csv('/Users/yuj/repo/SAR/訓練資料集_first/ex2/trainset/train_ex2_custinfo.csv', index=False)
# encoded_df_test.to_csv('/Users/yuj/repo/SAR/訓練資料集_first/ex2/testset/test_ex2_custinfo.csv', index=False)
