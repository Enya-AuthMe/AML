import authmecv as acv
import pandas as pd
from tqdm import tqdm
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


def find_month(date):
    month = [0, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 365]
    mon = bisect.bisect(month, date) - 1
    return mon


def past(x):
    y = len(x[x.act_date<=x.date])
    return y


def scaler(df):
    scaled_sr = (df-df.min()) / (df.max() - df.min())
    scaled_df = scaled_sr.to_frame()
    return scaled_df


# Load into DataFrame
df_list = acv.PowerDict({k: pd.read_csv(str(DATA_ROOT / v))
                        for k, v in DATA.items()})

main_df = pd.concat([df_list.train_date, df_list.test_date], ignore_index=True)
month = main_df.date.copy()
for i in range(len(month)):
    month[i] = find_month(month[i])
month_df = month.to_frame(name='month')
main_df = pd.concat([main_df, month_df], axis=1)
main_df = pd.merge(main_df, df_list.custinfo, on='alert_key')

# rename to act_date
ccba = df_list.ccba.rename(columns={'byymm': 'act_date'})
cdtx = df_list.cdtx.rename(columns={'date': 'act_date'})
dp = df_list.dp.rename(columns={'tx_date': 'act_date'})
remit = df_list.remit.rename(columns={'trans_date': 'act_date'})

# dummy encoding
risk_dum_df = pd.get_dummies(main_df.risk_rank, prefix='risk_rank')
occ_dum_df = pd.get_dummies(main_df.occupation_code, prefix='occ_code', dummy_na=True)
age_dum_df = pd.get_dummies(main_df.AGE, prefix='age')
main_df_tmp = main_df.drop(columns=['risk_rank', 'occupation_code', 'AGE'])

# minmax standardize
main_df_tmp.total_asset = scaler(main_df_tmp.total_asset)

# num of tranactions
ccba_df = pd.merge(main_df_tmp, ccba, on='cust_id', how='left')
cdtx_df = pd.merge(main_df_tmp, cdtx, on='cust_id', how='left')
dp_df = pd.merge(main_df_tmp, dp, on='cust_id', how='left')
remit_df = pd.merge(main_df_tmp, remit, on='cust_id', how='left')
ccba_df_grp = ccba_df.groupby(['alert_key']).apply(past).reset_index(name='ccba_num')
cdtx_df_grp = cdtx_df.groupby(['alert_key']).apply(past).reset_index(name='cdtx_num')
dp_df_grp = dp_df.groupby(['alert_key']).apply(past).reset_index(name='dp_num')
remit_df_grp = remit_df.groupby(['alert_key']).apply(past).reset_index(name='remit_num')
ccba_df_grp.ccba_num = scaler(ccba_df_grp.ccba_num)
cdtx_df_grp.cdtx_num = scaler(cdtx_df_grp.cdtx_num)
dp_df_grp.dp_num = scaler(dp_df_grp.dp_num)
remit_df_grp.remit_num = scaler(remit_df_grp.remit_num)


# merge
main_df_mrg = pd.merge(main_df_tmp, ccba_df_grp, on='alert_key')
main_df_mrg = pd.merge(main_df_mrg, cdtx_df_grp, on='alert_key')
main_df_mrg = pd.merge(main_df_mrg, dp_df_grp, on='alert_key')
main_df_mrg = pd.merge(main_df_mrg, remit_df_grp, on='alert_key')

# concat
main_df_con = pd.concat([main_df_mrg, risk_dum_df, occ_dum_df, age_dum_df], axis=1)

# split trainset testset
train_df = pd.merge(main_df_con, df_list.y_path, on='alert_key')
test_df = pd.merge(main_df_con, df_list.test_date, on=['alert_key', 'date'])

# drop useless columns
train_df_fin = train_df.drop(columns=['date', 'month', 'cust_id'])
test_df_fin = test_df.drop(columns=['date', 'month', 'cust_id'])
breakpoint()

# train_df_fin.to_csv('./訓練資料集_first/ex1/trainset/trainset_ex1_minmax_2.csv', index=False)
# test_df_fin.to_csv('./訓練資料集_first/ex1/testset/testset_ex1_minmax_2.csv', index=False)