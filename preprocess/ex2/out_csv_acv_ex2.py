import authmecv as acv
import pandas as pd
from tqdm import tqdm
import numpy as np
DIR = acv.get_curdir(__file__)
DATA_ROOT = DIR.parent.parent / '訓練資料集_first'
DATA = {
    'ccba': 'public_train_x_ccba_full_hashed.csv',          # 信用卡
    'cdtx': 'public_train_x_cdtx0001_full_hashed.csv',      # 消費紀錄
    'custinfo': 'public_train_x_custinfo_full_hashed.csv',  #
    'dp': 'public_train_x_dp_full_hashed.csv',              # 貸款
    'remit': 'public_train_x_remit1_full_hashed.csv',       # 外匯
    'date': 'public_x_alert_date.csv',
    'train_date': 'train_x_alert_date.csv',
    'y_path': 'train_y_answer.csv'
}

# Load into DataFrame
df_list = acv.PowerDict({k: pd.read_csv(str(DATA_ROOT / v))
                        for k, v in DATA.items()})
# main
main_df = pd.merge(df_list.y_path, df_list.train_date, on='alert_key')
main_df = pd.merge(main_df, df_list.custinfo, on='alert_key')


# 表格名
act_table = df_list.remit
# 洗錢人
main_df_tmp = main_df[main_df.sar_flag == 1]
bad_ls = main_df_tmp.cust_id.unique()
bad_id_dict = {'cust_id': bad_ls}
bad_id = pd.DataFrame.from_dict(bad_id_dict, 'columns')
bad_df_tmp = pd.merge(bad_id, act_table, on='cust_id', how='inner')
bad_df_tmp = bad_df_tmp.groupby(['cust_id']).size()
bad_df = bad_df_tmp.describe()
# 正常人
cust_ids = sorted(set(df_list.custinfo.cust_id))
cust_ids_dict = {'cust_id': cust_ids}
cust_ids_df = pd.DataFrame.from_dict(cust_ids_dict, 'columns')
cust_ids_df_tmp = pd.merge(bad_id, cust_ids_df, on='cust_id', how='outer', indicator=True)
good_id = cust_ids_df_tmp.loc[cust_ids_df_tmp._merge=='right_only', 'cust_id']
good_id = good_id.reset_index(drop=True).to_frame()
good_df_tmp = pd.merge(good_id, act_table, on='cust_id', how='inner')
good_df_tmp = good_df_tmp.groupby(['cust_id']).size()
good_df = good_df_tmp.describe()

end = pd.concat([good_df, bad_df], axis=1).T
print(end)
breakpoint()

# cust_id
num_dp_w_sar = []
num_dp_wo_sar = []
SAR_all = pd.DataFrame()
cust_ids = sorted(set(df_list.custinfo.cust_id))
for cust_id in acv.Tqdm(cust_ids):
    cust_info = df_list.custinfo[df_list.custinfo.cust_id == cust_id]
    cust_ccba = df_list.ccba[df_list.ccba.cust_id == cust_id]
    cust_cdtx = df_list.cdtx[df_list.cdtx.cust_id == cust_id]
    cust_dp = df_list.dp[df_list.dp.cust_id == cust_id]
    cust_remit = df_list.remit[df_list.remit.cust_id == cust_id]  # 這人的 remit
    SAR_info = main_df[main_df.cust_id == cust_id]  # 這人的 info
    # dp
    # SAR_info_tmp = pd.merge(SAR_info, cust_dp, left_on='date', right_on='tx_date')
    # SAR_info_grp = SAR_info_tmp.groupby('alert_key')['tx_time'].count()
    # SAR_info = pd.merge(SAR_info, SAR_info_grp, on='alert_key')
    breakpoint()
    if (SAR_info.sar_flag == 1).any():
        print(SAR_info)
        SAR_all = pd.concat([SAR_all, SAR_info])

# main_df.to_csv('tmp.csv')
breakpoint()
