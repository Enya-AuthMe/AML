import authmecv as acv
import pandas as pd
from tqdm import tqdm
DIR = acv.get_curdir(__file__)
DATA_ROOT = DIR.parent / '訓練資料集_first'
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
# cust_id
num_dp_w_sar = []
num_dp_wo_sar = []
SAR1_all = pd.DataFrame()
SAR0_all = pd.DataFrame()
cust_ids = sorted(set(df_list.custinfo.cust_id))
for cust_id in acv.Tqdm(cust_ids):
    cust_info = df_list.custinfo[df_list.custinfo.cust_id == cust_id]
    cust_ccba = df_list.ccba[df_list.ccba.cust_id == cust_id]
    cust_cdtx = df_list.cdtx[df_list.cdtx.cust_id == cust_id]
    cust_dp = df_list.dp[df_list.dp.cust_id == cust_id]
    cust_remit = df_list.remit[df_list.remit.cust_id == cust_id]
    SAR_info = main_df[main_df.cust_id == cust_id]
    # SAR_info_tmp = pd.merge(
    #     SAR_info, cust_dp, left_on='date', right_on='tx_date')
    # SAR_info_grp = SAR_info_tmp.groupby('alert_key')['tx_time'].count()
    # SAR_info = pd.merge(SAR_info, SAR_info_grp, on='alert_key')
    
    if (SAR_info.sar_flag == 1).any():
        print(SAR_info)
        SAR_info = SAR_info.reset_index(drop=True)
        SAR1_all = pd.concat([SAR1_all, SAR_info])
    else:
        SAR_info = SAR_info.reset_index(drop=True)
        SAR0_all = pd.concat([SAR0_all, SAR_info])
        SAR0_all = pd.concat([SAR0_all, SAR_info])


breakpoint()
# main_df.to_csv('tmp.csv')
# SAR_all.to_csv('sar_tmp.csv', index=False)