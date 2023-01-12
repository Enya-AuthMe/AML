import authmecv as acv
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def percentage(x):
    y = len(x[x.ML==1]) / len(x) * 100
    return y


# Load into DataFrame
df_list = acv.PowerDict({k: pd.read_csv(str(DATA_ROOT / v))
                        for k, v in DATA.items()})

y_df = pd.merge(df_list.y_path, df_list.train_date, on='alert_key')
train_df = pd.merge(y_df, df_list.custinfo, on='alert_key')
train_sar_df = train_df[train_df.sar_flag==1]
train_df_grp = train_df.groupby(['cust_id'])['risk_rank'].nunique().reset_index(name='types')

# changed
change_risk_rank_id_df = train_df_grp[train_df_grp.types>1]
change_risk_rank_df = pd.merge(train_df, change_risk_rank_id_df, on='cust_id')
sar_id_df = train_sar_df.cust_id.drop_duplicates().to_frame()
change_risk_rank_sar_df = pd.merge(change_risk_rank_df, sar_id_df, on='cust_id')
change_risk_rank_sar_id_df = change_risk_rank_sar_df.cust_id.drop_duplicates(keep='first').to_frame()

train_sar_tmp_df = pd.merge(train_sar_df, change_risk_rank_sar_id_df, on='cust_id', how='left', indicator=True)
train_sar_tmp_df = train_sar_tmp_df[train_sar_tmp_df._merge=='left_only']
fin_df = train_sar_tmp_df.drop(columns=['_merge'])

# plt 
data = fin_df.risk_rank.value_counts().values
labels = np.array(['rank_1', 'rank_3', 'rank_2'])
colors = sns.color_palette('pastel')
plt.pie(data, labels=labels, colors=colors, autopct = '%0.0f%%')
plt.title('SAR=1 (except for those who rank changed)')
plt.savefig('./sar_risk_rank.png')

# no changed
no_change_df_tmp = train_df_grp[train_df_grp.types==1]
no_change_df = pd.merge(train_df, no_change_df_tmp, on='cust_id')
no_change_df_grp = no_change_df.groupby(['cust_id'])['risk_rank'].max().to_frame()
all_df = no_change_df_grp

# plt 
data = all_df.risk_rank.value_counts().values
labels = np.array(['rank_1', 'rank_3', 'rank_2', 'rank_0'])
colors = sns.color_palette('pastel')
plt.pie(data, labels=labels, colors=colors, autopct = '%0.0f%%')
plt.title('All risk_rank distribution (except for those who rank changed)')
plt.savefig('./all_risk_rank_pie.png')

# sar prob of each rank
sar_id_df = train_sar_df.cust_id.drop_duplicates().to_frame()
sar_id_df = sar_id_df.assign(ML=1)
sar_prob_df = pd.merge(no_change_df_grp, sar_id_df, on='cust_id', how='left')
sar_prob_df.loc[sar_prob_df.ML.isna()==True, 'ML'] = 0
sar_prob_df_grp = sar_prob_df.groupby(['risk_rank']).apply(percentage)
print(sar_prob_df_grp)

breakpoint()


train_df.sort_values(['cust_id', 'date'])
train_sar_df.sort_values(['cust_id', 'date'])
breakpoint()