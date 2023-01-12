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


def most(x):
    y = x.AGE.mode()[0]
    return y


def percentage(x):
    y = len(x[x.ML==1]) / len(x) * 100
    return y


# Load into DataFrame
df_list = acv.PowerDict({k: pd.read_csv(str(DATA_ROOT / v))
                        for k, v in DATA.items()})

y_df = pd.merge(df_list.y_path, df_list.train_date, on='alert_key')
main_df = pd.merge(y_df, df_list.custinfo, on='alert_key')
main_sar_df = main_df[main_df.sar_flag==1]
main_df_grp = main_df.groupby(['cust_id'])['AGE'].nunique().reset_index(name='types')

# changed
chg_id = main_df_grp[main_df_grp.types>1]
chg_df = pd.merge(main_df, chg_id, on='cust_id')

age_grp = main_df.groupby(['cust_id']).apply(most).reset_index(name='main_AGE')

sar_id_df = main_sar_df.cust_id.drop_duplicates().to_frame()
sar_id_df = sar_id_df.assign(ML=1)
sar_prob_df = pd.merge(age_grp, sar_id_df, on='cust_id', how='left')
sar_prob_df.loc[sar_prob_df.ML.isna()==True, 'ML'] = 0
sar_prob_df_grp = sar_prob_df.groupby(['main_AGE']).apply(percentage)
print(sar_prob_df_grp)

# plt
x_sar = np.arange(0,11) # x軸的值
y_sar = sar_prob_df_grp.tolist()  # y軸的值

x_all = np.arange(0, 11)
y_all = age_grp.main_AGE.value_counts().sort_index().values

# 這段要在 breakpoint() 中執行一次
fig = plt.figure()  # 初始化畫布
fig, ax = plt.subplots()
plt.title("Age", fontdict={'fontweight': 'bold'}) # 圖的標題
sns.set_theme()
sns.set_style('whitegrid')
ax.bar(x_all, y_all, color='#7eb54e', width=0.4)
ax.bar(x_sar+0.4, y_sar, color='orange', width=0.4)
ax.legend(loc='upper right', labels=['Proportion of each class', 'Proportion of SAR=1 in each class'])
plt.savefig('age_both.png')

breakpoint()