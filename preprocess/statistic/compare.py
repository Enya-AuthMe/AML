import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

X_path = './訓練資料集_first/public_train_x_custinfo_full_hashed.csv'
Y_path = './訓練資料集_first/train_y_answer.csv'
X = pd.read_csv(X_path)
Y = pd.read_csv(Y_path)

SAR = 'sar_flag'
KEY = 'alert_key'
ID = 'cust_id'
RISK = 'risk_rank'
ASSET = 'total_asset'
AGE = 'AGE'
OCC = 'occupation_code'


# 類別分析
def class_anly(dic, buttom, tar_attr):
    d_key = list(dic)
    key_cust = d_key[0]
    key_perc = d_key[1]
    for occ_num in range(len(X[buttom].value_counts())):
        fir = 0
        for i, types in enumerate(tar_attr):
            if types == occ_num:
                fir += 1

        uno = 0
        for i, type in enumerate(X[buttom]):
            if type == occ_num:
                uno += 1

        print('---'*5)
        print(key_cust+' : ')
        print('[num of sar1 & %s %d] : %d' % (key_cust, occ_num, fir))
        print('[num of %s %d] : %d' % (key_cust, occ_num, uno))
        print('[percentage of %s %d] : %.4f' %
              (key_cust, occ_num, fir*100/uno))
        dic[key_cust].append(str(occ_num))
        dic[key_perc].append(fir*100/uno)
    df = pd.DataFrame.from_dict(dic)
    return df


# Step 1. 請選擇你要分析的欄位名稱
buttom = AGE

# 補齊 OCC 空值
X.loc[X[OCC].isnull(), OCC] = 21.0

# public 訓練集全範圍考慮
ls = []
for i, key in enumerate(Y[KEY]):
    row = X[X[KEY] == key].index.values
    for j, rows in enumerate(row):
        ls.append(X[buttom][rows])
        have_sar = ls
breakpoint()


# sar=1 的情況下...
sar_1_key = []
for i, sar in enumerate(Y[SAR]):
    if sar == 1:
        sar_1_key.append(Y[KEY][i])

ls = []
for i, key in enumerate(sar_1_key):
    row = X[X[KEY] == key].index.values
    for j, rows in enumerate(row):
        ls.append(X[buttom][rows])
        tar_attr = ls

# Step 2. 選擇要分析的屬性把 code 解開

# # 'asset'
# ttr = np.array(tar_attr)
# ttr[ttr == 0] = 1
# ttr_log = np.log10(ttr)
# df_ttr = pd.DataFrame(ttr_log)
# qct = pd.qcut(x=ttr_log, q=8)
# print(qct.value_counts())
# qct = pd.qcut(x=ttr_log, q=8, labels=[0, 1, 2, 3, 4, 5, 6, 7])
# print(qct.value_counts())
# # sns.boxplot(x=tar_attr)  # 箱形圖
# sns.histplot(tar_attr,kde=False)  # 長條圖 這兩種圖請擇一儲存
# # Saving as .png
# plt.savefig("asset.png")

# 'AGE'
dic_age = {'AGE': [], 'Percentage': []}
df_AGE = class_anly(dic_age, buttom, tar_attr)
# # sns.barplot(data=df_AGE, x="AGE", y="Percentage")
# # plt.savefig("age.png")

# # 'Occupation'
# dic_occ = {'occupation_code': [], 'Percentage': []}
# df_OCC = class_anly(dic_occ, buttom, tar_attr)

breakpoint()
