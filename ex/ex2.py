import bisect
import numpy as np
import pandas as pd
from tqdm import tqdm

ccba = './訓練資料集_first/public_train_x_ccba_full_hashed.csv'
cdtx = './訓練資料集_first/public_train_x_cdtx0001_full_hashed.csv'
custinfo = './訓練資料集_first/public_train_x_custinfo_full_hashed.csv'
dp = './訓練資料集_first/public_train_x_dp_full_hashed.csv'
remit = './訓練資料集_first/public_train_x_remit1_full_hashed.csv'
date = './訓練資料集_first/public_x_alert_date.csv'
train_date = './訓練資料集_first/train_x_alert_date.csv'
y_path = './訓練資料集_first/train_y_answer.csv'
train_class = './Class_Dataset/trainset/class_train_ex1.csv'

ccba = pd.read_csv(ccba)
cdtx = pd.read_csv(cdtx)
custinfo = pd.read_csv(custinfo)
dp = pd.read_csv(dp)
remit = pd.read_csv(remit)
test_date = pd.read_csv(date)
train_date = pd.read_csv(train_date)
y = pd.read_csv(y_path)
df = pd.read_csv(train_class)

def delete_empty(main_df):
    df_null = main_df.loc[main_df['total_asset'] == 0]
    df_null = df_null.loc[df_null['ccba_num'] == 0]
    df_null = df_null.loc[df_null['cdtx_num'] == 0]
    df_null = df_null.loc[df_null['dp_num'] == 0]
    df_null = df_null.loc[df_null['remit_num'] == 0]
    arr_null = df_null.index.values

    for i, idx in enumerate(arr_null):
        main_df.drop(idx, axis=0, inplace=True)
    main_df.reset_index(drop=True, inplace=True)

def get_month(date):
    month = [30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 365]
    mon = bisect.bisect(month, date)
    return mon



breakpoint()