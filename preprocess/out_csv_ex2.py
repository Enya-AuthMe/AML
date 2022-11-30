import pandas as pd
import numpy as np
from tqdm import tqdm

# original csv path
ccba = './訓練資料集_first/public_train_x_ccba_full_hashed.csv'
cdtx = './訓練資料集_first/public_train_x_cdtx0001_full_hashed.csv'
custinfo = './訓練資料集_first/public_train_x_custinfo_full_hashed.csv'
dp = './訓練資料集_first/public_train_x_dp_full_hashed.csv'
remit = './訓練資料集_first/public_train_x_remit1_full_hashed.csv'
date_path = './訓練資料集_first/public_x_alert_date.csv'
train_date = './訓練資料集_first/train_x_alert_date.csv'
y_path = './訓練資料集_first/train_y_answer.csv'

# read csv
ccba = pd.read_csv(ccba)
cdtx = pd.read_csv(cdtx)
custinfo = pd.read_csv(custinfo)
dp = pd.read_csv(dp)
remit = pd.read_csv(remit)
test_date = pd.read_csv(date_path)
train_date = pd.read_csv(train_date)
y = pd.read_csv(y_path)

# Control section
data_mode = 'train'  # 'train' or 'test'
breakpoint()

# def 們


def give_event(idx, dateset):
    key = dateset['alert_key'][idx]
    date = dateset['date'][idx]
    id = custinfo[custinfo['alert_key'] == key]['cust_id'].values[0]
    sar = y['sar_flag'][idx]
    return key, id, date, sar


def find_month(date):
    mon = 0
    flag = 0
    month = [0, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 365]
    month_comp = [0, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 365]
    month_comp.append(date)
    month_comp.sort()

    for i in range(len(month)):
        if month[i] != month_comp[i]:
            mon = month[i-1]
            flag = 1
            break
    if flag == 0:
        mon = month[len(month)-1]
    return mon


def find_today(id_table, date_tag, date, month):
    flag = 0
    today = 0
    for i in range(date, month-1, -1):
        if len(id_table[id_table[date_tag] == i]) != 0:
            today = i
            flag = 0
            break
        else:
            flag = 1
    if flag == 1:
        today = date
    return today, flag


def get_thisday(id_table, date_tag, tag, day, minmax):
    if minmax == 'min':
        bool_val = True
    else:
        bool_val = False
    target = id_table.loc[id_table[date_tag] == day, tag].value_counts(
        ascending=bool_val).index.values[0]
    return target


def get_history_sth(today_obj, id_table, date_tag, day):
    last_table = id_table[id_table[date_tag] < day]


def table(key, id, event_date, sar):
    # date SAR
    the_date = pd.DataFrame({'date': [event_date]})
    the_SAR = pd.DataFrame({'sar': [sar]})

    # 1 custinfo
    table1 = custinfo.loc[custinfo['alert_key'] == key, :]

    # 2 ccba
    mon = find_month(date=event_date)
    ccba_id = ccba.loc[ccba['cust_id'] == id, :]
    table2 = ccba_id[ccba_id['byymm'] == mon]

    if len(table2['usgam']) * len(table2['cycam']) == 0:
        table2['cuse_rate'] = np.nan
    else:
        table2['cuse_rate'] = table2['usgam'].values[0] / \
            table2['cycam'].values[0]
    # table2 = table2.drop(columns=['byymm'])

    # 3 cdtx
    cdtx_id = cdtx[cdtx['cust_id'] == id]
    today, flg = find_today(cdtx_id, 'date', event_date, mon)
    attr3 = ['country', 'country_history', 'country_last_day', 'cur_type_history', 'cur_type_last_day',
             'amt_total', 'amt_avg', 'amt_max', 'amt_min', 'amt_avg_last_day', 'amt_max_last_day',
             'amt_min_last_day', 'amt_avg_last_7day', 'amt_max_last_7day', 'amt_min_last_7day',
             'amt_avg_history', 'amt_max_history', 'amt_min_history', 'amt_diff_avg_history',
             'amt_diff_max_history', 'amt_diff_min_history', 'amt_avg_last_5', 'amt_max_last_5',
             'amt_avg_last_10', 'amt_max_last_10', 'cdtx_trans_num', 'cdtx_trans_num_last_day']
    # table3 = pd.DataFrame(放row,columns=attr3)

    arr = np.zeros((len(attr3)))
    if len(cdtx_id.loc[cdtx_id['date'] == today, 'country']) != 0:
        # 'country'
        arr[0] = get_thisday(cdtx_id, 'date', 'country', today, 'min')
        # 'cur_type'
        arr[3] = get_thisday(cdtx_id, 'date', 'cur_type', today, 'min')

        cdtx_last = cdtx_id[cdtx_id['date'] < today]
        breakpoint()
        if len(cdtx_last) != 0:
            # 'country_history'
            most_country_history = cdtx_last['country'].value_counts(
            ).index.values[0]
            if arr[0] == most_country_history:
                arr[1] = 1
            else:
                arr[1] = 0
            # 'country_last_day'
            idx_sort = cdtx_last.sort_values(
                'date', ascending=False).index.values[0]
            last_date = cdtx_last.loc[idx_sort, 'date']
            most_country_last = get_thisday(
                cdtx_last, 'date', 'country', last_date, 'min')
            if arr[0] == most_country_last:
                arr[2] = 1
            else:
                arr[2] = 0
            # 'cur_type_history'
            most_cur_type_history = cdtx_last['cur_type'].value_counts(
            ).index.values[0]

            # 'cur_type_last_day'

        else:
            arr[1] = np.nan
    else:
        arr[0] = np.nan

    # table3 = cdtx_id[cdtx_id['date'] == event_date]
    # breakpoint()

    # # 4 dp
    # dp_id = dp[dp['cust_id'] == id]
    # table4 = dp_id[dp_id['tx_date'] == event_date]
    # num4 = pd.DataFrame({'dp_num': [len(table4)]})

    # # 5 remi
    # remit_id = remit[remit['cust_id'] == id]
    # table5 = remit_id[remit_id['trans_date'] == event_date]
    # num5 = pd.DataFrame({'remit_num': [len(table5)]})

    # df = table1
    # # df = df.merge(table2, how='outer', on=['cust_id'])
    # df = df.reset_index(drop=True)
    # df = pd.concat([df, num2, num3, num4, num5, the_date, the_SAR], axis=1)
    # return df
    return flg


def get_data_date():
    if data_mode == 'train':
        data_date = train_date
    else:
        data_date = test_date
    return data_date


ls = []
data_date = get_data_date()
df = pd.DataFrame()
# for i in tqdm(range(len(data_date))):
for i in tqdm(range(500)):
    key, id, date, sar = give_event(idx=i, dateset=data_date)
    row = table(key=key, id=id, event_date=date, sar=sar)
    ls.append(row)
    # df = pd.concat([df, row])

breakpoint()

# df.to_csv('./testset_ex1.csv',index=False)
