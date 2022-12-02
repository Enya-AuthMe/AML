import pandas as pd
import numpy as np
from tqdm import tqdm
# from tools import find_month
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


class feature_tools:
    def find_month(date):
        mon = 0
        flag = 0
        month = [0, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 365]
        month_comp = [0, 30, 61, 91, 122, 153,
                      183, 214, 244, 275, 306, 334, 365]
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

    def find_today(id_table, date_tag: str, eventday: int, month: int):
        flag = 0
        today = 0
        for i in range(eventday, month-1, -1):
            if len(id_table[id_table[date_tag] == i]) != 0:
                today = i
                flag = 1
                break
        if flag == 0:
            today = eventday
        # flag=0 indicate before evnetday but in this mon there's no transaction record of this human
        return today, flag

    def get_thisday(id_table, date_tag: str, tag: str, today: int, discrete: bool, status: str):
        if discrete == True:
            if status == 'min':
                bool_val = True
            elif status == 'max':
                bool_val = False
            target = id_table.loc[id_table[date_tag] == today, tag].value_counts(
                ascending=bool_val).index.values[0]
        else:
            if status == 'mean':
                target = id_table.loc[id_table[date_tag]
                                      == today, tag].values.mean()
            elif status == 'total':
                target = id_table.loc[id_table[date_tag]
                                      == today, tag].values.sum()
            elif status == 'min':
                target = id_table.loc[id_table[date_tag]
                                      == today, tag].values.min()
            elif status == 'max':
                target = id_table.loc[id_table[date_tag]
                                      == today, tag].values.max()
        return target

    def id_table_past(id_table, date_tag: str, today: int):
        past_table = id_table[id_table[date_tag] < today]
        return past_table

    def get_history(id_table, date_tag: str, tag: str, today: int, discrete: bool, today_obj, status: str):
        last_table = id_table[id_table[date_tag] < today]
        if discrete:
            tag_history = last_table[tag].value_counts().index.values[0]
            if today_obj == tag_history:
                return 1
            else:
                return 0
        else:
            if status == 'mean':
                target = id_table.loc[id_table[date_tag]
                                      == today, tag].values.mean()
            elif status == 'min':
                target = id_table.loc[id_table[date_tag]
                                      == today, tag].values.min()
            elif status == 'max':
                target = id_table.loc[id_table[date_tag]
                                      == today, tag].values.max()
            return target

    def get_lastday(id_table, date_tag: str, tag: str, today: int, discrete: bool, status: str, today_obj):
        last_table = id_table[id_table[date_tag] < today]
        idx_sort = last_table.sort_values(
            date_tag, ascending=False).index.values[0]
        last_date = last_table.loc[idx_sort, date_tag]
        tag_lastday = feature_tools.get_thisday(
            last_table, date_tag, tag, last_date, discrete, status)
        if discrete:
            if today_obj == tag_lastday:
                return 1
            else:
                return 0
        else:
            return tag_lastday

    def get_last7day(id_table, date_tag: str, tag: str, today: int, status: str):
        last_table = id_table[id_table[date_tag] < today]
        if len(last_table) > 0:
            if today-7 < 0:
                if status == 'mean':
                    return last_table[tag].values.sum() / 7
                elif status == 'min':
                    return last_table[tag].values.min()
                elif status == 'max':
                    return last_table[tag].values.max()
            else:
                last_table = last_table[last_table[date_tag] >= (today-7)]
                if status == 'mean':
                    return last_table[tag].values.sum() / 7
                elif status == 'min':
                    return last_table[tag].values.min()
                elif status == 'max':
                    return last_table[tag].values.max()
        else:
            return np.nan

    def freq(id_table, date_tag: str, today: int):
        last_table = id_table[id_table[date_tag] < today]
        idx_sort = last_table.sort_values(
            date_tag, ascending=False).index.values[0]
        last_date = last_table.loc[idx_sort, date_tag]
        times = len(last_table[last_table['date'] == last_date])
        return times


def give_event(idx, dateset):
    key = dateset['alert_key'][idx]
    date = dateset['date'][idx]
    id = custinfo[custinfo['alert_key'] == key]['cust_id'].values[0]
    sar = y['sar_flag'][idx]
    return key, id, date, sar


def event_row(key, id, event_date, sar):
    # date SAR
    the_date = pd.DataFrame({'date': [event_date]})
    the_SAR = pd.DataFrame({'sar': [sar]})

    # 1 custinfo
    row1 = custinfo.loc[custinfo['alert_key'] == key, :]

    # 2 ccba
    mon = feature_tools.find_month(date=event_date)
    ccba_id = ccba.loc[ccba['cust_id'] == id, :]
    row2 = ccba_id[ccba_id['byymm'] == mon]

    if len(row2['usgam']) * len(row2['cycam']) == 0:
        row2['cuse_rate'] = np.nan
    else:
        row2['cuse_rate'] = row2['usgam'].values[0] / \
            row2['cycam'].values[0]

    # 3 cdtx
    cdtx_id = cdtx.loc[cdtx['cust_id'] == id, :]
    today, flg = feature_tools.find_today(cdtx_id, 'date', event_date, mon)
    col3 = ['country', 'country_last_day', 'country_history', 'cur_type', 'cur_type_last_day', 'cur_type_history',
            'amt_total', 'amt_avg', 'amt_min', 'amt_max', 'amt_avg_last_day', 'amt_min_last_day',
            'amt_max_last_day', 'amt_avg_last_7day', 'amt_min_last_7day', 'amt_max_last_7day',
            'amt_avg_history', 'amt_min_history', 'amt_max_history', 'amt_diff_avg_history',
            'amt_diff_min_history', 'amt_diff_max_history', 'cdtx_trans_num', 'cdtx_trans_num_last_day']

    arr = np.zeros((len(col3)))
    if len(cdtx_id.loc[cdtx_id['date'] == today]) != 0:
        arr[0] = feature_tools.get_thisday(
            cdtx_id, 'date', 'country', today, True, 'min')
        arr[3] = feature_tools.get_thisday(
            cdtx_id, 'date', 'cur_type', today, True, 'min')
        arr[6] = feature_tools.get_thisday(
            cdtx_id, 'date', 'amt', today, False, 'total')
        arr[7] = feature_tools.get_thisday(
            cdtx_id, 'date', 'amt', today, False, 'mean')
        arr[8] = feature_tools.get_thisday(
            cdtx_id, 'date', 'amt', today, False, 'min')
        arr[9] = feature_tools.get_thisday(
            cdtx_id, 'date', 'amt', today, False, 'max')
        arr[22] = len(cdtx_id[cdtx_id['date'] == today])

        id_past = feature_tools.id_table_past(cdtx_id, 'date', today)
        if len(id_past) != 0:

            arr[1] = feature_tools.get_lastday(
                cdtx_id, 'date', 'country', today, True, 'max', arr[0])
            arr[2] = feature_tools.get_history(
                cdtx_id, 'date', 'country', today, True, arr[0], np.nan)
            arr[4] = feature_tools.get_lastday(
                cdtx_id, 'date', 'cur_type', today, True, 'max', arr[3])
            arr[5] = feature_tools.get_history(
                cdtx_id, 'date', 'cur_type', today, True, arr[3], np.nan)

            arr[10] = feature_tools.get_lastday(
                cdtx_id, 'date', 'amt', today, False, 'mean', np.nan)
            arr[11] = feature_tools.get_lastday(
                cdtx_id, 'date', 'amt', today, False, 'max', np.nan)
            arr[12] = feature_tools.get_lastday(
                cdtx_id, 'date', 'amt', today, False, 'mean', np.nan)

            arr[13] = feature_tools.get_last7day(
                cdtx_id, 'date', 'amt', today, 'mean')
            arr[14] = feature_tools.get_last7day(
                cdtx_id, 'date', 'amt', today, 'max')
            arr[15] = feature_tools.get_last7day(
                cdtx_id, 'date', 'amt', today, 'min')

            arr[16] = feature_tools.get_history(
                cdtx_id, 'date', 'amt', today, False, np.nan, 'mean')
            arr[17] = feature_tools.get_history(
                cdtx_id, 'date', 'amt', today, False, np.nan, 'min')
            arr[18] = feature_tools.get_history(
                cdtx_id, 'date', 'amt', today, False, np.nan, 'max')
            arr[19] = arr[16] - arr[7]
            arr[20] = arr[17] - arr[7]
            arr[21] = arr[18] - arr[7]
            arr[23] = feature_tools.freq(cdtx_id, 'date', today)

        else:
            arr[[1, 2, 4, 5, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 23]] = np.nan
    else:
        arr[:] = np.nan
    row3 = pd.DataFrame(arr.reshape(-1, len(arr)), columns=col3)

    breakpoint()

    # 4 dp
    dp_id = dp.loc[dp['cust_id'] == id, :]
    today, flg = feature_tools.find_today(dp_id, 'date', event_date, mon)
    col4 = ['CR', 'DB', 'tx_type_info_asset_code', 'tx_type_asset_code_last_day', 'tx_type_info_asset_code_history', '?',
            'fiscTxId', 'fiscTxId_last_day', 'fiscTxId_history', 'txbranch', 'txbranch_history', 'ATM', 'dp__trans_num',
            'cdtx_trans_num_last_day'
            ]
    arr = np.zeros((len(col4)))
    
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
    row = event_row(key=key, id=id, event_date=date, sar=sar)
    ls.append(row)
    # df = pd.concat([df, row])

breakpoint()

# df.to_csv('./testset_ex1.csv',index=False)
