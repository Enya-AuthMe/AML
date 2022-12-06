import pandas as pd
import numpy as np
from tqdm import tqdm
import bisect
# from tools import find_month_firstday
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
    def find_month_firstday(date):
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

    def find_today_date(id_table, date_tag: str, eventday: int, month: int):
        flag = 0
        today_date = 0
        for i in range(eventday, month-1, -1):
            if len(id_table[id_table[date_tag] == i]) != 0:
                today_date = i
                flag = 1
                break
        if flag == 0:
            today_date = eventday
        # flag=0 indicate before evnetday but in this mon there's no transaction record of this human
        return today_date, flag

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
        past_table = id_table[id_table[date_tag] < today]
        if discrete:
            tag_history = past_table[tag].value_counts().index.values[0]
            if today_obj == tag_history:
                return 1
            else:
                return 0
        else:
            past_date = past_table[date_tag].value_counts().index.values
            target_arr = np.zeros((len(past_date)))
            for i, date in enumerate(past_date):
                target_arr[i] = past_table.loc[past_table[date_tag]
                                               == date, tag].mean()
            if status == 'mean':
                target = target_arr.mean()
            elif status == 'min':
                target = target_arr.min()
            elif status == 'max':
                target = target_arr.max()
            return target

    def get_lastday(id_table, date_tag: str, tag: str, today: int, discrete: bool, status: str, today_obj, num_lastday):
        past_table = id_table[id_table[date_tag] < today]
        idx_sort = past_table.sort_values(
            date_tag, ascending=False).index.values[0]
        last_date = past_table.loc[idx_sort, date_tag]
        tag_lastday = feature_tools.get_thisday(
            past_table, date_tag, tag, last_date, discrete, status)
        if discrete:
            if today_obj == tag_lastday:
                return 1
            else:
                return 0
        else:
            if len(past_table) > 0:  # 判斷有無過往值
                pastday_set = set(past_table[date_tag].to_list())
                if (len(pastday_set)-num_lastday) >= 0:  # 判斷過往值數量是否大於等於挑選日數
                    pastday_ls = list(pastday_set)
                    pastday_ls.sort(reverse=True)
                    end_day = pastday_ls[num_lastday-1]
                    past_table = past_table[past_table[date_tag] >= end_day]
                    pastday_avg_arr = np.zeros((num_lastday))
                    for i, pastdate in enumerate(pastday_ls[:num_lastday]):
                        day_avg = feature_tools.get_thisday(
                            past_table, date_tag, tag, pastdate, False, 'mean')
                        pastday_avg_arr[i] = day_avg
                    if status == 'mean':
                        target = pastday_avg_arr.mean()
                    elif status == 'min':
                        target = pastday_avg_arr.min()
                    elif status == 'max':
                        target = pastday_avg_arr.max()
                else:
                    target = np.nan
            else:
                target = np.nan
            return target

    def get_pastday(id_table, date_tag: str, tag: str, today: int, status: str, num_pastday):
        past_table = id_table[id_table[date_tag] < today]
        if (today - num_pastday) >= 0:  # 判斷目前日期減去選擇天數是否小於零
            past_table = past_table[past_table[date_tag]
                                    >= (today - num_pastday)]
            if len(past_table) > 0:  # 判斷過去天數內是否有交易紀錄
                past_date = past_table[date_tag].value_counts().index.values
                target_arr = np.zeros((len(past_date)))
                for i, date in enumerate(past_date):
                    target_arr[i] = past_table.loc[past_table[date_tag]
                                                   == date, tag].mean()
                if status == 'mean':
                    target = target_arr.mean()
                elif status == 'min':
                    target = target_arr.min()
                elif status == 'max':
                    target = target_arr.max()
            else:
                target = np.nan
        else:
            target = np.nan
        return target

    def freq(id_table, date_tag: str, today: int):
        past_table = id_table[id_table[date_tag] < today]
        last_date = past_table[date_tag].max()
        times = len(past_table[past_table[date_tag] == last_date])
        return times


def give_event(idx, dateset):
    key = dateset['alert_key'][idx]
    date = dateset['date'][idx]
    id = custinfo[custinfo['alert_key'] == key]['cust_id'].values[0]
    sar = y['sar_flag'][idx]
    return key, id, date, sar


def custinfo_row(key):
    row1 = custinfo.loc[custinfo['alert_key'] == key, :]
    row1.drop(columns=['cust_id'])
    return row1


def ccba_row(id, mon):
    ccba_id = ccba.loc[ccba['cust_id'] == id, :]
    row = ccba_id[ccba_id['byymm'] == mon]
    dict2 = {'cuse_rate': []}
    if len(row['usgam']) * len(row['cycam']) == 0:
        dict2['cuse_rate'] = np.nan
    else:
        dict2['cuse_rate'] = row['usgam'].values[0] / row['cycam'].values[0]
    cuse_rate = pd.DataFrame(dict2, index=[0])
    row = row.reset_index(drop=True)
    row2 = pd.concat([row, cuse_rate], axis=1)
    return row2


def cdtx_row(event_date, mon):
    cdtx_id = cdtx.loc[cdtx['cust_id'] == id, :]
    today, flg = feature_tools.find_today_date(
        cdtx_id, 'date', event_date, mon)
    col3 = ['country', 'country_last_1day', 'country_history',
            'cur_type', 'cur_type_last_1day', 'cur_type_history',
            'amt_total', 'amt_avg', 'amt_min', 'amt_max',
            'amt_avg_history', 'amt_min_history', 'amt_max_history',
            'amt_diff_avg_history', 'amt_diff_min_history', 'amt_diff_max_history',
            'cdtx_trans_num', 'cdtx_trans_num_last_day']

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
        arr[16] = len(cdtx_id[cdtx_id['date'] == today])

        id_past = feature_tools.id_table_past(cdtx_id, 'date', today)

        if len(id_past) != 0:
            arr[1] = feature_tools.get_lastday(
                cdtx_id, 'date', 'country', today, True, 'max', arr[0], 1)
            arr[2] = feature_tools.get_history(
                cdtx_id, 'date', 'country', today, True, arr[0], np.nan)
            arr[4] = feature_tools.get_lastday(
                cdtx_id, 'date', 'cur_type', today, True, 'max', arr[3], 1)
            arr[5] = feature_tools.get_history(
                cdtx_id, 'date', 'cur_type', today, True, arr[3], np.nan)

            arr[10] = feature_tools.get_history(
                cdtx_id, 'date', 'amt', today, False, np.nan, 'mean')
            arr[11] = feature_tools.get_history(
                cdtx_id, 'date', 'amt', today, False, np.nan, 'min')
            arr[12] = feature_tools.get_history(
                cdtx_id, 'date', 'amt', today, False, np.nan, 'max')
            arr[13] = arr[10] - arr[7]
            arr[14] = arr[11] - arr[8]
            arr[15] = arr[12] - arr[9]
            arr[17] = feature_tools.freq(cdtx_id, 'date', today)

        else:
            arr[[1, 2, 4, 5, 10, 11, 12, 13, 14, 15, 17]] = np.nan
    else:
        arr[:] = np.nan
    row3 = pd.DataFrame(arr.reshape(-1, len(arr)), columns=col3)
    return row3


def remit_row(event_date, mon):
    remit_id = remit.loc[remit['cust_id'] == id, :]
    today, flg = feature_tools.find_today_date(
        remit_id, 'trans_date', event_date, mon)
    dict5 = {'trans_no': [], 'trans_no_last_1day': [], 'trans_no_history': [],
             'trans_amt_usd_total': [], 'trans_amt_usd_avg': [], 'trans_amt_usd_min': [], 'trans_amt_usd_max': [],
             'trans_amt_usd_avg_history': [], 'trans_amt_usd_min_history': [], 'trans_amt_usd_max_history': [],
             'trans_amt_usd_diff_avg_history': [], 'trans_amt_usd_diff_min_history': [], 'trans_amt_usd_diff_max_history': [],
             'remit_trans_num': [], 'remit_trans_num_last_day': []}

    if len(remit_id.loc[remit_id['trans_date'] == today]) != 0:
        dict5['trans_no'] = feature_tools.get_thisday(
            remit_id, 'trans_date', 'trans_no', today, True, 'min')

        dict5['trans_amt_usd_total'] = feature_tools.get_thisday(
            remit_id, 'trans_date', 'trade_amount_usd', today, False, 'total')
        dict5['trans_amt_usd_avg'] = feature_tools.get_thisday(
            remit_id, 'trans_date', 'trade_amount_usd', today, False, 'mean')
        dict5['trans_amt_usd_min'] = feature_tools.get_thisday(
            remit_id, 'trans_date', 'trade_amount_usd', today, False, 'min')
        dict5['trans_amt_usd_max'] = feature_tools.get_thisday(
            remit_id, 'trans_date', 'trade_amount_usd', today, False, 'max')

        dict5['remit_trans_num'] = len(
            remit_id[remit_id['trans_date'] == today])

        id_past = feature_tools.id_table_past(remit_id, 'trans_date', today)
        if len(id_past) != 0:
            dict5['trans_no_last_1day'] = feature_tools.get_lastday(
                remit_id, 'trans_date', 'trans_no', today, True, 'max', dict5['trans_no'], 1)
            dict5['trans_no_history'] = feature_tools.get_history(
                remit_id, 'trans_date', 'trans_no', today, True, dict5['trans_no'], np.nan)

            dict5['trans_amt_usd_avg_history'] = feature_tools.get_history(
                remit_id, 'trans_date', 'trade_amount_usd', today, False, np.nan, 'mean')
            dict5['trans_amt_usd_min_history'] = feature_tools.get_history(
                remit_id, 'trans_date', 'trade_amount_usd', today, False, np.nan, 'min')
            dict5['trans_amt_usd_max_history'] = feature_tools.get_history(
                remit_id, 'trans_date', 'trade_amount_usd', today, False, np.nan, 'max')

            dict5['trans_amt_usd_diff_avg_history'] = dict5['trans_amt_usd_avg_history'] - \
                dict5['trans_amt_usd_avg']
            dict5['trans_amt_usd_diff_min_history'] = dict5['trans_amt_usd_min_history'] - \
                dict5['trans_amt_usd_min']
            dict5['trans_amt_usd_diff_max_history'] = dict5['trans_amt_usd_max_history'] - \
                dict5['trans_amt_usd_max']

            dict5['remit_trans_num_last_day'] = feature_tools.freq(
                remit_id, 'trans_date', today)
        else:
            dict5['trans_no_last_1day'] = np.nan
            dict5['trans_no_history'] = np.nan
            dict5['trans_amt_usd_avg_history'] = np.nan
            dict5['trans_amt_usd_min_history'] = np.nan
            dict5['trans_amt_usd_max_history'] = np.nan
            dict5['trans_amt_usd_diff_avg_history'] = np.nan
            dict5['trans_amt_usd_diff_min_history'] = np.nan
            dict5['trans_amt_usd_diff_max_history'] = np.nan
            dict5['remit_trans_num_last_day'] = np.nan
    else:
        dict5 = {x: np.nan for x in dict5}

    row5 = pd.DataFrame.from_dict(dict5, orient='index').T
    return row5


def event_row(key, id, event_date, sar):
    # date SAR
    the_date = pd.DataFrame({'date': [event_date]})
    the_SAR = pd.DataFrame({'sar': [sar]})

    mon = feature_tools.find_month_firstday(date=event_date)
    # 1 custinfo
    row1 = custinfo_row(key)
    row1 = row1.reset_index(drop=True)

    # 2 ccba
    row2 = ccba_row(id, mon)
    row2 = row2.reset_index(drop=True)

    # 3 cdtx
    row3 = cdtx_row(event_date, mon)
    row3 = row3.reset_index(drop=True)


    # 5 remi
    row5 = remit_row(event_date, mon)
    row5 = row5.reset_index(drop=True)

    df_row = pd.concat([row1, row2, row3, row5, the_date, the_SAR], axis=1)
    return df_row


def get_data_date():
    if data_mode == 'train':
        data_date = train_date
    else:
        data_date = test_date
    return data_date


ls = []
data_date = get_data_date()
df = pd.DataFrame()
for i in tqdm(range(len(data_date))):
# for i in tqdm(range(500)):
    key, id, date, sar = give_event(idx=i, dateset=data_date)
    row = event_row(key=key, id=id, event_date=date, sar=sar)
    df = pd.concat([df, row])
    df.reset_index(drop=True)

breakpoint()

# df.to_csv('./testset_ex1.csv',index=False)
