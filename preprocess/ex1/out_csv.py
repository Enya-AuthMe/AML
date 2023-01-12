import pandas as pd
from tqdm import tqdm

# original csv path
ccba = './訓練資料集_first/public_train_x_ccba_full_hashed.csv'
cdtx = './訓練資料集_first/public_train_x_cdtx0001_full_hashed.csv'
custinfo = './訓練資料集_first/public_train_x_custinfo_full_hashed.csv'
dp = './訓練資料集_first/public_train_x_dp_full_hashed.csv'
remit = './訓練資料集_first/public_train_x_remit1_full_hashed.csv'
date = './訓練資料集_first/public_x_alert_date.csv'
train_date = './訓練資料集_first/train_x_alert_date.csv'
y_path = './訓練資料集_first/train_y_answer.csv'

# read csv
ccba = pd.read_csv(ccba)
cdtx = pd.read_csv(cdtx)
custinfo = pd.read_csv(custinfo)
dp = pd.read_csv(dp)
remit = pd.read_csv(remit)
test_date = pd.read_csv(date)
train_date = pd.read_csv(train_date)
y = pd.read_csv(y_path)

# Control section
data_mode = 'test'  # 'train' or 'test'
breakpoint()


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


def table(key, id, date, sar):
    # date SAR
    the_date = pd.DataFrame({'date': [date]})
    the_SAR = pd.DataFrame({'sar': [sar]})

    # 1 custinfo
    table1 = custinfo[custinfo['alert_key'] == key]

    # 2 ccba
    mon = find_month(date=date)
    ccba_id = ccba[ccba['cust_id'] == id]
    table2 = ccba_id[ccba_id['byymm'] == mon]
    table2 = table2.drop(columns=['byymm'])
    num2 = pd.DataFrame({'ccba_num': [len(table2)]})

    # 3 cdtx
    cdtx_id = cdtx[cdtx['cust_id'] == id]
    table3 = cdtx_id[cdtx_id['date'] == date]
    num3 = pd.DataFrame({'cdtx_num': [len(table3)]})

    # 4 dp
    dp_id = dp[dp['cust_id'] == id]
    table4 = dp_id[dp_id['tx_date'] == date]
    num4 = pd.DataFrame({'dp_num': [len(table4)]})

    # 5 remi
    remit_id = remit[remit['cust_id'] == id]
    table5 = remit_id[remit_id['trans_date'] == date]
    num5 = pd.DataFrame({'remit_num': [len(table5)]})

    df = table1
    # df = df.merge(table2, how='outer', on=['cust_id'])
    df = df.reset_index(drop=True)
    df = pd.concat([df, num2, num3, num4, num5, the_date, the_SAR], axis=1)
    return df


def get_data_date():
    if data_mode == 'train':
        data_date = train_date
    else:
        data_date = test_date
    return data_date


data_date = get_data_date()
df = pd.DataFrame()
for i in tqdm(range(len(data_date))):
    key, id, date, sar = give_event(idx=i, dateset=data_date)
    row = table(key=key, id=id, date=date, sar=sar)
    df = pd.concat([df, row])
    breakpoint()

breakpoint()

# df.to_csv('./testset_ex1.csv',index=False)