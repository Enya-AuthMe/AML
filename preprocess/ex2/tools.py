import numpy as np
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

# def find_month(date):
#     month = [0, 30, 61, 91, 122, 153, 183, 214, 244, 275, 306, 334, 365]
#     mon = bisect.bisect(month, date) - 1
#     return mon

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
        target_arr = np.zeros((len(past_table[date_tag].value_counts().index)))
        if status == 'mean':
            for i, date in enumerate(past_table[date_tag]):
                target_arr[i] = past_table.loc[past_table[date_tag]==date, tag].mean()
            target = target_arr.mean()
        elif status == 'min':
            for i, date in enumerate(past_table[date_tag]):
                target_arr[i] = past_table.loc[past_table[date_tag]==date, tag].mean()
            target = target_arr.min()
        elif status == 'max':
            for i, date in enumerate(past_table[date_tag]):
                target_arr[i] = past_table.loc[past_table[date_tag]==date, tag].mean()
            target = target_arr.max()
        return target

def get_pastday(id_table, date_tag: str, tag: str, today: int, discrete: bool, status: str, today_obj, num_pastday):
    past_table = id_table[id_table[date_tag] < today]
    idx_sort = past_table.sort_values(
        date_tag, ascending=False).index.values[0]
    last_date = past_table.loc[idx_sort, date_tag]
    tag_lastday = get_thisday(
        past_table, date_tag, tag, last_date, discrete, status)
    if discrete:
        if today_obj == tag_lastday:
            return 1
        else:
            return 0
    else:
        if len(past_table) > 0:  # 判斷有無過往值
            pastday_set = set(past_table[date_tag].to_list())
            if (len(pastday_set)-num_pastday) >= 0:  # 判斷過往值數量是否大於等於挑選日數
                pastday_ls = list(pastday_set)
                pastday_ls.sort(reverse=True)
                end_day = pastday_ls[num_pastday-1]
                past_table = past_table[past_table[date_tag] >= end_day]
                pastday_avg_arr = np.zeros((num_pastday))
                for i, pastdate in enumerate(pastday_ls[:num_pastday]):
                    day_avg = get_thisday(
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

def get_lastday(id_table, date_tag: str, tag: str, today: int, discrete: bool, status: str, today_obj, num_pastday):
    past_table = id_table[id_table[date_tag] < today]
    idx_sort = past_table.sort_values(
        date_tag, ascending=False).index.values[0]
    last_date = past_table.loc[idx_sort, date_tag]

def freq(id_table, date_tag: str, today: int):
    past_table = id_table[id_table[date_tag] < today]
    idx_sort = past_table.sort_values(
        date_tag, ascending=False).index.values[0]
    last_date = past_table.loc[idx_sort, date_tag]
    times = len(past_table[past_table['date'] == last_date])
    return times